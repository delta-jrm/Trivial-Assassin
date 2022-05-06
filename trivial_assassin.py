from Components import _QAParsing as qa
from Components import _KeywordSearch as kws
from Components import _KeywordDistance as kwd
from Components import _TFIDF as tfidf
from nltk.corpus import stopwords

def solve_question(question, unhandled_answers):

    # Set up the inital answer reference information.
    search_answer, predicted_answer, distance_answer = ("", 0), ("", 0), ("", 0)

    # Take the input and parse the question to obtain keywords.
    A, B, C = qa.parseQuestion(question)
    answers, pos_answers, lower_answers = qa.parseAnswers(unhandled_answers)
    print("\nQuestion: " + question + "  ||  Answers: " + str(unhandled_answers))

    # Use the Named Entity Recognition Keywords and the Part of Speech Keywords to obtain the pages.
    NER_search = kws.searchNamedEntityRecognitionKeywordPages(A, lower_answers)
    POS_search = kws.searchPartOfSpeechKeywordPages(B, C, lower_answers)

    # Try using the Wikipedia search function to see if the answer can be obtained from a top-levelquery.
    is_solved, output = kws.searchForKeywordPages(
        NER_search, POS_search, lower_answers)

    # Check the results provided by the search, an answer may have been identified.
    if is_solved == 1 and output != '' and output:
        # If answer identified, print and set anwer result.
        print("KEYWORD SEARCH RESULT FOUND\n" + ("-" * 50) + "\n\tIDENTIFIED ANSWER: " + str(output[0]).title())
        search_answer = (output[0].lower(), 100)
        keyword_corpus = tfidf.composeCorpusFromKeywordPages(output, lower_answers)

    # In the case that pages with the titles found that contained more than one answer, a targeted page corpus will be composed.
    elif is_solved == 2:
        # Compose target corpus, build dataframe from corpus, obtain vocab and TF-IDF vector.
        keyword_corpus = tfidf.composeCorpusFromTargetedKeywordPages(
            NER_search, output[0], POS_search, output[1])

    # In the case that page titles did not contain any answer terms, a broad search will be conducted.
    elif is_solved == 0 or (is_solved == 1 and output == ''):
        # Compose target corpus, build dataframe from corpus, obtain vocab and TF-IDF vector.
        keyword_corpus = tfidf.composeCorpusFromKeywordPages(output, lower_answers)

    else:
        print("NO ANSWERS PROPERLY IDENTIFIED... EXITING")
        exit()

    # Build the dataframe and TF-IDF information.
    keyword_df = tfidf.createKeywordCorpusDF(keyword_corpus)
    vocab, keyword_tfidf = tfidf.calculateKeywordCorpusTFIDF(keyword_df)

    # Use comparison points to obtain answers from TF-IDF vectors and distance measures between answers.
    if type(vocab) is not str:
        predicted_answer = tfidf.obtainAnswerFromTFIDF(
            vocab, keyword_tfidf, lower_answers)
    else:
        predicted_answer = tfidf.obtainAnswerFromTFIDF(
            str(vocab), keyword_tfidf, lower_answers)

    # Print answer.
    print("\n\tIDENTIFIED ANSWER FROM TF-IDF SEARCH: " + str(predicted_answer).title())

    # Search through the docs and find the ones containing answers.
    ps, hp = kwd.obtainHighPriorityDocs(A, B, C, pos_answers, keyword_corpus)

    # Take the identified docs and try to obtain an answer and print.
    distance_answer = kwd.obtainBestGuessFromDF(ps, hp)
    print("\n\tIDENTIFIED ANSWER FROM KEYWORD-TO-ANSWER DISTANCE COMPARISON: " + str(distance_answer).title())

    # Take final scores and answers and split them into two lists.
    final_scores = [search_answer[1], predicted_answer[1], distance_answer[1]]
    answers = [search_answer[0], predicted_answer[0], distance_answer[0]]

    # Determine answer occurence count, and take the average between scores with multiple occurences.
    weighted_answers = {}
    for a in answers:
        if a == "":  # Skip answer if empty
            continue

        score = answers.count(a)
        if score < 2:  # Handle situation in which a single or no andwer is found
            try:
                weighted_answers[a] = final_scores[answers.index(a)]
            except ValueError:
                print("\nNo Search-Based answer found...")
        else:
            # Obtain weighted scores of answers based on ones provided from keword, TFIDF, and distance.
            indices = [i for i, x in enumerate(answers) if x == a]

            weighted_answers[a] = 0
            sc = 0
            for j in indices:
                cs = weighted_answers[answers[j]]
                sc = final_scores[j]
                weighted_answers[answers[j]] = sc + cs
            weighted_answers[a] = round(weighted_answers[a] / len(indices), 2)

    try:
        # Provide a suggested answer (the one with the maximum average score)
        suggested_answer = max(weighted_answers, key=weighted_answers.get)
        print("\n\nTOP ANSWER SELECTION: " + suggested_answer.title() + "  ||  " + "CONFIDENCE: " + str(weighted_answers[suggested_answer]) + "% LIKELIHOOD  ||  " + "METHODS SUPPORTING CHOICE: " + str(answers.count(suggested_answer)))

    except:
        suggested_answer = "NONE"
        print("\n\nTOP ANSWER SELECTION: " + suggested_answer.title() + "  ||  " + "CONFIDENCE: 0% LIKELIHOOD  ||  " + "METHODS SUPPORTING CHOICE: 3")

def main():

    # Print out a cool banner!
    print( "\n" + ("-" * 50) + "\nTrivial Assassin V3.2\n" + ("-" * 50))

    # Obtain the question from the user, retrieve priority-sorted list of tagged terms used for searching for a wikipedia page.
    question = input("Please enter a trivia question:\n")

    # Obtain a list of potential answers, retrieve the list of POS-tagged answers.
    unhandled_answers = input("\nPlease enter a comma-seperated list of potential answers (ex. red, blue, green):\n")

    solve_question(question, unhandled_answers)

main()