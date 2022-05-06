from Components import _QAParsing as qa
from Components import _KeywordSearch as kws
from Components import _KeywordDistance as kwd
from Components import _TFIDF as tfidf
from nltk.corpus import stopwords
import pandas as pd
import re

def solve_multiple_questions(filename):

    # Read in the dataframe of questions, see Data directory for example template.
    df = pd.read_csv(filename)
    metrics = {}
    
    for i, row in df.iterrows():

        # Set up the inital question reference information.
        question = row[0]
        unhandled_answers = row[1]
        correct_answer = row[2]
        search_answer, predicted_answer, distance_answer = ("", 0), ("", 0), ("", 0)
    
        # Take the input and parse the question to obtain keywords.
        A, B, C = qa.parseQuestion(question)
        answers, pos_answers, lower_answers = qa.parseAnswers(unhandled_answers)
        print("\nQuestion: " + question + "  ||  Answers: " + str(unhandled_answers))
        
        # Use the Named Entity Recognition Keywords and the Part of Speech Keywords to obtain the pages.
        NER_search = kws.searchNamedEntityRecognitionKeywordPages(A, lower_answers)
        POS_search = kws.searchPartOfSpeechKeywordPages(B, C, lower_answers)

        # Try using the Wikipedia search function to see if the answer can be obtained from a top-levelquery.
        is_solved, output = kws.searchForKeywordPages(NER_search, POS_search, lower_answers)

        # Check the results provided by the search, an answer may have been identified.
        if is_solved == 1 and output != '' and output:
            print("KEYWORD SEARCH RESULT FOUND\n" + ("-" * 50) + "\n\tIDENTIFIED ANSWER: " + str(output[0]).title())   # If answer identified, print and set anwer result.
            search_answer = (output[0].lower(), 100)
            continue
        
        # In the case that pages with the titles found that contained more than one answer, a targeted page corpus will be composed.
        elif is_solved == 2:
            keyword_corpus = tfidf.composeCorpusFromTargetedKeywordPages(NER_search, output[0], POS_search, output[1]) # Compose target corpus, build dataframe from corpus, obtain vocab and TF-IDF vector.

        # In the case that page titles did not contain any answer terms, a broad search will be conducted.
        elif is_solved == 0 or (is_solved == 1 and output == ''):
            keyword_corpus = tfidf.composeCorpusFromKeywordPages(output, lower_answers) # Compose target corpus, build dataframe from corpus, obtain vocab and TF-IDF vector.
    
        else:
            print("NO ANSWERS PROPERLY IDENTIFIED... EXITING")
            continue

        # Build the dataframe and TF-IDF information.
        keyword_df = tfidf.createKeywordCorpusDF(keyword_corpus)
        vocab, keyword_tfidf = tfidf.calculateKeywordCorpusTFIDF(keyword_df)

        # Use comparison points to obtain answers from TF-IDF vectors and distance measures between answers.
        if type(vocab) is not str:
            predicted_answer = tfidf.obtainAnswerFromTFIDF(vocab, keyword_tfidf, lower_answers)
        else:
            predicted_answer = tfidf.obtainAnswerFromTFIDF(str(vocab), keyword_tfidf, lower_answers)

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
            if score < 2: # Handle situation in which a single or no andwer is found
                try:
                    weighted_answers[a] = final_scores[answers.index(a)]
                except ValueError:
                    print("\nNo Search-Based answer found...")
            else:
                indices = [i for i, x in enumerate(answers) if x == a] # Obtain weighted scores of answers based on ones provided from keword, TFIDF, and distance.
                
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

        # Input metrics information to track the result of calculated answer.
        cleaned_correct_answer = re.sub("[^\s\w-]", "", correct_answer)
        print(str(suggested_answer) + ":" + cleaned_correct_answer.lower()) # Provide the suggested answer and the correct answer.
        print(("_" * 100) + "\n" )
        if suggested_answer in cleaned_correct_answer.lower():
            metrics[question] = ["CORRECT", suggested_answer, str(weighted_answers[suggested_answer]), str(answers.count(suggested_answer)), correct_answer]
        elif suggested_answer == "NONE":
            metrics[question] = ["INCORRECT", suggested_answer, "0", "3", correct_answer]
        else:
            metrics[question] = ["INCORRECT", suggested_answer, str(weighted_answers[suggested_answer]), str(answers.count(suggested_answer)), correct_answer]

    # Initialize metrics information
    correct_count = 0
    incorrect_count = 0
    total_score_correct = 0
    total_score_incorrect = 0

    # Calculate incorrect, correct, and total score counts.
    for x, y in metrics.items():
        print("Result: " + y[0] + "  ||  " + "Question: " + x + "  ||  Suggested Answer: " + y[1] + "  ||  Confidence Score: " + y[2]  + "%  ||  Supporting Identifiers: " + y[3] + "  ||  Correct Answer: " + y[4])
        if y[0] == "CORRECT":
            correct_count = correct_count + 1
            total_score_correct = float(y[2]) + total_score_correct
        if y[0] == "INCORRECT":
            incorrect_count = incorrect_count + 1
            total_score_incorrect = float(y[2]) + total_score_incorrect

    # Print out some results.
    print("\n\nOVERALL RESULTS" + "\n" + ("-" * 50))
    
    print("\tTotal Correct: " + str(correct_count) + " (" + str(round(100 * (correct_count/(correct_count+incorrect_count)),2)) + "%)")
    if correct_count != 0:
        print("\tAverage Correct Confidence Score: " + str(round(.1 * (total_score_correct/correct_count), 2)) + "%")
    else:
        print("\tAverage Correct Confidence Score: 0.0%")
    
    print("\tTotal Incorrect: " + str(incorrect_count) + " (" + str(round(100 * (incorrect_count/(incorrect_count+correct_count)),2)) + "%)")
    if incorrect_count != 0:
        print("\tAverage Incorrect Confidence Score: " + str(round(.1 * total_score_incorrect/incorrect_count)) + "%")
    else:
        print("\tAverage Incorrect Confidence Score: 0.0%")

def main():

    # Print out a cool banner!
    print( "\n" + ("-" * 50) + "\nTrivial Assassin Mission V3.2\n" + ("-" * 50))

    print("Trivial Assassin Mission is a method to test the assassin against a list of given questions, answers, and the correct answer. Examples have been provided within the data directory.")


    # Obtain the question from the user, retrieve priority-sorted list of tagged terms used for searching for a wikipedia page.
    filename = input("Please enter a file path pointing to a CSV file containing the list of questions to test with:\n")

    solve_multiple_questions(filename)

main()