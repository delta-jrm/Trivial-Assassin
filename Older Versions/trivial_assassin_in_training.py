from dis import dis
from operator import indexOf
import sys
from matplotlib.pyplot import summer
import requests
import bs4
from pyfiglet import figlet_format
import wikipedia
import json
import re
import nltk
from nltk.corpus import stopwords
import spacy
from spacy import displacy
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import numpy as np
import itertools

DETERMINERS = ['who', 'what', 'when', 'where', 'why', 'how', 'which', 'whose']

class Answer:
    def __init__(self, count, answers):
        self.count = count
        self.answers = answers

def parseQuestion(q):

    # Configure the stopword reference to not filter out question words.
    stop_words_with_determiners = set(stopwords.words('english'))  # Initialize stopwords
    stop_words_without_determiners = [w for w in stop_words_with_determiners if not w.lower() in DETERMINERS]

    # Remove special characters from the question and tokenize it.
    q_final = []
    q_nospec = re.sub("[^\s\w-]", "", q)  # Remove special characters
    q_tokens = nltk.word_tokenize(q_nospec) # Tokenize sentence
    q_toktag = nltk.pos_tag(q_tokens) # Tag it

    # Run Named Entity Recognition to identify keywords without reliance on tagging.
    nlp = spacy.load('en_core_web_sm')
    sentence_doc = nlp(q_nospec)

    # Remove stopwords from the question.
    q_filtered = [w for w in q_tokens if not w.lower() in stop_words_without_determiners]
    q_final = []
    for w in q_tokens:
        if w not in stop_words_without_determiners:
            q_final.append(w.strip())

    # Tag the question.
    q_tagged = nltk.pos_tag(q_final)

    # Filter terms by tag to sort by priority for lower-priority non NER searches.
    q_priority_B = [(w, t) for (w, t) in q_tagged if re.match("NNP|CD|VBP", t)]     # Priority B includes proper nouns, used as a source to ideally pull wiki pages from.
    q_priority_C = [(w, t) for (w, t) in q_tagged if re.match("(NN[^P]{,1}|RB|EX|JJ)", t) and (w, t) not in q_priority_B] # Priority C is noun keywords which may provide more specific context as to what will be looked for in the returned wikipedia documents.

    # Select terms using NER.
    q_priority_A_untagged = []
    for chunk in sentence_doc.noun_chunks:
        if (str(chunk) not in q_priority_B) and (str(chunk) not in q_priority_C):
            q_priority_A_untagged.append(str(chunk))
    q_priority_A = nltk.pos_tag(q_priority_A_untagged)  # Priority A includes named entities, used as a source to ideally pull wiki pages from.

    # Return question keywords by priority.
    return q_priority_A, q_priority_B, q_priority_C


def parseAnswers(a):
    # Get rid of comma + space combo.
    cleaned = a.replace(", ", ",").split(",")

    parsed = []
    for a in cleaned:
        nos = re.sub("'s", "", a)
        nospec = re.sub("[^\s\w-]", "", nos)
        parsed.append(nospec)

    # POS tag it.
    ans = nltk.pos_tag(parsed)

    # Return the cleaned list and the POS-tagged one.
    return parsed, ans


def searchNamedEntityRecognitionKeywordPages(A):
    
    # Put together a string of keywords to search with.
    NER = ""
    for word, tag in A:
        NER = NER + word + " "
    
    # Perform a search using the priority-A keywords identified in question-parsing.
    NER_search = wikipedia.search(NER)

    # Return search results.
    return NER_search


def searchPartOfSpeechKeywordPages(B, C):
   
    # Put together a string of keywords to search with.
    POS = ""
    for word, tag in B:
        if word not in POS:
            POS = POS + word + " "
    for word, tag in C:
        if word not in POS:
            POS = POS + word + " "
    
    # Perform a search using the priority B & C keywords identified in question-parsing.
    POS_search = wikipedia.search(POS) 

    # Return search results.
    return POS_search


def searchForKeywordPages(NER, POS, answers):

    # Initialize the solved state as 0 || Solved States are as follows: 0 = Unsolved with no answers
    # identified in search results, 1 = Solved with answers identified in search results, 2 =
    # Unsolved with multiple answers identified in search results.
    solved_state = 0
    POS_answer = Answer(0, {})
    NER_answer = Answer(0, {})

    NER_compare = " ".join(NER)
    POS_compare = " ".join(POS)

    NER_lower = []
    for title in NER:
        NER_lower.append(title.lower())

    POS_lower = []
    for title in POS:
        POS_lower.append(title.lower())

    # Identify any answers colliding with returned search results in POS and NER
    for a in answers:
        if a in NER_compare.lower():
            NER_answer.count = NER_answer.count + 1
            try:
                NER_answer.answers[a] = NER_lower.index(a)
            except:
                for item in NER:
                    if a in item.lower():
                        NER_answer.answers[a] = NER.index(item)
    for a in answers:
        if a in POS_compare.lower():
            POS_answer.count = POS_answer.count + 1
            try:
                POS_answer.answers[a] = POS.index(a)
            except:
                for item in POS:
                    if a in item.lower():
                        POS_answer.answers[a] = POS.index(item)

    # Nothing found, solved state 0.
    if NER_answer.count == 0 and POS_answer.count == 0:
        return solved_state, set(NER + POS)

    # Answer found, solved state 1.
    elif NER_answer.count == 1 and POS_answer.count == 0:
        solved_state = 1
        return solved_state, list(NER_answer.answers.keys())

    # Answer found, solved state 1.
    elif NER_answer.count == 0 and POS_answer.count == 1:
        solved_state = 1
        return solved_state, list(POS_answer.answers.keys())

    # Potential answer found, remove duplicates and resolve to state 1 or 2 accordingly.
    elif NER_answer.count == 1 and POS_answer.count == 1:
        if NER_answer.answers.keys() == POS_answer.answers.keys():
            solved_state = 1
            return solved_state, list(POS_answer.answers.keys())

        else:
            solved_state = 2
            return solved_state, [NER_answer.answers, POS_answer.answers]

    # Multiple answers found, solved state 2.
    elif NER_answer.count >= 1 and POS_answer.count >= 1:
        solved_state = 2

        duplicates = []
        for ans in NER_answer.answers.keys():
            if ans in POS_answer.answers.keys():
                duplicates.append(ans)

        for d in duplicates:
            NER_answer.answers.pop(d, None)

        return solved_state, [NER_answer.answers, POS_answer.answers]

    else:
        solved_state = -1
        return solved_state, "ERROR"


def cleanKeywordPage(content):

    # Get rid of special characters, return result.
    nospec = re.sub("[^\s\w-]", "", content)  # Remove special characters
    return nospec


def composeCorpusFromKeywordPages(keywordpages):

    # Establish an empty results dictionary.
    results = {}

    # For each page identified in keywords, obtain the title and the content, add it to results.
    for pg in keywordpages:
        try:
            reference = wikipedia.page(pg, auto_suggest=False)
            title = reference.title
            text = cleanKeywordPage(reference.content)
            results[title] = text
        except wikipedia.DisambiguationError:
            continue
        except wikipedia.PageError:
            continue

    # Return the corpus of documents.
    return results


def composeCorpusFromTargetedKeywordPages(NER, NER_answers, POS, POS_answers):

    # Establish an empty results dictionary.
    results = {}

    # For each page identified in keywords, obtain the title and the content, add it to results.
    for idx in NER_answers.values():
        try:
            reference = wikipedia.page(NER[idx], auto_suggest=False)
            title = reference.title
            text = cleanKeywordPage(reference.content)
            results[title] = text
        except wikipedia.DisambiguationError:
            continue
        except wikipedia.PageError:
            continue

    for idx in POS_answers.values():
        try:
            reference = wikipedia.page(POS[idx], auto_suggest=False)
            title = reference.title
            text = cleanKeywordPage(reference.content)
            results[title] = text
        except wikipedia.DisambiguationError:
            continue
        except wikipedia.PageError:
            continue

    # Return the corpus of documents.
    return results


def createKeywordCorpusDF(keywordcorpus):

    # Build the dataframe.
    kw_df = pd.DataFrame(keywordcorpus.items(), columns=['keyword', 'text'])

    # Add a column with the tokenized version of the text.
    kw_df['tokenized'] = kw_df['text'].apply(nltk.word_tokenize)

    # Take action to remove any remaining stopwords from tokenized and add a new column of the text without stop words in.
    stop_words = set(stopwords.words('english'))
    kw_df['tokenized'] = kw_df['tokenized'].apply(
        lambda x: [word for word in x if word not in stop_words])
    kw_df['text_no_stopwords'] = [' '.join(map(str, l)) for l in kw_df['tokenized']]
    return kw_df


def calculateKeywordCorpusTFIDF(keyworddf):

    # Set up Vectorizer for TFIDF with tuning to account for MBTI corpus conditions
    kw_vector = TfidfVectorizer(  # max_df=0.40,         # Drop words that appear more than X%, currently unused in favor of min.
        min_df=2,             # only use words that appear at least X times
        stop_words='english',  # remove stop words
        lowercase=True,        # Convert everything to lower case
        use_idf=True,          # Use idf
        norm=u'l2',            # Normalization
        smooth_idf=True        # Prevents divide-by-zero errors
    )

    # Parse the entire set of keyword-relate documents into TF-IDF.
    text = keyworddf['text_no_stopwords']
    try:
        ttfidf = kw_vector.fit_transform(text)
    except:
        return "NONE", "NONE"
    tfidf_array = ttfidf.toarray()

    # Set up vocabulary and final tf_idf dataframe.
    vocab = kw_vector.get_feature_names_out()
    tfidf_df = pd.DataFrame(np.round(tfidf_array, 6), columns=vocab)
    return vocab, tfidf_df


def obtainAnswerFromTFIDF(vocab, keywordTFIDF, answers):
    final_answer = ""

    # Combine all the counts to sum total term counts across every document.
    summed_series = keywordTFIDF.sum(axis=0)
    vals_tfidf = summed_series.values
    idx_tfidf = np.argsort(vals_tfidf)
    top_tfidf = idx_tfidf.flatten()

    # Find answer occurences in TF-IDF vector and map to the corresponding score.
    answers_tfidf = {}
    for entry in reversed(top_tfidf):
        if vocab[entry].lower() in answers:
            answers_tfidf[vocab[entry]] = summed_series[entry]
    
    print("\nTF-IDF Answer Identification\n" + ("-" * 50))
    total = sum(answers_tfidf.values())
    for w,v in sorted(answers_tfidf.items(), key=lambda kv: kv[1], reverse=True):
        score = v/total
        print("\t" + w.title() + "  -  SCORE: " + str(round(score*100, 2)) + "% LIKELIHOOD")

    # Get the max value for the final answer, show probability scores indicating term importance.
    try:
        final_answer = max(answers_tfidf, key=answers_tfidf.get)
        return final_answer, round((answers_tfidf[final_answer]/total)*100, 2)
    except ValueError:
        return final_answer, 0

# Utilized external driver code for the frame of this function from GeeksForGeeks. Credit: https://www.geeksforgeeks.org/minimum-distance-between-words-of-a-string/
def obtainWordDistance(keyword, refword, text_list):

        if keyword == refword:
            return 0

        min_dist = len(text_list) + 1
    
        for index in range(len(text_list)):
    
            if text_list[index] == keyword:
                for search in range(len(text_list)):
    
                    if text_list[search] == refword:
                        curr = abs(index - search) - 1;

                        if curr < min_dist:
                            min_dist = curr
    
            return min_dist


def obtainBestGuessFromDF(A, B, C, answers, keyworddict):

    # Get a list of terms from the keywords identified and the answers provided, used for removing tags.
    all_tagged = A + B + C + answers
    term_tagged = A + B + C

    # Get term list including answers without tags.
    all_terms = []
    for word,tag in all_tagged:
        if word not in all_terms:
            all_terms.append(word)
    
    # Get term list of just keyword terms without tags.
    terms = []
    for word,tag in term_tagged:
        if word not in terms:
            terms.append(word)
    
    # Get answer list without tags.
    ans = []
    for word,tag in answers:
        if word not in ans:
            ans.append(word)
    
    # Identify documents with the matching keywords and answers. Place them into high-priority dictionary
    high_priority = {}
    priority_scores = {}
    if type(keyworddict) is dict:
        for title,text in keyworddict.items():
            term_score = 0
            ans_score = 0
            ref_ans = []
            ref_term = []  
            for a in ans:
                try:
                    tlower = text.lower()
                except:
                    tlower = str(text.str.lower())
                if a.lower() in tlower:
                    ans_score = ans_score + 1
                    ans_count = text.count(a)
                    ref_ans.append((a.lower(), ans_count))

            for t in terms:
                if t.lower() in tlower:
                    term_score = term_score + 1
                    ref_term.append(t.lower())

            if ans_score >= 1:
                priority_scores[title] = ans_score + term_score
                high_priority[title] = [ref_ans, ref_term, tlower]

    try:
        answer_doc = max(priority_scores, key=priority_scores.get)
    except:
        return '', 0
    ref_doc = high_priority[answer_doc][2].split(" ")
    hp_ans = high_priority[answer_doc][0]
    hp_term = high_priority[answer_doc][1]

    if len(hp_ans) == 1:
        dist_answer = hp_ans[0][0]
        print("\nKeyword-to-Answer Distance Identification\n" + ("-" * 50))
        print("\t" + dist_answer.title() + "  -  SCORE: 100% LIKELIHOOD")
    
    elif len(hp_ans) > 1:
        distance_measure = {}
        answer_counts = {}
        for a,s in hp_ans:
            answer_counts[a] = s
            distcount = 0
            for b in hp_term:
                dist = obtainWordDistance(a, b, ref_doc)
                distcount = distcount + dist
            distance_measure[a] = distcount
        
        dist_scores = {}
        total_scores = {}

        total_dist = sum(distance_measure.values())
        total_ans = sum(answer_counts.values())
        if total_dist == 0:
            total_dist = 1
        print("\nKeyword-to-Answer Distance Identification\n" + ("-" * 50))
        for w,v in distance_measure.items():
            dist_score = v/total_dist
            dist_scores[w] = dist_score
        
        if total_ans == 0:
            total_ans = 1
        for w,v in answer_counts.items():
            count_score = v/total_ans
            total_scores[w] = count_score * dist_scores[w]
        
        tally = sum(total_scores.values())
        if tally == 0:
            tally = 1
        for w,v in sorted(total_scores.items(), key=lambda kv: kv[1], reverse=True):
            score = v/tally
            print("\t" + w.title() + "  -  SCORE: " + str(round(score*100, 2)) + "% LIKELIHOOD")
        dist_answer = max(total_scores, key=total_scores.get)
    
    try:
        return dist_answer, round((total_scores[dist_answer]/tally)*100, 2)
    except:
        return dist_answer, 0



def main():

    # Print out a cool banner!
    print("Trivial Assassin V1.0\n" + ("-" * 50))

    # Obtain the question from the user, retrieve priority-sorted list of tagged terms used for searching for a wikipedia page.
    question = input("Please enter a trivia question:\n")
    A, B, C = parseQuestion(question)

    # Obtain a list of potential answers, retrieve the list of POS-tagged answers.
    unhandled_answers = input("\nPlease enter a comma-seperated list of potential answers (ex. red, blue, green):\n")
    answers, pos_answers = parseAnswers(unhandled_answers)
    lower_answers = []
    for a in answers:
        lower_answers.append(a.lower())

    # Try using the Wikipedia search function to see if the answer can be obtained from a top-levelquery.
    NER_search = searchNamedEntityRecognitionKeywordPages(A)
    POS_search = searchPartOfSpeechKeywordPages(B, C)
    is_solved, output = searchForKeywordPages(NER_search, POS_search, lower_answers)
    print("\nSuggested Terms Found: " + ", ".join(output) + "...\n")

    search_answer, predicted_answer, distance_answer = ("", 0), ("", 0), ("", 0)
    # Check the results provided by the search, an answer may have been identified.
    if is_solved == 1:
        print("Answers Contained in Search Identification\n" + ("-" * 50))
        print("\nIDENTIFIED ANSWER FROM KEYWORD SEARCH: " + str(output[0]).title())
        search_answer = (output[0].lower(), 100)
    
    # In the case that pages with the titles found that contained more than one answer, a targeted page corpus will be composed.
    elif is_solved == 2:

        # Compose target corpus, build dataframe from corpus, obtain vocab and TF-IDF vector.
        keyword_corpus = composeCorpusFromTargetedKeywordPages(NER_search, output[0], POS_search, output[1])
        keyword_df = createKeywordCorpusDF(keyword_corpus)
        vocab, keyword_tfidf = calculateKeywordCorpusTFIDF(keyword_df)

        # Use comparison points to obtain answers from TF-IDF vectors and distance measures between answers.
        predicted_answer = obtainAnswerFromTFIDF(vocab, keyword_tfidf, lower_answers)
        print("\nIDENTIFIED ANSWER FROM TF-IDF SEARCH: " + str(predicted_answer).title())
        distance_answer = obtainBestGuessFromDF(A, B, C, pos_answers, keyword_df)
        print("\nIDENTIFIED ANSWER FROM KEYWORD-TO-ANSWER DISTANCE COMPARISON: " + str(distance_answer).title())

    # In the case that page titles did not contain any answer terms, a broad search will be conducted.
    elif is_solved == 0:

        # Compose target corpus, build dataframe from corpus, obtain vocab and TF-IDF vector.
        keyword_corpus = composeCorpusFromKeywordPages(output)
        keyword_df = createKeywordCorpusDF(keyword_corpus)
        vocab, keyword_tfidf = calculateKeywordCorpusTFIDF(keyword_df)

        # Use comparison points to obtain answers from TF-IDF vectors and distance measures between answers.
        predicted_answer = obtainAnswerFromTFIDF(vocab, keyword_tfidf, lower_answers)
        print("\nIDENTIFIED ANSWER FROM TF-IDF SEARCH: " + str(predicted_answer).title())
        distance_answer = obtainBestGuessFromDF(A, B, C, pos_answers, keyword_corpus)
        print("\nIDENTIFIED ANSWER FROM KEYWORD-TO-ANSWER DISTANCE COMPARISON: " + str(distance_answer).title())

    else:
        print("NO ANSWERS PROPERLY IDENTIFIED... EXITING")
        exit()

    # Take final scores and answers and split them into two lists.
    final_scores = [search_answer[1], predicted_answer[1], distance_answer[1]]
    answers = [search_answer[0], predicted_answer[0], distance_answer[0]]

    # Determine answer occurence count, and take the average between scores with multiple occurences.
    weighted_answers = {}
    for a in answers:
        if a == "":
            continue

        score = answers.count(a)
        if score < 2:
            try:
                weighted_answers[a] = final_scores[answers.index(a)]
            except ValueError:
                print("\nNo Search-Based answer found...")
        else:
            indices = [i for i, x in enumerate(answers) if x == a]
            
            weighted_answers[a] = 0
            sc = 0
            for j in indices:
                cs = weighted_answers[answers[j]]
                sc = final_scores[j]
                weighted_answers[answers[j]] = sc + cs
            weighted_answers[a] = round(weighted_answers[a] / len(indices), 2)


    suggested_answer = max(weighted_answers, key=weighted_answers.get)
    print("\n\nTOP ANSWER SELECTION: " + suggested_answer.title() + "  ||  " + "CONFIDENCE: " + str(weighted_answers[suggested_answer]) + "% LIKELIHOOD  ||  " + "METHODS SUPPORTING CHOICE: " + str(answers.count(suggested_answer)))
    print()

def test_main(filename):
    df = pd.read_csv(filename)
    metrics = {}

    for i, row in df.iterrows():
        question = row[0]
        unhandled_answers = row[1]
        correct_answer = row[2]
    
        A, B, C = parseQuestion(question)
        answers, pos_answers = parseAnswers(unhandled_answers)
        print("\n\nQuestion: " + question + "  ||  Answers: " + str(unhandled_answers))
        lower_answers = []
        for a in answers:
            lower_answers.append(a.lower())

        # Try using the Wikipedia search function to see if the answer can be obtained from a top-levelquery.
        NER_search = searchNamedEntityRecognitionKeywordPages(A)
        POS_search = searchPartOfSpeechKeywordPages(B, C)
        is_solved, output = searchForKeywordPages(NER_search, POS_search, lower_answers)
        try:
            print("\tSuggested Terms Found: " + ", ".join(output) + "...\n")
        except:
            print("\tIdentified output terms...")

        search_answer, predicted_answer, distance_answer = ("", 0), ("", 0), ("", 0)
        # Check the results provided by the search, an answer may have been identified.
        if is_solved == 1 and output != '':
            print("Answers Contained in Search Identification\n" + ("-" * 50))
            print("\nIDENTIFIED ANSWER FROM KEYWORD SEARCH: " + str(output[0]).title())
            search_answer = (output[0].lower(), 100)
        
        # In the case that pages with the titles found that contained more than one answer, a targeted page corpus will be composed.
        elif is_solved == 2:

            # Compose target corpus, build dataframe from corpus, obtain vocab and TF-IDF vector.
            keyword_corpus = composeCorpusFromTargetedKeywordPages(NER_search, output[0], POS_search, output[1])
            keyword_df = createKeywordCorpusDF(keyword_corpus)
            vocab, keyword_tfidf = calculateKeywordCorpusTFIDF(keyword_df)

            # Use comparison points to obtain answers from TF-IDF vectors and distance measures
            # between answers.
            if type(vocab) is not str:
                predicted_answer = obtainAnswerFromTFIDF(vocab, keyword_tfidf, lower_answers)
            else:
                predicted_answer = ("", 0)
            print("\nIDENTIFIED ANSWER FROM TF-IDF SEARCH: " + str(predicted_answer).title())


            distance_answer = obtainBestGuessFromDF(A, B, C, pos_answers, keyword_corpus)
            print("\nIDENTIFIED ANSWER FROM KEYWORD-TO-ANSWER DISTANCE COMPARISON: " + str(distance_answer).title())

        # In the case that page titles did not contain any answer terms, a broad search will be conducted.
        elif is_solved == 0 or (is_solved == 1 and output == ''):

            # Compose target corpus, build dataframe from corpus, obtain vocab and TF-IDF vector.
            keyword_corpus = composeCorpusFromKeywordPages(output)
            keyword_df = createKeywordCorpusDF(keyword_corpus)
            vocab, keyword_tfidf = calculateKeywordCorpusTFIDF(keyword_df)

            # Use comparison points to obtain answers from TF-IDF vectors and distance measures between answers.
            if type(vocab) is not str:
                predicted_answer = obtainAnswerFromTFIDF(vocab, keyword_tfidf, lower_answers)
            else:
                predicted_answer = ("", 0)
            print("\nIDENTIFIED ANSWER FROM TF-IDF SEARCH: " + str(predicted_answer).title())
            distance_answer = obtainBestGuessFromDF(A, B, C, pos_answers, keyword_corpus)
            print("\nIDENTIFIED ANSWER FROM KEYWORD-TO-ANSWER DISTANCE COMPARISON: " + str(distance_answer).title())
    
        else:
            print("NO ANSWERS PROPERLY IDENTIFIED... EXITING")
            continue
        # Take final scores and answers and split them into two lists.
        final_scores = [search_answer[1], predicted_answer[1], distance_answer[1]]
        answers = [search_answer[0], predicted_answer[0], distance_answer[0]]

        # Determine answer occurence count, and take the average between scores with multiple occurences.
        weighted_answers = {}
        for a in answers:
            if a == "":
                continue

            score = answers.count(a)
            if score < 2:
                try:
                    weighted_answers[a] = final_scores[answers.index(a)]
                except ValueError:
                    print("\nNo Search-Based answer found...")
            else:
                indices = [i for i, x in enumerate(answers) if x == a]
                
                weighted_answers[a] = 0
                sc = 0
                for j in indices:
                    cs = weighted_answers[answers[j]]
                    sc = final_scores[j]
                    weighted_answers[answers[j]] = sc + cs
                weighted_answers[a] = round(weighted_answers[a] / len(indices), 2)

        try:
            suggested_answer = max(weighted_answers, key=weighted_answers.get)
            print("\n\nTOP ANSWER SELECTION: " + suggested_answer.title() + "  ||  " + "CONFIDENCE: " + str(weighted_answers[suggested_answer]) + "% LIKELIHOOD  ||  " + "METHODS SUPPORTING CHOICE: " + str(answers.count(suggested_answer)))

        except:
            suggested_answer = "NONE"
            print("\n\nTOP ANSWER SELECTION: " + suggested_answer.title() + "  ||  " + "CONFIDENCE: 0% LIKELIHOOD  ||  " + "METHODS SUPPORTING CHOICE: 3")

        cleaned_correct_answer = re.sub("[^\s\w-]", "", correct_answer)
        print(str(suggested_answer) + ":" + cleaned_correct_answer.lower())
        print(str(suggested_answer in cleaned_correct_answer.lower()))
        if suggested_answer in cleaned_correct_answer.lower():
            metrics[question] = ["CORRECT", suggested_answer, str(weighted_answers[suggested_answer]), str(answers.count(suggested_answer)), correct_answer]
        elif suggested_answer == "NONE":
            metrics[question] = ["INCORRECT", suggested_answer, "0", "3", correct_answer]
        else:
            metrics[question] = ["INCORRECT", suggested_answer, str(weighted_answers[suggested_answer]), str(answers.count(suggested_answer)), correct_answer]
        print("\n\n")

    correct_count = 0
    incorrect_count = 0
    total_score_correct = 0
    total_score_incorrect = 0
    for x, y in metrics.items():
        print("Result: " + y[0] + "  ||  " + "Question: " + x + "  ||  Suggested Answer: " + y[1] + "  ||  Confidence Score: " + y[2]  + "%  ||  Supporting Identifiers: " + y[3] + "  ||  Correct Answer: " + y[4])
        if y[0] == "CORRECT":
            correct_count = correct_count + 1
            total_score_correct = float(y[2]) + total_score_correct
        if y[0] == "INCORRECT":
            incorrect_count = incorrect_count + 1
            total_score_incorrect = float(y[2]) + total_score_incorrect
    print("\n\nOVERALL RESULTS" + "\n" + ("-" * 50))
    
    print("\tTotal Correct: " + str(correct_count) + " (" + str(correct_count/(correct_count+incorrect_count)) + "%)")
    if correct_count != 0:
        print("\tAverage Correct Confidence Score: " + str(total_score_correct/correct_count) + "%")
    else:
        print("\tAverage Correct Confidence Score: 0.0%")
    
    print("\tTotal Incorrect: " + str(incorrect_count) + " (" + str(incorrect_count/(incorrect_count+correct_count)) + "%)")
    if incorrect_count != 0:
        print("\tAverage Incorrect Confidence Score: " + str(total_score_incorrect/incorrect_count) + "%")
    else:
        print("\tAverage Incorrect Confidence Score: 0.0%")

    
            

test_main("C:\\Users\\murra\\OneDrive\Desktop\\Natual Language Processing\\Trivial-Killer-master\\Trivia_Questions.csv")
    