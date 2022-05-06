import wikipedia
import re
import nltk
from nltk.corpus import stopwords
import spacy
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

DETERMINERS = ['who', 'what', 'when', 'where', 'why', 'how', 'which', 'whose']

# CLASS: Answer
# DESC: Used within SearchForKeywordPages method, should contain the integer count and dictionary
# answers. Answers should contain the word: index where index is the index of the result in the returned wikipedia search.
class Answer:
    def __init__(self, count, answers):
        self.count = count
        self.answers = answers

# METHOD: parseQuestion
# DESC: Takes a question input, and returns three lists of tagged question keywords sorted by priority (A, B, C)
# ACCEPTS: STRING (q - Question)    
# RETURNS: LIST (q_priority_A - Keyword List Priority A NER), LIST (q_priority_B - Keyword List Priority B NNP), LIST (q_priority_C - lower-priority tags)
def parseQuestion(q):

    # Configure the stopword reference to not filter out question words.
    stop_words_with_determiners = set(stopwords.words('english'))  # Initialize stopwords
    stop_words_without_determiners = [w for w in stop_words_with_determiners if not w.lower() in DETERMINERS]

    # Remove special characters from the question and tokenize it.
    q_final = []
    q_nospec = re.sub("[^\s\w-]", "", q)  # Remove special characters
    q_tokens = nltk.word_tokenize(q_nospec) # Tokenize sentence

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
    q_priority_B = [(w, t) for (w, t) in q_tagged if re.match("(NNP|CD|NN)$", t)]     # Priority B includes proper nouns, used as a source to ideally pull wiki pages from. ADD IN VBP
    q_priority_C = [(w, t) for (w, t) in q_tagged if re.match("(RB|EX|JJ)", t) and (w, t) not in q_priority_B] # Priority C is noun keywords which may provide more specific context as to what will be looked for in the returned wikipedia documents.

    # Select terms using NER.
    q_priority_A_untagged = []
    for chunk in sentence_doc.noun_chunks:
        if (str(chunk) not in q_priority_B) and (str(chunk) not in q_priority_C): # Check to see no answers are being falsely replicated.
            q_priority_A_untagged.append(str(chunk))
    q_priority_A = nltk.pos_tag(q_priority_A_untagged)  # Priority A includes named entities, used as a source to ideally pull wiki pages from.

    # Return question keywords by priority.
    return q_priority_A, q_priority_B, q_priority_C


# METHOD: parseAnswers
# DESC: Takes an answer input, and returns a list of answers and a tagged list of answers
# ACCEPTS: STRING (a - Answers)    
# RETURNS: LIST (parsed - untagged answer list), LIST (ans - tagged answer list), LIST (lower case answers)
def parseAnswers(a):
    # Get rid of comma + space combo.
    cleaned = a.replace(", ", ",").split(",")

    # Clean the special chatacters out.
    parsed = []
    for a in cleaned:
        nos = re.sub("('s|\.[A-Za-z]{0,4})", "", a)   # Remove 's occurences
        nospec = re.sub("[^\s\w-]", "", nos) # Get rid of special characters except -
        if a != '':
            parsed.append(nospec)

    # POS tag it.
    ans = nltk.pos_tag(parsed)

    # Convert answers to lowercase.
    lower_answers = []
    for a in parsed:
        if a != '':
            lower_answers.append(a.lower())

    # Return the cleaned list and the POS-tagged one.
    return parsed, ans, lower_answers


# METHOD: searchNamedEntityRecognitionKeywordPages
# DESC: Takes the list of keywords for named entity recognition, puts them together to search wikipedia.
# ACCEPTS: LIST (A - Priority A Keywords)    
# RETURNS: LIST <WIKIPEDIA_RESULTS for Priority A keywords>
def searchNamedEntityRecognitionKeywordPages(A, answers):
    
    if len(A) > 3:
        for i in range(2, len(A)-1):
            # Put together a string of keywords to search with.
            NER = ""
            for word, tag in A[0:i]:
                NER = NER + word + " "
            
            # Perform a search using the priority-A keywords identified in question-parsing.
            NER_search = wikipedia.search(NER)

            for answer in answers:
                for search in NER_search:
                    cleaned = re.sub("[^\s\w-]", "", search)
                    if answer in cleaned.lower():
                        break
    else:
            NER = ""
            for word, tag in A:
                NER = NER + word + " "
            
            # Perform a search using the priority-A keywords identified in question-parsing.
            NER_search = wikipedia.search(NER)

    # Return search results.
    return NER_search

# METHOD: searchPartOfSpeechKeywordPages
# DESC: Takes the list of keywords for part of speech sorting, puts them together to search wikipedia.
# ACCEPTS: LIST (B - Priority B Keywords), LIST (C - Priority C Keywords)    
# RETURNS: LIST <WIKIPEDIA_RESULTS for Priority B+C keywords>
def searchPartOfSpeechKeywordPages(B, C, answers):
   
    if len(B) > 3:
        for i in range(2, len(B)-1):
            # Put together a string of keywords to search with.
            POS = ""
            for word, tag in B[0:i]:
                if word not in POS:
                    POS = POS + word + " "
            for word, tag in C[-1:]:
                if word not in POS:
                    POS = POS + word + " "
            
            # Perform a search using the priority B & C keywords identified in question-parsing.
            POS_search = wikipedia.search(POS) 

            for answer in answers:
                for search in POS_search:
                    cleaned = re.sub("[^\s\w-]", "", search)
                    if answer in cleaned.lower():
                        break
    else:
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

# METHOD: searchForKeywordPages
# DESC: Takes the list of answers and keyword pages, and checks if there are any occurences of the answers in the keyword pages. If there are, the solved state will change depending on the number
# of occurences found. 
# ACCEPTS: LIST <WIKIPEDIA_RESULTS for Priority A keywords>, LIST <WIKIPEDIA_RESULTS for Priority B+C keywords>, LIST (parsed - untagged answer list)
# RETURNS: INT (Solved State), LIST (Either containing STR or DICT depending on solved-state), LIST (lowercase answers.)
def searchForKeywordPages(NERu, POSu, answers):

    # Take first X page occurences where 0:X
    NER = NERu[0:4]
    POS = POSu[0:4]

    # Initialize empty answer objects for the NER and POS Searches.
    POS_answer = Answer(0, {})
    NER_answer = Answer(0, {})

    # Take NER and POS lists, convert them to strings used for comparison.
    NER_compare = " ".join(NER)
    POS_compare = " ".join(POS)

    # Convert NER and POS to lowercase, put lists in NER/POS_lower.
    NER_lower = []
    for title in NER:
        if title != "":
            NER_lower.append(title.lower())

    POS_lower = []
    for title in POS:
        if title != "":
            POS_lower.append(title.lower())

    # Identify any answers colliding with returned search results in NER
    for a in answers:
        if a in NER_compare.lower():
            NER_answer.count = NER_answer.count + 1 # If answer found, increment count
            try:
                NER_answer.answers[a] = NER_lower.index(a) # Track answer and index.
            except:
                for item in NER:
                    if a in item.lower():
                        NER_answer.answers[a] = NER.index(item) # Track answer and index, comparison made here if the list of answers contains some element of the provided term.
    
    # Identify any answers colliding with returned search results in POS
    for a in answers:
        if a in POS_compare.lower():
            POS_answer.count = POS_answer.count + 1 # If answer found, increment count
            try:
                POS_answer.answers[a] = POS.index(a) # Track answer and index.
            except:
                for item in POS:
                    if a in item.lower():
                        POS_answer.answers[a] = POS.index(item)  # Track answer and index, comparison made here if the list of answers contains some element of the provided term.

    # Determine the solved state and obtain the answers.
    solved_state, result = determineSolvedState(NER_answer, POS_answer, NER, POS)
    return solved_state, result

# METHOD: determineSolvedState
# DESC: Carries out selection for solved state and list of answers, used withink searchForKeywordPages
# ACCEPTS: ANSWER_OBJECT (NER Answer), ANSWER_OBJECT (POS Answer), LIST <WIKIPEDIA_RESULTS for Priority A keywords>, LIST <WIKIPEDIA_RESULTS for Priority B+C keywords>
# RETURNS: INT (Solved State), LIST (Either containing STR or DICT depending on solved-state)
def determineSolvedState(NERAns, POSAns, NER, POS):
    # Initialize the solved state as 0
    # Solved States are as follows: 
    # 0 = Unsolved with no answers identified in search results, 
    # 1 = Solved with single answer identified in search results, 
    # 2 = Unsolved with multiple answers identified in search results.

    solved_state = 0
    solved_state = 0

    # Nothing found, solved state 0.
    if NERAns.count == 0 and POSAns.count == 0:
        return solved_state, list(set(NER + POS))

    # Answer found, solved state 1.
    elif NERAns.count == 1 and POSAns.count == 0:
        solved_state = 1
        return solved_state, list(NERAns.answers.keys()) # RETURNS INT AND LIST CONTAINING STRINGS

    # Answer found, solved state 1.
    elif NERAns.count == 0 and POSAns.count == 1:
        solved_state = 1
        return solved_state, list(POSAns.answers.keys()) # RETURNS INT AND LIST CONTAINING STRINGS

    # Potential answer found, remove duplicates and resolve to state 1 or 2 accordingly.
    elif NERAns.count == 1 and POSAns.count == 1:
        if NERAns.answers.keys() == POSAns.answers.keys():
            solved_state = 1
            return solved_state, list(POSAns.answers.keys()) # RETURNS INT AND LIST CONTAINING STRINGS

        else:
            solved_state = 2
            return solved_state, [NERAns.answers, POSAns.answers] # RETURNS INT AND LIST CONTAINING DICTIONARIES

    # Multiple answers found, solved state 2.
    elif NERAns.count >= 1 and POSAns.count >= 1:
        solved_state = 2

        # Find duplicates
        duplicates = []
        for ans in NERAns.answers.keys():
            if ans in POSAns.answers.keys():
                duplicates.append(ans)

        # Remove duplicates.
        for d in duplicates:
            NERAns.answers.pop(d, None)

        return solved_state, [NERAns.answers, POSAns.answers] # RETURNS INT AND LIST CONTAINING DICTIONARIES

    else:
        solved_state = -1
        return solved_state, "ERROR" # RETURNS INT AND STRING

# METHOD: cleanKeywordPage
# DESC: Removes special characters from a given content page.
# ACCEPTS: STRING (content)    
# RETURNS: STRING (content, but without special characters.)
def cleanKeywordPage(content):

    # Get rid of special characters, return result.
    nolinks = re.sub("('s|\.[A-Za-z]{0,3})", "", content)
    nospec = re.sub("[^\s\w-]", "", nolinks)  # Remove special characters
    return nospec

# METHOD: composeCorpusFromKeywordPages
# DESC: Takes a list of search terms, obtains the page, and puts it into a dictionary.
# ACCEPTS: LIST (keywordpages - CONTAINS STRINGS obtained from running searchForKeywordPages) 
# RETURNS: DICT (results, contains the title mapped to cleaned page contents.)
def composeCorpusFromKeywordPages(keywordpages, answers):

    # Establish an empty results dictionary.
    results = {}

    # For each page identified in keywords, obtain the title and the content, add it to results.
    for pg in list(keywordpages):
        try:
            reference = wikipedia.page(pg, auto_suggest=False) # Gets the page - DO NOT REMOVE AUTO_SUGGEST=FALSE.
            title = reference.title # Gets the title
            text = cleanKeywordPage(reference.content) # Gets the clean page content.
            ans_count = 0
            for ans in answers:
                if ans in text.lower(): # Finds if answer is contained in text, increases count.
                    ans_count = ans_count + 1
            results[title] = text
        except wikipedia.DisambiguationError:
            continue
        except wikipedia.PageError:
            continue

    # Return the corpus of documents.
    return results

# METHOD: composeCorpusFromTargetedKeywordPages
# DESC: Takes the list of search terms and search answers, using targeted indexes to pull the
# reference pages. This is used when multiple answers show up in search results.
# ACCEPTS: LIST (keywordpages - CONTAINS DICT obtained from running searchForKeywordPages) 
# RETURNS: DICT (results, contains the title mapped to cleaned page contents.)
def composeCorpusFromTargetedKeywordPages(NER, NER_answers, POS, POS_answers):

    # Establish an empty results dictionary.
    results = {}

    # For each page identified in NER keywords, obtain the title and the content, add it to results.
    for idx in NER_answers.values():
        try:
            reference = wikipedia.page(NER[idx], auto_suggest=False) # Get page
            title = reference.title
            text = cleanKeywordPage(reference.text) # Clean the reference content.
            results[title] = text
        except wikipedia.DisambiguationError:
            continue
        except wikipedia.PageError:
            continue

    # For each page identified in POS keywords, obtain the title and the content, add it to results.
    for idx in POS_answers.values():
        try:
            reference = wikipedia.page(POS[idx], auto_suggest=False) # Get Page
            title = reference.title
            text = cleanKeywordPage(reference.text) # Clean the reference content.
            results[title] = text
        except wikipedia.DisambiguationError:
            continue
        except wikipedia.PageError:
            continue

    # Return the corpus of documents.
    return results

# METHOD: createKeywordCorpusDF
# DESC: Takes the resulting dictionary (corpus) from composeCorpusFromTargetedKeywordPages and converts it into a dataframe.
# ACCEPTS: DICT (results, contains the title mapped to cleaned page contents.)
# RETURNS: PD_DATAFRAME (kw_df - Dataframe containing <keyword, text, tokenized, text_no_stopwords>)
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

    # Return the dataframe.
    return kw_df

# METHOD: calculateKeywordCorpusTFIDF
# DESC: Takes a built dataframe and calculates resulting tf_idf information for terms and provided pages.
# ACCEPTS: PD_DATAFRAME (kw_df - Dataframe containing <keyword, text, tokenized, text_no_stopwords>)
# RETURNS: LIST (VOCAB), PD_DATAFRAME (TF_IDF DF Information)
def calculateKeywordCorpusTFIDF(keyworddf):

    # Set up Vectorizer for TFIDF with tuning to account for MBTI corpus conditions
    kw_vector = TfidfVectorizer(  # max_df=0.40,         # Drop words that appear more than X%, currently unused in favor of min.
        min_df=2,              # only use words that appear at least X times
        stop_words='english',  # remove stop words
        lowercase=True,        # Convert everything to lower case
        use_idf=True,          # Use idf
        norm=u'l2',            # Normalization
        smooth_idf=True        # Prevents divide-by-zero errors
    )

    # Parse the entire set of keyword-related documents into TF-IDF.
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

# METHOD: obtainAnswerFromTFIDF
# DESC: Takes a TF-IDF Dataframe and looks for the answers, providing an estimation on the answer.
# ACCEPTS: LIST (VOCAB), PD_DATAFRAME (TF_IDF DF Information), LIST (answers - lower-case non-tagged list of answers)
# RETURNS: STR (Final Answer), INT (Score)
def obtainAnswerFromTFIDF(vocab, keywordTFIDF, answers):
    final_answer = ""

    # Combine all the counts to sum total term counts across every document.
    try:
        summed_series = keywordTFIDF.sum(axis=0)
    except:
        return final_answer, 0

    # Get summed series and top TF-IDF information.
    vals_tfidf = summed_series.values
    idx_tfidf = np.argsort(vals_tfidf)
    top_tfidf = idx_tfidf.flatten()

    # Pull individual words out of the answers vector for Tf-IDF comparison.
    sw = list(stopwords.words('english'))  # Initialize stopwords

    # Pull individual words from the given answers without stopwords
    # Ex: For "Alice in Wonderland" check "alice" and "wonderland" in the vector.
    answer_compare = {}
    for a in answers:
        entry = a.split(" ") # Split answer into individual words.
        words = []
        for e in entry:
            if e not in sw: # Remove stop words
                words.append(e.lower())
        answer_compare[a] = words

    # Check if individual words from the answers are found in the TF-IDF vector. 
    # Look for values of answers found in TF_IDF and track them.
    v = list(vocab)
    answers_tfidf = {}
    for keyword,subwords in answer_compare.items():
        keyword_score = 0
        for word in subwords: # For each word in an answer, check if it's in the TFIDF information, and add it to the total keyword score.
            if word in v:
                x = v.index(word)
                keyword_score = float(summed_series[x]) + keyword_score
        answers_tfidf[keyword] = keyword_score # Mark a corresponding keyword score

    # Find answer occurences in TF-IDF vector and map to the corresponding score.
    """
    for entry in reversed(top_tfidf):
        if vocab[entry].lower() in " ".join(answers) and vocab[entry].lower() not in " ".join(answers_tfidf.keys()):
            answers_tfidf[vocab[entry]] = summed_series[entry]"""
    
    print("\nTF-IDF Answer Identification\n" + ("~" * 50))
    
    # Get the total sum of answer values to calculate average for the score.
    total = sum(answers_tfidf.values())
    if total == 0:
        total = 1

    answers_found = []

    # Calculate average TFIDF score for each, print to console.
    for w,v in sorted(answers_tfidf.items(), key=lambda kv: kv[1], reverse=True):
        score = v/total
        print("\t" + w.title() + "  -  SCORE: " + str(round(score*100, 2)) + "% LIKELIHOOD")
        answers_found.append(w)

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

# METHOD: obtainHighPriorityDocs
# DESC: Takes the given list of terms, answers, and the keyword corpus and looks for two things -
# count of answer occurence in an entry and minimum distance between answers and terms. Returns a
# list of high-priority documents containing these answers and keywords, along with counts for terms.
# ACCEPTS: LIST (VOCAB), PD_DATAFRAME (TF_IDF DF Information), LIST (answers - lower-case non-tagged list of answers), DICT (Keyword Corpus)
# RETURNS: DICT (Containing INT score counts for document), DICT (Containing list of reference answer, terms, and the lower text itself.)
def obtainHighPriorityDocs(A, B, C, answers, keyworddict):

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
        # Loop through dictionary to count answer and term occurence.
        for title,text in keyworddict.items():
            term_score = 0
            ans_score = 0
            ref_ans = []
            ref_term = []  

            # Find occurence of answer in reference text, add to answer score, answer count, and answer reference.
            for a in ans:
                try:
                    tlower = text.lower()
                except:
                    tlower = str(text.str.lower())
                if a.lower() in tlower:
                    ans_score = ans_score + 1
                    ans_count = text.count(a)
                    ref_ans.append((a.lower(), ans_count))

            # Find occurence of term in reference text, add to term score, term count, and term reference.
            for t in terms:
                if t.lower() in tlower:
                    term_score = term_score + 1
                    ref_term.append(t.lower())

            # If an answer is found within the document, add it to the high_priority list.
            if ans_score >= 1:
                priority_scores[title] = ans_score + term_score
                high_priority[title] = [ref_ans, ref_term, tlower]
    
    return priority_scores, high_priority

# METHOD: obtainBestGuessFromDF
# DESC: Takes the igh priority docs and calculates probability score based on minimum distance from
# answers to keywords, as well as weighting these scores with the frequency count of the answers.
# ACCEPTS: DICT (Containing INT score counts for document), DICT (Containing list of reference answer, terms, and the lower text itself.)
# RETURNS: STR (Final Answer based on Keyword Distance), INT (Score)
def obtainBestGuessFromDF(priority_scores, high_priority):
    # If a single valid document was found, take the maximum-score document to use for the final score calculation.
    
    # Obtain a list of the answers (lowercase).
    temp = []
    for answers in high_priority.values():
        if answers[0][0]:
            answer = answers[0][0][0].lower()
            temp.append(answer)
    full_answers = list(set(temp)) # Create list of full answers.

    # Obtain a list of the document names.
    full_docs = priority_scores.keys()

    report_card = {}
    print("\nKeyword-to-Answer Distance Identification\n" + ("~" * 50))

    # Iterate through document calculating scores for each respective answer.
    for answer_doc in full_docs:

        # Set the reference doc, reference answers, and reference keywords.
        ref_doc = high_priority[answer_doc][2].split(" ")
        hp_ans = high_priority[answer_doc][0]
        hp_term = high_priority[answer_doc][1]

        # If there's more than one, go through an answer score calculation.
        if len(hp_ans) > 1:
            distance_measure = {}
            answer_counts = {}

            # Obtain the answer count for the document.
            for a,s in hp_ans:
                answer_counts[a] = s
                distcount = 0

                # Calculate total minimum distance given all terms.
                for b in hp_term:
                    dist = obtainWordDistance(a, b, ref_doc)
                    distcount = distcount + dist
                distance_measure[a] = distcount / len(hp_term)
            
            dist_scores = {}
            total_scores = {}

            # Get the sum of the distance measure values and the answer counts.
            total_dist = sum(distance_measure.values())
            total_ans = sum(answer_counts.values())

            # Prevent divide-by 0 error
            if total_dist == 0:
                total_dist = -1
            
            # Calculate average distance for the distance score.
            for w,v in distance_measure.items():
                dist_score = v/total_dist
                dist_scores[w] = dist_score
            
            # Prevent divide-by 0 error
            if total_ans == 0:
                total_ans = -1
            
            # Calculate average answer count for the answer score.
            for w,v in answer_counts.items():
                count_score = v/total_ans
                total_scores[w] = count_score * (2 * dist_scores[w])
            
            # Get tally of every total score to calculate average.
            tally = sum(total_scores.values())

            # Prevent divide-by 0 error
            if tally == 0:
                tally = -1
            
            # Sort the total scores from most to least.
            itemref = sorted(total_scores.items(), key=lambda kv: kv[1], reverse=True)

            # Calculate total score and add the given total scores to the report card.
            for w,v in itemref:
                score = v/tally
                total_scores[w] = score
            #dist_answer = max(total_scores, key=total_scores.get)
            report_card[answer_doc] = total_scores

    # Get the totals together from the report card. Add all of the scores up for each answer
    totals = {}
    for f in full_answers:
        answer_score = 0
        for report,scores in report_card.items():
            if f in scores.keys():
                #weight = priority_scores[report]
                answer_score = scores[f] + answer_score
        totals[f] = answer_score # Add the score for the answer to the final totals.

    # Get sum of all scores to calculate average.
    total_summed = sum(totals.values())

    # Establish final ref information.
    final = {}
    dist_answer = ""
    finalref = sorted(totals.items(), key=lambda kv: kv[1], reverse=True)

    # Print out the final score likelihoods and determine the answer by taking the max.
    for ans,totscore in finalref:
        if total_summed != 0:
            t = round((totscore/total_summed * 100),2)
        else:
            t = 0
        print("\t" + ans.title() + "  -  SCORE: " + str(t) + "% LIKELIHOOD")
        final[ans] = t

    try:
        dist_answer = max(final, key=final.get)
        return dist_answer, final[dist_answer]

    except:
        return dist_answer, 0


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
        A, B, C = parseQuestion(question)
        answers, pos_answers, lower_answers = parseAnswers(unhandled_answers)
        print("\nQuestion: " + question + "  ||  Answers: " + str(unhandled_answers))
        
        # Use the Named Entity Recognition Keywords and the Part of Speech Keywords to obtain the pages.
        NER_search = searchNamedEntityRecognitionKeywordPages(A, lower_answers)
        POS_search = searchPartOfSpeechKeywordPages(B, C, lower_answers)

        # Try using the Wikipedia search function to see if the answer can be obtained from a top-levelquery.
        is_solved, output = searchForKeywordPages(NER_search, POS_search, lower_answers)

        # Check the results provided by the search, an answer may have been identified.
        if is_solved == 1 and output != '' and output:
            print("KEYWORD SEARCH RESULT FOUND\n" + ("-" * 50) + "\n\tIDENTIFIED ANSWER: " + str(output[0]).title())   # If answer identified, print and set anwer result.
            search_answer = (output[0].lower(), 100)
            keyword_corpus = composeCorpusFromKeywordPages(output, lower_answers) # Compose target corpus, build dataframe from corpus, obtain vocab and TF-IDF vector.

        # In the case that pages with the titles found that contained more than one answer, a targeted page corpus will be composed.
        elif is_solved == 2:
            keyword_corpus = composeCorpusFromTargetedKeywordPages(NER_search, output[0], POS_search, output[1]) # Compose target corpus, build dataframe from corpus, obtain vocab and TF-IDF vector.

        # In the case that page titles did not contain any answer terms, a broad search will be conducted.
        elif is_solved == 0 or (is_solved == 1 and output == ''):
            keyword_corpus = composeCorpusFromKeywordPages(output, lower_answers) # Compose target corpus, build dataframe from corpus, obtain vocab and TF-IDF vector.
    
        else:
            print("NO ANSWERS PROPERLY IDENTIFIED... EXITING")
            continue

        # Build the dataframe and TF-IDF information.
        keyword_df = createKeywordCorpusDF(keyword_corpus)
        vocab, keyword_tfidf = calculateKeywordCorpusTFIDF(keyword_df)

        # Use comparison points to obtain answers from TF-IDF vectors and distance measures between answers.
        if type(vocab) is not str:
            predicted_answer = obtainAnswerFromTFIDF(vocab, keyword_tfidf, lower_answers)
        else:
            predicted_answer = obtainAnswerFromTFIDF(str(vocab), keyword_tfidf, lower_answers)

        # Print answer.
        print("\n\tIDENTIFIED ANSWER FROM TF-IDF SEARCH: " + str(predicted_answer).title())

        # Search through the docs and find the ones containing answers.
        ps, hp = obtainHighPriorityDocs(A, B, C, pos_answers, keyword_corpus)

        # Take the identified docs and try to obtain an answer and print.
        distance_answer = obtainBestGuessFromDF(ps, hp)
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
    version = ""

    # Print out a cool banner!
    print( "\n" + ("-" * 50) + "\nTrivial Assassin - Double Agent V3.2\n" + ("-" * 50))

    print("Trivial Assassin - Double Agent enables the user to select either the single-question (training) or multiple-question (mission) functionality.")
    version = input ("Would you like to ask a single-question or multiple-questions? Please enter the number corresponding to the desired choice.\nSingle: 0\nMultiple: 1\nSelection: ")

    if version == "0":
        # Print out a cool banner!
        print( "\n" + ("-" * 50) + "\nTrivial Assassin V3.2\n" + ("-" * 50))

        # Obtain the question from the user, retrieve priority-sorted list of tagged terms used for searching for a wikipedia page.
        question = input("Please enter a trivia question:\n")

        # Obtain a list of potential answers, retrieve the list of POS-tagged answers.
        unhandled_answers = input("\nPlease enter a comma-seperated list of potential answers (ex. red, blue, green):\n")

        solve_question(question, unhandled_answers)

    elif version == "1":
        # Print out a cool banner!
        print( "\n" + ("-" * 50) + "\nTrivial Assassin Mission V3.2\n" + ("-" * 50))

        print("Trivial Assassin Mission is a method to test the assassin against a list of given questions, answers, and the correct answer. Examples have been provided within the data directory.")


        # Obtain the question from the user, retrieve priority-sorted list of tagged terms used for searching for a wikipedia page.
        filename = input("Please enter a file path pointing to a CSV file containing the list of questions to test with:\n")

        solve_multiple_questions(filename)
    
    else:
        print("Invalid Input... Exiting...")

main()