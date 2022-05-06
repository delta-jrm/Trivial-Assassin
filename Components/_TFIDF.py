import wikipedia
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from nltk.corpus import stopwords
import numpy as np
import pandas as pd

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
