import re
import nltk
from nltk.corpus import stopwords
import spacy

DETERMINERS = ['who', 'what', 'when', 'where', 'why', 'how', 'which', 'whose']

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
# RETURNS: LIST (parsed - untagged answer list), LIST (ans - tagged answer list), LIST (lower-case answers)
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

