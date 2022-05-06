import wikipedia
import re

# CLASS: Answer
# DESC: Used within SearchForKeywordPages method, should contain the integer count and dictionary
# answers. Answers should contain the word: index where index is the index of the result in the returned wikipedia search.
class Answer:
    def __init__(self, count, answers):
        self.count = count
        self.answers = answers

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