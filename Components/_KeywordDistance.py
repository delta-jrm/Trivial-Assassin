
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