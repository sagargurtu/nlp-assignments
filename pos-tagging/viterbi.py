import numpy as np

def run_viterbi(emission_scores, trans_scores, start_scores, end_scores):
    """Run the Viterbi algorithm.

    N - number of tokens (length of sentence)
    L - number of labels

    As an input, you are given:
    - Emission scores, as an NxL array
    - Transition scores (Yp -> Yc), as an LxL array
    - Start transition scores (S -> Y), as an Lx1 array
    - End transition scores (Y -> E), as an Lx1 array

    You have to return a tuple (s,y), where:
    - s is the score of the best sequence
    - y is the size N array/seq of integers representing the best sequence.
    """
    L = start_scores.shape[0]
    assert end_scores.shape[0] == L
    assert trans_scores.shape[0] == L
    assert trans_scores.shape[1] == L
    assert emission_scores.shape[1] == L
    N = emission_scores.shape[0]

    # y will contain the final tag sequence
    y = [0] * N

    # scores table populated with -inf initially
    scores_table = np.ones((N, L)) * float('-inf')

    # back pointers table populated with zeroes intially
    back_pointers = np.zeros((N, L), dtype=np.int32)

    # Compute starting transition scores first
    scores_table[0] = emission_scores[0] + np.transpose(start_scores)

    # For subsequent scores, add transition and emission scores to previous computed best scores
    # and get max. Read report for more info.
    for word_index in range(N - 1):
        for tag_index in range(L):
            scores_table[word_index + 1][tag_index], back_pointers[word_index + 1][tag_index] = \
                max([(scores_table[word_index][t] + trans_scores[t][tag_index] +
                      emission_scores[word_index + 1][tag_index], t) for t in range(L)])

    # Compute end transition scores and get the final score
    final_scores = list(scores_table[N - 1] + np.transpose(end_scores))
    final_score = max(final_scores)
    y[N - 1] = final_scores.index(final_score)

    # Get tag sequence using back pointers
    tag_index = N - 1
    while tag_index != 0:
        y[tag_index-1] = back_pointers[tag_index][int(y[tag_index])]
        tag_index -= 1

    return final_score, y
