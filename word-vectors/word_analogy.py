import os
import pickle
import re
import numpy as np


model_path = './models/'
# loss_model = 'cross_entropy'
loss_model = 'nce'

model_filepath = os.path.join(model_path, 'word2vec_%s.model'%(loss_model))

dictionary, steps, embeddings = pickle.load(open(model_filepath, 'rb'))

"""
==========================================================================

Write code to evaluate a relation between pairs of words.
You can access your trained model via dictionary and embeddings.
dictionary[word] will give you word_id
and embeddings[word_id] will return the embedding for that word.

word_id = dictionary[word]
v1 = embeddings[word_id]

or simply

v1 = embeddings[dictionary[word_id]]

==========================================================================
"""

# the output file for predictions
output_f = open("word_analogy_predictions.txt", "w")

# open the input file to evaluate word analogy
f = open("word_analogy_dev.txt", "r")
lines = f.readlines()

regexPattern = '|'.join(map(re.escape, [",", "||"]))

# for each line in the input file
for line in lines:

    line = line.strip()  # strips \n from the end of line
    segments = re.split(regexPattern, line)  # splits the line into segments with delimiters , ||
    transformations = []

    # for each segment in first three segments get their transformations i.e
    # if "A:B" get B-A of their embeddings
    for segment in segments[:3]:
        segment = segment.replace("\"", "")
        pair = re.split('|'.join(map(re.escape, [":"])), segment)
        context, target = pair
        context_vector = embeddings[dictionary[context]]
        target_vector = embeddings[dictionary[target]]
        transformation = np.subtract(target_vector, context_vector)
        transformations.append(transformation)

    # calculate the mean transformation for the current line
    diff_vector = np.mean(transformations, axis=0)

    similarities = {}

    # for each segment to predict, calculate cosine similarity between its transformation
    # the diff vector and store it in similarities dict
    for segment in segments[3:]:
        output_f.write(segment + " ")
        pair = re.split('|'.join(map(re.escape, [":"])), segment.replace("\"", ""))
        context, target = pair
        context_vector = embeddings[dictionary[context]]
        target_vector = embeddings[dictionary[target]]
        transformation = np.subtract(target_vector, context_vector)
        similarity = np.dot(transformation, diff_vector) / (np.linalg.norm(transformation) * np.linalg.norm(diff_vector))
        similarities[len(similarities)] = similarity

    # get the least and most illustrative pair by sorting the dict based on values
    least_index = min(similarities, key=similarities.get) + 3
    most_index = max(similarities, key=similarities.get) + 3

    # write to the output file
    output_f.write(segments[least_index] + " " + segments[most_index] + "\n")

f.close()
output_f.close()

print(steps)
