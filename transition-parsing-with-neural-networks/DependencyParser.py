import collections
import tensorflow as tf
import numpy as np
import pickle
import math
from progressbar import ProgressBar

from DependencyTree import DependencyTree
from ParsingSystem import ParsingSystem
from Configuration import Configuration
import Config
import Util

"""
This script defines a transition-based dependency parser which makes
use of a classifier powered by a neural network. The neural network
accepts distributed representation inputs: dense, continuous
representations of words, their part of speech tags, and the labels
which connect words in a partial dependency parse.

This is an implementation of the method described in

Danqi Chen and Christopher Manning. A Fast and Accurate Dependency Parser Using Neural Networks. In EMNLP 2014.

Author: Danqi Chen, Jon Gauthier
Modified by: Heeyoung Kwon (2017)
Modified by: Jun S. Kang (2018 Mar)
"""


class DependencyParserModel(object):

    def __init__(self, graph, embedding_array, Config):

        self.build_graph(graph, embedding_array, Config)

    def build_graph(self, graph, embedding_array, Config):
        """

        :param graph:
        :param embedding_array:
        :param Config:
        :return:
        """

        with graph.as_default():
            self.embeddings = tf.Variable(embedding_array, dtype=tf.float32)
            # self.embeddings = tf.Variable(embedding_array, dtype=tf.float32, trainable=False)

            """
            ===================================================================

            Define the computational graph with necessary variables.
            
            1) You may need placeholders of:
                - Many parameters are defined at Config: batch_size, n_Tokens, etc
                - # of transitions can be get by calling parsing_system.numTransitions()
                
            self.train_inputs = 
            self.train_labels = 
            self.test_inputs =
            ...
            
                
            2) Call forward_pass and get predictions
            
            ...
            self.prediction = self.forward_pass(embed, weights_input, biases_input, weights_output)


            3) Implement the loss function described in the paper
             - lambda is defined at Config.lam
            
            ...
            self.loss =
            
            ===================================================================
            """

            # Initialize placeholders for train inputs, labels and test inputs
            self.train_inputs = tf.placeholder(tf.int32, shape=[Config.batch_size, Config.n_Tokens])
            self.train_labels = tf.placeholder(tf.float32, shape=[Config.batch_size, parsing_system.numTransitions()])
            self.test_inputs = tf.placeholder(tf.int32)

            # Lookup and reshape embeddings
            embed = tf.nn.embedding_lookup(self.embeddings, self.train_inputs)
            embed = tf.reshape(embed, [Config.batch_size, Config.embedding_size * Config.n_Tokens])

            # Initialize weight inputs
            weights_input1 = tf.Variable(
                tf.random_normal([Config.embedding_size * Config.n_Tokens, Config.hidden_layer1], mean=0.0, stddev=0.1))

            # Initialize bias inputs
            biases_input1 = tf.Variable(tf.zeros([Config.hidden_layer1]))

            # Initialize weight output
            weights_output = tf.Variable(
                tf.random_normal([Config.hidden_layer1, parsing_system.numTransitions()], mean=0.0, stddev=0.1))

            # Initialize weight inputs for different hidden layers
            '''
            weights_input2 = tf.Variable(tf.random_normal([Config.hidden_layer1, Config.hidden_layer2], 
                                                          mean=0.0, stddev=0.1))
            weights_input3 = tf.Variable(tf.random_normal([Config.hidden_layer2, Config.hidden_layer3], 
                                                          mean=0.0, stddev=0.1))
            '''

            # Initialize bias inputs for different hidden layers
            '''
            biases_input2 = tf.Variable(tf.zeros([Config.hidden_layer2]))
            biases_input3 = tf.Variable(tf.zeros([Config.hidden_layer3]))
            '''

            # Initialize weights for words, pos and labels to be used for parallel hidden layers
            '''
            word_weights = tf.Variable(tf.random_normal([Config.embedding_size * 18, Config.hidden_layer1],
                                                        mean=0.0, stddev=0.1))
            pos_weights = tf.Variable(tf.random_normal([Config.embedding_size * 18, Config.hidden_layer1],
                                                       mean=0.0, stddev=0.1))
            label_weights = tf.Variable(tf.random_normal([Config.embedding_size * 12, Config.hidden_layer1],
                                                         mean=0.0, stddev=0.1))
            '''

            # Get forward pass prediction
            # Change the method signature and add extra parameters for different experiments
            self.prediction = self.forward_pass(embed, weights_input1, biases_input1, weights_output)

            # Calculate l2 regularization by considering all above variables
            theta = tf.nn.l2_loss(weights_input1) + tf.nn.l2_loss(biases_input1) + \
                    tf.nn.l2_loss(weights_output) + tf.nn.l2_loss(self.embeddings)

            regularization = Config.lam * theta

            # Implement loss function using softmax cross entropy with logits
            self.loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(self.train_labels, axis=1),
                                                               logits=self.prediction)) + regularization

            optimizer = tf.train.GradientDescentOptimizer(Config.learning_rate)
            grads = optimizer.compute_gradients(self.loss)
            clipped_grads = [(tf.clip_by_norm(grad, 5), var) for grad, var in grads]
            self.app = optimizer.apply_gradients(clipped_grads)

            # For test data, we only need to get its prediction
            test_embed = tf.nn.embedding_lookup(self.embeddings, self.test_inputs)
            test_embed = tf.reshape(test_embed, [1, -1])
            self.test_pred = self.forward_pass(test_embed, weights_input1, biases_input1, weights_output)

            # intializer
            self.init = tf.global_variables_initializer()

    def train(self, sess, num_steps):
        """

        :param sess:
        :param num_steps:
        :return:
        """
        self.init.run()
        print("Initailized")

        average_loss = 0
        for step in range(num_steps):
            start = (step * Config.batch_size) % len(trainFeats)
            end = ((step + 1) * Config.batch_size) % len(trainFeats)
            if end < start:
                start -= end
                end = len(trainFeats)
            batch_inputs, batch_labels = trainFeats[start:end], trainLabels[start:end]

            feed_dict = {self.train_inputs: batch_inputs, self.train_labels: batch_labels}

            _, loss_val = sess.run([self.app, self.loss], feed_dict=feed_dict)
            average_loss += loss_val

            if step % Config.display_step == 0:
                if step > 0:
                    average_loss /= Config.display_step
                print("Average loss at step ", step, ": ", average_loss)
                average_loss = 0
            if step % Config.validation_step == 0 and step != 0:
                print("\nTesting on dev set at step ", step)
                predTrees = []
                for sent in devSents:
                    numTrans = parsing_system.numTransitions()

                    c = parsing_system.initialConfiguration(sent)
                    while not parsing_system.isTerminal(c):
                        feat = getFeatures(c)
                        pred = sess.run(self.test_pred, feed_dict={self.test_inputs: feat})

                        optScore = -float('inf')
                        optTrans = ""

                        for j in range(numTrans):
                            if pred[0, j] > optScore and parsing_system.canApply(c, parsing_system.transitions[j]):
                                optScore = pred[0, j]
                                optTrans = parsing_system.transitions[j]

                        c = parsing_system.apply(c, optTrans)

                    predTrees.append(c.tree)
                result = parsing_system.evaluate(devSents, predTrees, devTrees)
                print(result)

        print("Train Finished.")

    def evaluate(self, sess, testSents):
        """

        :param sess:
        :return:
        """

        print("Starting to predict on test set")
        predTrees = []
        for sent in testSents:
            numTrans = parsing_system.numTransitions()

            c = parsing_system.initialConfiguration(sent)
            while not parsing_system.isTerminal(c):
                # feat = getFeatureArray(c)
                feat = getFeatures(c)
                pred = sess.run(self.test_pred, feed_dict={self.test_inputs: feat})

                optScore = -float('inf')
                optTrans = ""

                for j in range(numTrans):
                    if pred[0, j] > optScore and parsing_system.canApply(c, parsing_system.transitions[j]):
                        optScore = pred[0, j]
                        optTrans = parsing_system.transitions[j]

                c = parsing_system.apply(c, optTrans)

            predTrees.append(c.tree)
        print("Saved the test results.")
        Util.writeConll('result_test.conll', testSents, predTrees)


    def forward_pass(self, embed, weights_input1, biases_input1, weights_output):
        """

        :param embed:
        :param weights:
        :param biases:
        :return:
        """
        """
        =======================================================

        Implement the forwrad pass described in
        "A Fast and Accurate Dependency Parser using Neural Networks"(2014)

        =======================================================
        """

        # Define different activation functions
        def activation_function(input):

            # Cube
            af = tf.pow(input, 3)

            # Tanh
            # af = tf.nn.tanh(input)

            # Sigmoid
            # af = tf.nn.sigmoid(input)

            # Relu
            # af = tf.nn.relu(input)

            return af

        # Experiment with different number of layers
        h1 = activation_function(tf.add(tf.matmul(embed, weights_input1), biases_input1))
        # h2 = activation_function(tf.add(tf.matmul(h1, weights_input2), biases_input2))
        # h3 = activation_function(tf.add(tf.matmul(h2, weights_input3), biases_input3))

        # Experiment with parallel layers
        '''
        word_dim = Config.embedding_size * 18
        h1 = activation_function(tf.matmul(embed[:, 0:word_dim], word_weights))
        pos_dim = word_dim + (Config.embedding_size * 18)
        h2 = activation_function(tf.matmul(embed[:, word_dim:pos_dim], pos_weights))
        label_dim = pos_dim + (Config.embedding_size * 12)
        h3 = activation_function(tf.matmul(embed[:, pos_dim:label_dim], label_weights))
        h = activation_function(tf.add(tf.add(tf.add(h1, h2), h3), biases_input1))
        '''

        h = tf.matmul(h1, weights_output)

        return h


def genDictionaries(sents, trees):
    word = []
    pos = []
    label = []
    for s in sents:
        for token in s:
            word.append(token['word'])
            pos.append(token['POS'])

    rootLabel = None
    for tree in trees:
        for k in range(1, tree.n + 1):
            if tree.getHead(k) == 0:
                rootLabel = tree.getLabel(k)
            else:
                label.append(tree.getLabel(k))

    if rootLabel in label:
        label.remove(rootLabel)

    index = 0
    wordCount = [Config.UNKNOWN, Config.NULL, Config.ROOT]
    wordCount.extend(collections.Counter(word))
    for word in wordCount:
        wordDict[word] = index
        index += 1

    posCount = [Config.UNKNOWN, Config.NULL, Config.ROOT]
    posCount.extend(collections.Counter(pos))
    for pos in posCount:
        posDict[pos] = index
        index += 1

    labelCount = [Config.NULL, rootLabel]
    labelCount.extend(collections.Counter(label))
    for label in labelCount:
        labelDict[label] = index
        index += 1

    return wordDict, posDict, labelDict


def getWordID(s):
    if s in wordDict:
        return wordDict[s]
    else:
        return wordDict[Config.UNKNOWN]


def getPosID(s):
    if s in posDict:
        return posDict[s]
    else:
        return posDict[Config.UNKNOWN]


def getLabelID(s):
    if s in labelDict:
        return labelDict[s]
    else:
        return labelDict[Config.UNKNOWN]


def getFeatures(c):

    """
    =================================================================

    Implement feature extraction described in
    "A Fast and Accurate Dependency Parser using Neural Networks"(2014)

    =================================================================
    """

    # We have to pick ids of words, pos tags and labels for the parser.
    # According to the paper, we will pick Sw = 18 elements, Sp = 18 elements
    # and Sl = 12 elements
    words = []
    pos_tags = []
    labels = []

    # For the top 3 words on the stack and buffer, collect word ids and pos tag ids.
    # Collected Sw = 2*3 = 6, St = 2*3 = 6
    for index in range(0, 3):
        stackIndex, bufferIndex = c.getStack(index), c.getBuffer(index)
        words.append(getWordID(c.getWord(stackIndex))), words.append(getWordID(c.getWord(bufferIndex)))
        pos_tags.append(getPosID(c.getPOS(stackIndex))), pos_tags.append(getPosID(c.getPOS(bufferIndex)))

    # For the top 2 words on the stack, collect word id, pos tag id and label id of the first and second
    # leftmost and rightmost child, and leftmost of leftmost and rightmost of rightmost child.
    # Collected Sw = 2*6 = 12, St = 2*6 = 12, Sl = 2*6 = 12
    for index in range(0, 2):
        stackIndex = c.getStack(index)
        for childIndex in [c.getLeftChild(stackIndex, 1), c.getRightChild(stackIndex, 1),
                           c.getLeftChild(stackIndex, 2), c.getRightChild(stackIndex, 2),
                           c.getLeftChild(c.getLeftChild(stackIndex, 1), 1),
                           c.getRightChild(c.getRightChild(stackIndex, 1), 1)]:
            words.append(getWordID(c.getWord(childIndex)))
            pos_tags.append(getPosID(c.getPOS(childIndex)))
            labels.append(getLabelID(c.getLabel(childIndex)))

    return list(words + pos_tags + labels)


def genTrainExamples(sents, trees):
    numTrans = parsing_system.numTransitions()

    features = []
    labels = []
    pbar = ProgressBar()
    for i in pbar(list(range(len(sents)))):
        if trees[i].isProjective():
            c = parsing_system.initialConfiguration(sents[i])

            while not parsing_system.isTerminal(c):
                oracle = parsing_system.getOracle(c, trees[i])
                feat = getFeatures(c)
                label = []
                for j in range(numTrans):
                    t = parsing_system.transitions[j]
                    if t == oracle:
                        label.append(1.)
                    elif parsing_system.canApply(c, t):
                        label.append(0.)
                    else:
                        label.append(-1.)

                if 1.0 not in label:
                    print(i, label)
                features.append(feat)
                labels.append(label)
                c = parsing_system.apply(c, oracle)
    return features, labels


def load_embeddings(filename, wordDict, posDict, labelDict):
    dictionary, word_embeds = pickle.load(open(filename, 'rb'), encoding="latin1")

    embedding_array = np.zeros((len(wordDict) + len(posDict) + len(labelDict), Config.embedding_size))
    knownWords = list(wordDict.keys())
    foundEmbed = 0
    for i in range(len(embedding_array)):
        index = -1
        if i < len(knownWords):
            w = knownWords[i]
            if w in dictionary:
                index = dictionary[w]
            elif w.lower() in dictionary:
                index = dictionary[w.lower()]
        if index >= 0:
            foundEmbed += 1
            embedding_array[i] = word_embeds[index]
        else:
            embedding_array[i] = np.random.rand(Config.embedding_size) * 0.02 - 0.01
    print("Found embeddings: ", foundEmbed, "/", len(knownWords))

    return embedding_array


if __name__ == '__main__':

    wordDict = {}
    posDict = {}
    labelDict = {}
    parsing_system = None

    trainSents, trainTrees = Util.loadConll('train.conll')
    devSents, devTrees = Util.loadConll('dev.conll')
    testSents, _ = Util.loadConll('test.conll')
    genDictionaries(trainSents, trainTrees)

    embedding_filename = 'word2vec.model'

    embedding_array = load_embeddings(embedding_filename, wordDict, posDict, labelDict)

    labelInfo = []
    for idx in np.argsort(list(labelDict.values())):
        labelInfo.append(list(labelDict.keys())[idx])
    parsing_system = ParsingSystem(labelInfo[1:])
    print(parsing_system.rootLabel)

    print("Generating Traning Examples")
    trainFeats, trainLabels = genTrainExamples(trainSents, trainTrees)
    print("Done.")

    # Build the graph model
    graph = tf.Graph()
    model = DependencyParserModel(graph, embedding_array, Config)

    num_steps = Config.max_iter
    with tf.Session(graph=graph) as sess:

        model.train(sess, num_steps)

        model.evaluate(sess, testSents)

