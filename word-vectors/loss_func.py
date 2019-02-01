import tensorflow as tf

def cross_entropy_loss(inputs, true_w):
    """
    ==========================================================================

    inputs: The embeddings for context words. Dimension is [batch_size, embedding_size].
    true_w: The embeddings for predicting words. Dimension of true_w is [batch_size, embedding_size].

    Write the code that calculate A = log(exp({u_o}^T v_c))

    A =


    And write the code that calculate B = log(\sum{exp({u_w}^T v_c)})


    B =

    ==========================================================================
    """

    # get the batch size from the inputs tensor
    batch_size = inputs.shape[0]

    # for each input embedding calculate A
    # the variables use names which are self explanatory for the above given formula
    u_o = true_w
    u_oT = tf.transpose(u_o)
    v_c = inputs
    A = tf.reshape(tf.diag_part(tf.matmul(v_c, u_oT)), [1, batch_size])

    # for each input embedding calculate B
    # the variables use names which are self explanatory for the above given formula
    u_w = true_w
    u_wT = tf.transpose(u_w)
    term = tf.matmul(v_c, u_wT)
    term_exp = tf.exp(term)
    summation = tf.reduce_sum(term_exp, 1)
    B = tf.log(summation)

    return tf.subtract(B, A)

def nce_loss(inputs, weights, biases, labels, sample, unigram_prob):
    """
    ==========================================================================

    inputs: Embeddings for context words. Dimension is [batch_size, embedding_size].
    weigths: Weights for nce loss. Dimension is [Vocabulary, embeeding_size].
    biases: Biases for nce loss. Dimension is [Vocabulary, 1].
    labels: Word_ids for predicting words. Dimesion is [batch_size, 1].
    samples: Word_ids for negative samples. Dimension is [num_sampled].
    unigram_prob: Unigram probability. Dimesion is [Vocabulary].

    Implement Noise Contrastive Estimation Loss Here

    ==========================================================================
    """

    # get the batch size, vocabulary size and number of negative samples
    batch_size = labels.shape[0]
    vocabulary = biases.shape[0]
    samples = sample.shape[0]

    # reshape and convert to tensors
    labels = tf.reshape(labels, [batch_size])
    biases = tf.reshape(biases, [vocabulary, 1])
    sample = tf.convert_to_tensor(sample)
    unigram_prob = tf.reshape(tf.convert_to_tensor(unigram_prob), [vocabulary, 1])

    # lookup label and sample weights
    label_weights = tf.nn.embedding_lookup(weights, labels)
    sample_weights = tf.nn.embedding_lookup(weights, sample)

    # lookup biases for labels and samples
    label_biases = tf.nn.embedding_lookup(biases, labels)
    sample_biases = tf.nn.embedding_lookup(biases, sample)

    # lookup unigram probabilities for labels and samples
    label_unigrams = tf.nn.embedding_lookup(unigram_prob, labels)
    sample_unigrams = tf.nn.embedding_lookup(unigram_prob, sample)

    # following variable names self describe their usage in the nce equation
    u_cTu_o = tf.reshape(tf.diag_part(tf.matmul(inputs, tf.transpose(label_weights))), [batch_size, 1])
    u_cTu_x = tf.matmul(sample_weights, tf.transpose(inputs))

    # calculate log[kPr(w_o)] and log[kPr(w_x)]
    k_pr_w_o = tf.scalar_mul(samples, label_unigrams)
    log_k_pr_w_o = tf.log(k_pr_w_o + 1e-10)
    k_pr_w_x = tf.scalar_mul(samples, sample_unigrams)
    log_k_pr_w_x = tf.log(k_pr_w_x + 1e-10)

    # calculate s(wo, wc) and s(wx, wc)
    s_w_o_w_c = tf.add(u_cTu_o, label_biases)
    s_w_x_w_c = tf.add(u_cTu_x, sample_biases)

    # calculate the remaining terms and their log sigmoids as in the formula
    left_term = s_w_o_w_c - log_k_pr_w_o
    right_term = s_w_x_w_c - log_k_pr_w_x
    log_sigmoid_left = tf.transpose(tf.log_sigmoid(left_term))
    sigmoid_right = tf.sigmoid(right_term)
    one_minus_sigmoid_right = tf.ones(sigmoid_right.shape) - sigmoid_right
    log_sigmoid_right = tf.log(one_minus_sigmoid_right + 1e-10)

    # perform summation for the right side term
    right_summation = tf.reduce_sum(log_sigmoid_right, 0, keepdims=True)

    # add up both the terms and add negative sign
    J = log_sigmoid_left + right_summation
    J = tf.scalar_mul(-1, J)

    return J
