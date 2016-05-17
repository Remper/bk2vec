import tensorflow as tf


def _sum_rows(x):
    """Returns a vector summing up each row of the matrix x."""
    # _sum_rows(x) is equivalent to math_ops.reduce_sum(x, 1) when x is
    # a matrix.  The gradient of _sum_rows(x) is more efficient than
    # reduce_sum(x, 1)'s gradient in today's implementation. Therefore,
    # we use _sum_rows(x) in the nce_loss() computation since the loss
    # is mostly used for training.
    cols = tf.shape(x)[1]
    ones_shape = tf.pack([cols, 1])
    ones = tf.ones(ones_shape, x.dtype)
    return tf.reshape(tf.matmul(x, ones), [-1])


def nce_loss(weights, biases, inputs, labels, num_sampled, num_classes,
             num_true=1,
             sampled_values=None,
             name="nce_loss"):
    if not isinstance(weights, list):
        weights = [weights]

    with tf.op_scope(
                    weights + [biases, inputs, labels], name, "compute_sampled_logits"):
        if labels.dtype != tf.dtypes.int64:
            labels = tf.cast(labels, tf.dtypes.int64)
        labels_flat = tf.reshape(labels, [-1])

        # Sample the negative labels.
        #   sampled shape: [num_sampled] tensor
        #   true_expected_count shape = [batch_size, 1] tensor
        #   sampled_expected_count shape = [num_sampled] tensor
        if sampled_values is None:
            sampled_values = tf.nn.log_uniform_candidate_sampler(
                true_classes=labels,
                num_true=num_true,
                num_sampled=num_sampled,
                unique=True,
                range_max=num_classes)
        # NOTE: pylint cannot tell that 'sampled_values' is a sequence
        # pylint: disable=unpacking-non-sequence
        sampled, true_expected_count, sampled_expected_count = sampled_values
        # pylint: enable=unpacking-non-sequence

        # labels_flat is a [batch_size * num_true] tensor
        # sampled is a [num_sampled] int tensor
        all_ids = tf.concat(0, [labels_flat, sampled])

        # weights shape is [num_classes, dim]
        all_w = tf.gather(weights, all_ids)
        all_b = tf.gather(biases, all_ids)
        # true_w shape is [batch_size * num_true, dim]
        # true_b is a [batch_size * num_true] tensor
        true_w = tf.slice(
            all_w, [0, 0], tf.pack([tf.shape(labels_flat)[0], -1]))
        true_b = tf.slice(all_b, [0], tf.shape(labels_flat))

        # inputs shape is [batch_size, dim]
        # true_w shape is [batch_size * num_true, dim]
        # row_wise_dots is [batch_size, num_true, dim]
        dim = tf.shape(true_w)[1:2]
        new_true_w_shape = tf.concat(0, [[-1, num_true], dim])
        row_wise_dots = tf.mul(
            tf.expand_dims(inputs, 1),
            tf.reshape(true_w, new_true_w_shape))
        # We want the row-wise dot plus biases which yields a
        # [batch_size, num_true] tensor of true_logits.
        dots_as_matrix = tf.reshape(row_wise_dots, tf.concat(0, [[-1], dim]))
        true_logits = tf.reshape(_sum_rows(dots_as_matrix), [-1, num_true])
        true_b = tf.reshape(true_b, [-1, num_true])
        true_logits += true_b

        # Lookup weights and biases for sampled labels.
        #   sampled_w shape is [num_sampled, dim]
        #   sampled_b is a [num_sampled] float tensor
        sampled_w = tf.slice(all_w, tf.pack([tf.shape(labels_flat)[0], 0]), [-1, -1])
        sampled_b = tf.slice(all_b, tf.shape(labels_flat), [-1])

        # inputs has shape [batch_size, dim]
        # sampled_w has shape [num_sampled, dim]
        # sampled_b has shape [num_sampled]
        # Apply X*W'+B, which yields [batch_size, num_sampled]
        sampled_logits = tf.matmul(inputs,
                                   sampled_w,
                                   transpose_b=True) + sampled_b

        # Subtract log of Q(l), prior probability that l appears in sampled.
        true_logits -= tf.log(true_expected_count)
        sampled_logits -= tf.log(sampled_expected_count)

        # Construct output logits and labels. The true labels/logits start at col 0.
        logits = tf.concat(1, [true_logits, sampled_logits])
        # true_logits is a float tensor, ones_like(true_logits) is a float tensor
        # of ones. We then divide by num_true to ensure the per-example labels sum
        # to 1.0, i.e. form a proper probability distribution.
        labels = tf.concat(
            1, [tf.ones_like(true_logits) / num_true,
                tf.zeros_like(sampled_logits)])

    sampled_losses = tf.nn.sigmoid_cross_entropy_with_logits(logits,
                                                             labels,
                                                             name="sampled_losses")
    # sampled_losses is batch_size x {true_loss, sampled_losses...}
    # We sum out true and sampled losses.
    return _sum_rows(sampled_losses)
