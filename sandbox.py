from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class GatherTest(tf.test.TestCase):
    use_gpu = False

    def testScalar1D(self):
        with self.test_session(use_gpu=self.use_gpu):
            params = tf.constant([0, 1, 2, 3, 7, 5])
            indices = tf.constant(4)
            gather_t = tf.gather(params, indices)
            gather_val = gather_t.eval()
        self.assertAllEqual(7, gather_val)
        self.assertEqual([], gather_t.get_shape())

    def testLogUniformCandidateSampler(self):
        with self.test_session(use_gpu=self.use_gpu):
            labels = tf.constant([1, 0, 1, 0, 1, 0], dtype=tf.int64, shape=[6, 1])
            num_true = 1
            num_sampled = 2
            num_classes = 2
            result = tf.nn.log_uniform_candidate_sampler(
                true_classes=labels,
                num_true=num_true,
                num_sampled=num_sampled,
                unique=True,
                range_max=num_classes
            )[0].eval()
        self.assertNotEqual([num_sampled], result.shape)

    def testCrossEntropy(self):
        with self.test_session(use_gpu=self.use_gpu):
            logits = tf.constant([0.5, 0.5, 10.0, 0.5])
            labels = tf.constant([0.0, 0.0, 1.0, 0.0])
            result = tf.nn.sigmoid_cross_entropy_with_logits(logits,
                                                    labels).eval()
        self.assertNotEqual([], result.shape)

    def testIdentity(self):
        with self.test_session(use_gpu=self.use_gpu):
            test = tf.constant(10)
            result = tf.identity(test).eval()
        self.assertEqual(10, result)

    def testNCELoss(self):
        with self.test_session(use_gpu=self.use_gpu):
            inputs = tf.constant([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
            weights = tf.constant([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.5, 0.5], [0.3, 0.3]])
            biases = tf.constant([0.5, 0.5])
            labels = tf.constant([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
            num_sampled = 2
            result = tf.nn.nce_loss(
                weights,
                biases,
                inputs,
                labels,
                num_sampled,
                6).eval()


class GatherGpuTest(GatherTest):
    use_gpu = True


tf.test.main()
