from layers import *

flags = tf.app.flags
FLAGS = flags.FLAGS

class Optimizer():
    def __init__(self, supp, main, preds, labels, num_nodes, num_edges):
        pos_weight = float(num_nodes ** 2 - num_edges) / num_edges
        norm = num_nodes ** 2 / float((num_nodes ** 2 - num_edges) * 2)

        labels_sub = labels

        self.loss_supp = 0
        for viewRec in supp:
            self.loss_supp += norm * tf.reduce_mean(
                tf.nn.weighted_cross_entropy_with_logits(
                    logits=tf.reshape(viewRec, [-1]), targets=labels_sub, pos_weight=pos_weight))

        self.loss_main = norm * tf.reduce_mean(
            tf.nn.weighted_cross_entropy_with_logits(
                logits=tf.reshape(main, [-1]), targets=labels_sub, pos_weight=pos_weight))

        self.loss_preds = norm * tf.reduce_mean(
            tf.nn.weighted_cross_entropy_with_logits(
                logits=tf.reshape(preds, [-1]), targets=labels_sub, pos_weight=pos_weight))

        self.cost = 1 * self.loss_main + \
                    FLAGS.Alpha * self.loss_supp + \
                    FLAGS.Beta * self.loss_preds

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)  # Adam Optimizer

        self.opt_op = self.optimizer.minimize(self.cost)
        self.grads_vars = self.optimizer.compute_gradients(self.cost)
