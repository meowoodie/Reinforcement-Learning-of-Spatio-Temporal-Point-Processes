import sys
import arrow
import utils
import numpy as np
import tensorflow as tf

from tfgen import SpatialTemporalHawkes

class MLE_Hawkes_Generator(object):
    """
    Reinforcement Learning Based Point Process Generator
    """

    def __init__(self, T, S, layers, n_comp, batch_size, C=1., data_dim=3, keep_latest_k=None, lr=1e-3, reg_scale=0.):
        """
        Params:
        - T: the maximum time of the sequences
        - S: the space of location
        - C: the constant in diffusion kernel
        - batch_size:    batch size of the training data
        - maximum:       upper bound of the conditional intensity
        - data_dim:      data dimension (=3 by default)
        - keep_latest_k: only compute latest k points in log-likelihood calculation
        - lr:            learning rate for the SGD optimizer
        """
        self.batch_size = batch_size
        # Hawkes process
        self.hawkes     = SpatialTemporalHawkes(T, S, layers=layers, n_comp=n_comp, C=C, maximum=1e+3, verbose=False)
        # regularization
        l1_regularizer  = tf.contrib.layers.l1_regularizer(scale=reg_scale, scope=None)
        penalty_term    = tf.contrib.layers.apply_regularization(l1_regularizer, self.hawkes.Wss)
        # input tensors: expert sequences (time, location, marks)
        self.input_seqs = tf.placeholder(tf.float32, [batch_size, None, data_dim]) # [batch_size, seq_len, data_dim]
        self.cost       = -1 * self.log_likelihood(S, keep_latest_k=keep_latest_k) / batch_size # + penalty_term
        # Adam optimizer
        global_step    = tf.Variable(0, trainable=False)
        learning_rate  = tf.train.exponential_decay(lr, global_step, decay_steps=100, decay_rate=0.99, staircase=True)
        self.optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.6, beta2=0.9).minimize(self.cost, global_step=global_step)

    def log_likelihood(self, S, keep_latest_k):
        """
        compute the log-likelihood of the input data given the hawkes point process. 
        """
        # log-likelihood
        loglikli = 0.
        for b in range(batch_size):
            seq       = self.input_seqs[b, :, :]
            # mask_t    = tf.cast(seq[:, 0] > 0, tf.float32)
            # trunc_seq = tf.boolean_mask(seq, mask_t)
            # seq_len   = tf.shape(trunc_seq)[0]
            # # calculate the log conditional pdf for each of data points in the sequence.
            # loglikli += tf.reduce_sum(tf.scan(
            #     lambda a, i: self.hawkes.log_conditional_pdf(trunc_seq[:i, :], keep_latest_k=keep_latest_k),
            #     tf.range(1, seq_len+1), # from the first point to the last point
            #     initializer=np.array(0., dtype=np.float32)))
            loglikli += self.hawkes.log_likelihood(seq)
        return loglikli

    def train(self, sess, 
            epoches,               # number of epoches (how many times is the entire dataset going to be trained)
            expert_seqs,           # [n, seq_len, data_dim=3]
            pretrained=False):
        """train the point process generator given expert sequences."""

        # initialization
        if not pretrained:
            # initialize network parameters
            init_op = tf.global_variables_initializer()
            sess.run(init_op)
            print("[%s] parameters are initialized." % arrow.now(), file=sys.stderr)

        # data configurations
        # - number of expert sequences
        n_data    = expert_seqs.shape[0]
        # - number of batches
        n_batches = int(n_data / batch_size)

        # training over epoches
        for epoch in range(epoches):
            # shuffle indices of the training samples
            shuffled_ids = np.arange(n_data)
            np.random.shuffle(shuffled_ids)

            # training over batches
            avg_train_cost = []
            for b in range(n_batches):
                idx              = np.arange(batch_size * b, batch_size * (b + 1))
                # training and testing indices selected in current batch
                batch_train_ids  = shuffled_ids[idx]
                # training and testing batch data
                batch_train_seqs = expert_seqs[batch_train_ids, :, :]
                # optimization procedure
                sess.run(self.optimizer, feed_dict={self.input_seqs: batch_train_seqs})
                # cost for train batch and test batch
                train_cost = sess.run(self.cost, feed_dict={self.input_seqs: batch_train_seqs})
                print("[%s] batch training cost: %.2f." % (arrow.now(), train_cost), file=sys.stderr)
                # record cost for each batch
                avg_train_cost.append(train_cost)

            # training log output
            avg_train_cost = np.mean(avg_train_cost)
            print('[%s] Epoch %d (n_train_batches=%d, batch_size=%d)' % (arrow.now(), epoch, n_batches, batch_size), file=sys.stderr)
            print('[%s] Training cost:\t%f' % (arrow.now(), avg_train_cost), file=sys.stderr)

            

if __name__ == "__main__":
    # Unittest example

    # data = np.load('../Spatio-Temporal-Point-Process-Simulator/data/apd.robbery.permonth.npy')
    data = np.load('../Spatio-Temporal-Point-Process-Simulator/data/northcal.earthquake.perseason.npy')
    da   = utils.DataAdapter(init_data=data)
    seqs = da.normalize(data)
    seqs = seqs[:, 1:, :] # remove the first element in each seqs, since t = 0
    print(da)
    print(seqs.shape)

    # training model
    with tf.Session() as sess:
        S          = [[-1., 1.], [-1., 1.]]
        T          = [0., 10.]
        batch_size = 10
        epoches    = 10
        layers     = [5]

        ppg = MLE_Hawkes_Generator(
            T=T, S=S, layers=layers, n_comp=5,
            batch_size=batch_size, data_dim=3, 
            keep_latest_k=None, lr=1e-1, reg_scale=0.)
        
        ppg.train(sess, epoches, seqs)

        ppg.hawkes.save_params_npy(sess, 
            path="../Spatio-Temporal-Point-Process-Simulator/data/earthquake_mle_gaussian_mixture_params.npz")