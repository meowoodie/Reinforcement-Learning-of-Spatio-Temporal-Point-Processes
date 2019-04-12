import os
import sys
import arrow
import utils
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tfgen import SpatialTemporalHawkes, MarkedSpatialTemporalLSTM

os.environ['KMP_DUPLICATE_LIB_OK']='True'

class RL_Hawkes_Generator(object):
    """
    Reinforcement Learning Based Point Process Generator
    """

    def __init__(self, T, S, layers, n_comp, batch_size, C=1., maximum=1e+3, keep_latest_k=None, lr=1e-5, eps=0.2):
        """
        Params:
        - T: the maximum time of the sequences
        - S: the space of location
        - C: the constant in diffusion kernel
        """
        # model hyper-parameters
        self.T          = T          # time space
        self.S          = S          # location space
        self.batch_size = batch_size # batch size
        self.maximum    = maximum    # upper bound of the conditional intensity
        # Hawkes process generator
        self.hawkes     = SpatialTemporalHawkes(T, S, layers=layers, n_comp=n_comp, C=C, maximum=1e+3, verbose=False)
        # input tensors: expert sequences (time, location)
        self.input_expert_seqs    = tf.placeholder(tf.float32, [batch_size, None, 3])
        self.input_learner_seqs   = tf.placeholder(tf.float32, [batch_size, None, 3])
        # TODO: make esp decay exponentially
        # coaching
        self.coached_learner_seqs = self._coaching(self.input_learner_seqs, self.input_expert_seqs, eps=eps)
        self.learner_seqs_loglik  = self._log_likelihood(learner_seqs=self.coached_learner_seqs, keep_latest_k=keep_latest_k)
        # build policy optimizer
        self._policy_optimizer(
            expert_seqs=self.input_expert_seqs, 
            learner_seqs=self.coached_learner_seqs,
            learner_seqs_loglik=self.learner_seqs_loglik, 
            lr=lr)
    
    def _log_likelihood(self, learner_seqs, keep_latest_k):
        """
        compute the log-likelihood of the input data given the hawkes point process. 
        """
        # max length of the sequence in learner_seqs
        max_len   = tf.shape(learner_seqs)[1]
        # log-likelihoods
        logliklis = []
        for b in range(self.batch_size):
            seq       = learner_seqs[b, :, :]
            mask_t    = tf.cast(seq[:, 0] > 0, tf.float32)
            trunc_seq = tf.boolean_mask(seq, mask_t)
            seq_len   = tf.shape(trunc_seq)[0]
            # calculate the log conditional pdf for each of data points in the sequence.
            loglikli  = tf.scan(
                lambda a, i: self.hawkes.log_conditional_pdf(trunc_seq[:i, :], keep_latest_k=keep_latest_k),
                tf.range(1, seq_len+1), # from the first point to the last point
                initializer=np.array(0., dtype=np.float32))
            # padding zeros for loglikli
            paddings  = tf.zeros(max_len - seq_len, dtype=tf.float32)
            loglikli  = tf.concat([loglikli, paddings], axis=0)
            logliklis.append(loglikli)
        logliklis = tf.expand_dims(tf.stack(logliklis, axis=0), -1)
        return logliklis

    def _policy_optimizer(self, expert_seqs, learner_seqs, learner_seqs_loglik, lr):
        """policy optimizer"""
        # concatenate batches in the sequences
        concat_expert_seq         = self.__concatenate_batch(expert_seqs)          # [batch_size * expert_seq_len, data_dim]
        concat_learner_seq        = self.__concatenate_batch(learner_seqs)         # [batch_size * learner_seq_len, data_dim]
        concat_learner_seq_loglik = self.__concatenate_batch(learner_seqs_loglik)  # [batch_size * learner_seq_len, 1]

        # calculate average rewards
        print("[%s] building reward." % arrow.now(), file=sys.stderr)
        reward = self._reward(concat_expert_seq, concat_learner_seq) 
        # TODO: record the discrepency

        # cost and optimizer
        print("[%s] building optimizer." % arrow.now(), file=sys.stderr)
        self.cost      = tf.reduce_sum(tf.multiply(reward, concat_learner_seq_loglik), axis=0) / self.batch_size
        # self.cost      = tf.reduce_sum( \
        #                  tf.reduce_sum(tf.reshape(reward, [self.batch_size, tf.shape(learner_seqs)[1]]), axis=1) * \
        #                  tf.reduce_sum(tf.reshape(concat_learner_seq_loglik, [self.batch_size, tf.shape(learner_seqs)[1]]), axis=1))  / self.batch_size
        # Adam optimizer
        global_step    = tf.Variable(0, trainable=False)
        learning_rate  = tf.train.exponential_decay(lr, global_step, decay_steps=100, decay_rate=0.99, staircase=True)
        self.optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.6, beta2=0.9).minimize(self.cost, global_step=global_step)

    def _reward(self, expert_seq, learner_seq, kb=5): 
        """reward function"""
        # get mask for concatenated expert and learner sequences
        learner_mask_t = tf.expand_dims(tf.cast(learner_seq[:, 0] > 0, tf.float32), -1)
        expert_mask_t  = tf.expand_dims(tf.cast(expert_seq[:, 0] > 0, tf.float32), -1)

        # calculate mask for kernel matrix
        learner_learner_kernel_mask = tf.matmul(learner_mask_t, tf.transpose(learner_mask_t))
        expert_learner_kernel_mask  = tf.matmul(expert_mask_t, tf.transpose(learner_mask_t))

        # calculate upper-half kernel matrix
        # - [learner_seq_len, learner_seq_len], [expert_seq_len, learner_seq_len]
        learner_learner_kernel, expert_learner_kernel = self.__kernel_matrix(learner_seq, expert_seq, kb)

        learner_learner_kernel = tf.multiply(learner_learner_kernel, learner_learner_kernel_mask)
        expert_learner_kernel  = tf.multiply(expert_learner_kernel, expert_learner_kernel_mask)

        # calculate reward for each of data point in learner sequence
        emp_ll_mean = tf.reduce_sum(learner_learner_kernel, axis=0) / self.batch_size # [batch_size * learner_seq_len]
        emp_el_mean = tf.reduce_sum(expert_learner_kernel, axis=0) / self.batch_size  # [batch_size * learner_seq_len]
        return tf.expand_dims(emp_ll_mean - emp_el_mean, -1)                          # [batch_size * learner_seq_len, 1]

    def _coaching(self, learner_seqs, expert_seqs, eps):
        """
        coach the learner by replacing part of generated learner sequences with the expert 
        sequence for the (greedy) exploration.
        """
        # align learner and expert sequences
        learner_seqs, expert_seqs, seq_len = self.__align_learner_expert_seqs(learner_seqs, expert_seqs)
        # coaching and retain mask
        p             = tf.random_uniform([self.batch_size, 1, 1], 0, 1)    # [batch_size, 1]
        coaching_mask = tf.tile(tf.cast(p <= eps, dtype=tf.float32), [1, seq_len, 3]) # [batch_size, 1]
        retain_mask   = 1. - coaching_mask
        # replace part of learner sequences by expert sequences
        learner_seqs  = tf.multiply(learner_seqs, retain_mask) + tf.multiply(expert_seqs, coaching_mask)
        return learner_seqs
        
    @staticmethod
    def __align_learner_expert_seqs(learner_seqs, expert_seqs):
        """
        align learner sequences and expert sequences, i.e., make two batch of sequences have the same 
        sequence length by padding zeros to the tail.
        """
        batch_size       = tf.shape(learner_seqs)[0]
        learner_seq_len  = tf.shape(learner_seqs)[1]
        expert_seq_len   = tf.shape(expert_seqs)[1]
        max_seq_len      = tf.cond(tf.less(learner_seq_len, expert_seq_len), 
            lambda: expert_seq_len, lambda: learner_seq_len)
        learner_paddings = tf.zeros([batch_size, max_seq_len - learner_seq_len, 3])
        expert_paddings  = tf.zeros([batch_size, max_seq_len - expert_seq_len, 3])
        learner_seqs     = tf.concat([learner_seqs, learner_paddings], axis=1)
        expert_seqs      = tf.concat([expert_seqs, expert_paddings], axis=1)
        return learner_seqs, expert_seqs, max_seq_len
    
    @staticmethod
    def __concatenate_batch(seqs):
        """Concatenate each batch of the sequences into a single sequence."""
        array_seq = tf.unstack(seqs, axis=0)     # [batch_size, seq_len, data_dim]
        seq       = tf.concat(array_seq, axis=0) # [batch_size*seq_len, data_dim]
        return seq
 
    @staticmethod
    def __kernel_matrix(learner_seq, expert_seq, kernel_bandwidth):
        """
        Construct kernel matrix based on learn sequence and expert sequence, each entry of the matrix 
        is the distance between two data points in learner_seq or expert_seq. return two matrix, left_mat 
        is the distances between learn sequence and learn sequence, right_mat is the distances between 
        learn sequence and expert sequence.
        """
        # calculate l2 distances
        learner_learner_mat = utils.l2_norm(learner_seq, learner_seq) # [batch_size*learner_seq_len, batch_size*learner_seq_len]
        expert_learner_mat  = utils.l2_norm(expert_seq, learner_seq)  # [batch_size*expert_seq_len, batch_size*learner_seq_len]
        # exponential kernel
        learner_learner_mat = tf.exp(-learner_learner_mat / kernel_bandwidth)
        expert_learner_mat  = tf.exp(-expert_learner_mat / kernel_bandwidth)
        return learner_learner_mat, expert_learner_mat

    def train(self, sess, 
            epoches,               # number of epoches (how many times is the entire dataset going to be trained)
            expert_seqs,           # [n, seq_len, 3]
            trainplot=True,        # plot the change of intensity over epoches
            pretrained=False):
        """Train the point process generator given expert sequences."""

        # initialization
        if not pretrained:
            print("[%s] parameters are initialized." % arrow.now(), file=sys.stderr)
            # initialize network parameters
            init_op = tf.global_variables_initializer()
            sess.run(init_op)

        # data configurations
        # - number of expert sequences
        n_data    = expert_seqs.shape[0]
        # - number of batches
        n_batches = int(n_data / self.batch_size)
        
        # if trainplot:
        #     ppim = utils.PointProcessDistributionMeter(self.T, self.S, self.batch_size)

        # training over epoches
        for epoch in range(epoches):
            # shuffle indices of the training samples
            shuffled_ids = np.arange(n_data)
            np.random.shuffle(shuffled_ids)

            # training over batches
            avg_train_cost = []
            for b in range(n_batches):
                idx             = np.arange(self.batch_size * b, self.batch_size * (b + 1))
                # training and testing indices selected in current batch
                batch_train_ids = shuffled_ids[idx]
                # training and testing batch data
                batch_train_expert  = expert_seqs[batch_train_ids, :, :]
                batch_train_learner = self.hawkes.sampling(sess, self.batch_size)
                # optimization procedure
                sess.run(self.optimizer, feed_dict={
                    self.input_expert_seqs:  batch_train_expert, 
                    self.input_learner_seqs: batch_train_learner})
                # cost for train batch and test batch
                train_cost = sess.run(self.cost, feed_dict={
                    self.input_expert_seqs:  batch_train_expert, 
                    self.input_learner_seqs: batch_train_learner})
                print("[%s] batch training cost: %.2f." % (arrow.now(), train_cost), file=sys.stderr)
                # record cost for each batch
                avg_train_cost.append(train_cost)

                # if trainplot:
                #     # update distribution plot
                #     ppim.update_time_distribution(batch_train_learner[:, : , 0], batch_train_expert[:, :, 0])
                #     ppim.update_location_distribution(batch_train_learner[:, : , 1:], batch_train_expert[:, :, 1:])

            # training log output
            avg_train_cost = np.mean(avg_train_cost)
            print('[%s] Epoch %d (n_train_batches=%d, batch_size=%d)' % \
                (arrow.now(), epoch, n_batches, self.batch_size), file=sys.stderr)
            print('[%s] Training cost:\t%f' % (arrow.now(), avg_train_cost), file=sys.stderr)

if __name__ == "__main__":
	# Unittest example

	# np.random.seed(0)
	# tf.set_random_seed(1)

    data = np.load('../Spatio-Temporal-Point-Process-Simulator/data/apd.robbery.permonth.npy')
    # data = np.load('../Spatio-Temporal-Point-Process-Simulator/data/northcal.earthquake.perseason.npy')
    da   = utils.DataAdapter(init_data=data)
    seqs = da.normalize(data)
    seqs = seqs[:, 1:, :] # remove the first element in each seqs, since t = 0
    print(da)
    print(seqs.shape)

    # training model
    with tf.Session() as sess:
        # model configuration
        batch_size = 10
        epoches    = 5
        lr         = 1e-3
        T          = [0., 10.]
        S          = [[-1., 1.], [-1., 1.]]
        layers     = [5]
        n_comp     = 5

        ppg = RL_Hawkes_Generator(
            T=T, S=S, layers=layers, n_comp=n_comp, batch_size=batch_size, 
            C=1., maximum=1e+3, keep_latest_k=None, lr=lr, eps=0)

        ppg.train(sess, epoches, seqs, trainplot=False)

        ppg.hawkes.save_params_npy(sess, 
            path="../Spatio-Temporal-Point-Process-Simulator/data/robbery_rl_gaussian_mixture_params.npz")
