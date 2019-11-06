#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Imitation Learning for Point Process

A LSTM based model for generating marked spatial-temporal points.

References:
- https://arxiv.org/abs/1811.05016

Dependencies:
- Python 3.6.7
- tensorflow==1.5.0
"""

import sys
import arrow
import utils
import numpy as np
import tensorflow as tf

from stppg import GaussianMixtureDiffusionKernel, HawkesLam, SpatialTemporalPointProcess

class SpatialTemporalHawkes(object):
    """
    Customized Spatial Temporal Hawkes

    A Hawkes model parametrized by multi-layers neural networks, which provides flexible self-exciting 
    points pattern.
    """

    def __init__(self, T, S, layers=[20, 20], n_comp=5, C=1., maximum=1e+3, verbose=False):
        """
        """
        # constant hyper parameters
        self.INIT_PARAM  = .01
        self.SIGMA_SHIFT = .1
        self.SIGMA_SCALE = .5
        self.MU_SCALE    = .01
        # configurations
        self.C       = C       # constant
        self.T       = T       # time space
        self.S       = S       # location space
        self.maximum = maximum # upper bound of conditional intensity
        self.verbose = verbose
        # model parameters
        self.mu      = tf.constant(0.1) # tf.get_variable(name="mu", initializer=tf.constant(0.1), dtype=tf.float32)
        self.beta    = tf.constant(1.)  # tf.get_variable(name="beta", initializer=tf.constant(1.), dtype=tf.float32)
        self.Wss     = []
        self.bss     = []
        self.Wphis   = []
        # construct multi-layers neural networks
        # - define the layers where 2 is for the input layer (x and y); 
        #   And 5 is for the output layer (mu_x, mu_y, sigma_x, sigma_y, rho)
        self.layers = [2] + layers + [5]
        # - define the number of the components in Gaussian mixture diffusion kernel
        self.n_comp = n_comp
        # - construct component weighting vectors
        for k in range(self.n_comp):
            Wphi = tf.get_variable(name="Wphi%d" % k, 
                initializer=self.INIT_PARAM * tf.random.normal(shape=[2, 1]),
                dtype=tf.float32)
            self.Wphis.append(Wphi)
            # - construct weight & bias matrix layer by layer for each of Gaussian components
            Ws = []
            bs = []
            for i in range(len(self.layers)-1):
                # random initialization
                W = tf.get_variable(name="W%d%d" % (k, i), 
                    initializer=self.INIT_PARAM * tf.random.normal(shape=[self.layers[i], self.layers[i+1]]),
                    dtype=tf.float32)
                b = tf.get_variable(name="b%d%d" % (k, i), 
                    initializer=self.INIT_PARAM * tf.random.normal(shape=[self.layers[i+1]]),
                    dtype=tf.float32)
                Ws.append(W)
                bs.append(b)
            self.Wss.append(Ws)
            self.bss.append(bs)

    def sampling(self, sess, batch_size):
        """fetch model parameters, and generate samples accordingly."""
        # get current model parameters
        mu, beta = sess.run([self.mu, self.beta])
        Wss      = sess.run(self.Wss)
        bss      = sess.run(self.bss)
        Wphis    = sess.run(self.Wphis)
        # construct kernel function and conditional intensity lambda
        kernel   = GaussianMixtureDiffusionKernel(
            self.n_comp, layers=self.layers[1:-1], beta=beta, C=self.C, 
            SIGMA_SHIFT=self.SIGMA_SHIFT, SIGMA_SCALE=self.SIGMA_SCALE, MU_SCALE=self.MU_SCALE,
            Wss=Wss, bss=bss, Wphis=Wphis)
        lam      = HawkesLam(mu, kernel, maximum=self.maximum)
        # sampling points given model parameters
        pp       = SpatialTemporalPointProcess(lam)
        seqs, sizes = pp.generate(T=self.T, S=self.S, batch_size=batch_size, verbose=self.verbose)
        return seqs

    def _nonlinear_mapping(self, k, s):
        """nonlinear mapping from location space to parameters space"""
        # construct multi-layers neural networks
        output = s # [n_his, 2]
        for i in range(len(self.layers)-1):
            output = tf.nn.sigmoid(tf.nn.xw_plus_b(output, self.Wss[k][i], self.bss[k][i])) # [n_his, n_b]
        # project to parameters space
        mu_x    = (output[:, 0] - 0.5) * 2 * self.MU_SCALE           # [n_his]: mu_x spans (-MU_SCALE, MU_SCALE)
        mu_y    = (output[:, 1] - 0.5) * 2 * self.MU_SCALE           # [n_his]: mu_y spans (-MU_SCALE, MU_SCALE)
        sigma_x = output[:, 2] * self.SIGMA_SCALE + self.SIGMA_SHIFT # [n_his]: sigma_x spans (SIGMA_SHIFT, SIGMA_SHIFT + SIGMA_SCALE)
        sigma_y = output[:, 3] * self.SIGMA_SCALE + self.SIGMA_SHIFT # [n_his]: sigma_y spans (SIGMA_SHIFT, SIGMA_SHIFT + SIGMA_SCALE)
        rho     = output[:, 4] * 1.5 - .75                           # [n_his]: rho spans (-.75, .75)
        return mu_x, mu_y, sigma_x, sigma_y, rho

    def _gaussian_kernel(self, k, t, s, his_t, his_s):
        """
        A Gaussian diffusion kernel function based on the standard kernel function proposed 
        by Musmeci and Vere-Jones (1992). The angle and shape of diffusion ellipse is able  
        to vary according to the location. 

        k indicates the k-th gaussian component that is used to compute the nonlinear mappings.  
        """
        eps     = 1e-8            # IMPORTANT: Avoid delta_t be zero
        delta_t = t - his_t + eps # [n_his]
        delta_s = s - his_s       # [n_his, 2]
        delta_x = delta_s[:, 0]   # [n_his]
        delta_y = delta_s[:, 1]   # [n_his]
        mu_x, mu_y, sigma_x, sigma_y, rho = self._nonlinear_mapping(k, his_s)
        return tf.exp(- self.beta * delta_t) * \
            (self.C / (2 * np.pi * sigma_x * sigma_y * delta_t * tf.sqrt(1 - tf.square(rho)))) * \
            tf.exp((- 1. / (2 * delta_t * (1 - tf.square(rho)))) * \
                ((tf.square(delta_x - mu_x) / tf.square(sigma_x)) + \
                (tf.square(delta_y - mu_y) / tf.square(sigma_y)) - \
                (2 * rho * (delta_x - mu_x) * (delta_y - mu_y) / (sigma_x * sigma_y))))
    
    def _softmax(self, s, k):
        """
        Gaussian mixture components are weighted by phi^k, which are computed by a softmax function, i.e., 
        phi^k(x, y) = e^{[x y]^T w^k} / \sum_{i=1}^K e^{[x y]^T w^i}
        """
        # s:        [n_his, 2]
        # Wphis[k]: [2, 1]
        numerator   = tf.exp(tf.matmul(s, self.Wphis[k]))                 # [n_his, 1]
        denominator = tf.concat([ 
            tf.exp(tf.matmul(s, self.Wphis[i])) 
            for i in range(self.n_comp) ], axis=1)                        # [n_his, K=n_comp]
        phis = tf.squeeze(numerator) / tf.reduce_sum(denominator, axis=1) # [n_his]
        return phis
    
    def _gaussian_mixture_kernel(self, t, s, his_t, his_s):
        """
        A Gaussian mixture diffusion kernel function is superposed by multiple Gaussian diffusion 
        kernel function. The number of the Gaussian components is specified by n_comp. 
        """
        nus = []
        for k in range(self.n_comp):
            phi = self._softmax(his_s, k)                            # [n_his]             
            nu  = phi * self._gaussian_kernel(k, t, s, his_t, his_s) # [n_his]
            nu  = tf.expand_dims(nu, -1)                             # [n_his, 1]
            nus.append(nu)                                           # K * [n_his, 1]
        nus = tf.concat(nus, axis=1)      # [n_his, K]
        return tf.reduce_sum(nus, axis=1) # [n_his]

    def _lambda(self, t, s, his_t, his_s):
        """lambda function for the Hawkes process."""
        lam = self.mu + tf.reduce_sum(self._gaussian_mixture_kernel(t, s, his_t, his_s))
        return lam

    def log_conditional_pdf(self, points, keep_latest_k=None):
        """log pdf conditional on history."""
        if keep_latest_k is not None: 
            points   = points[-keep_latest_k:, :]
        # number of the points
        len_points   = tf.shape(points)[0]
        # variables for calculating triggering probability
        s, t         = points[-1, 1:], points[-1, 0]
        his_s, his_t = points[:-1, 1:], points[:-1, 0]

        def pdf_no_history():
            return tf.log(tf.clip_by_value(self._lambda(t, s, his_t, his_s), 1e-8, 1e+10))
        
        def pdf_with_history():
            # triggering probability
            log_trig_prob = tf.log(tf.clip_by_value(self._lambda(t, s, his_t, his_s), 1e-8, 1e+10))
            # variables for calculating tail probability
            tn, ti        = points[-2, 0], points[:-1, 0]
            t_ti, tn_ti   = t - ti, tn - ti
            # tail probability
            # TODO: change to gaussian mixture (add phi)
            log_tail_prob = - \
                self.mu * (t - tn) * utils.lebesgue_measure(self.S) - \
                tf.reduce_sum(tf.scan(
                    lambda a, i: self.C * (tf.exp(- self.beta * tn_ti[i]) - tf.exp(- self.beta * t_ti[i])) / \
                        tf.clip_by_value(self.beta, 1e-8, 1e+10),
                    tf.range(tf.shape(t_ti)[0]),
                    initializer=np.array(0., dtype=np.float32)))
            return log_trig_prob + log_tail_prob
        # TODO: Unsolved issue:
        #       pdf_with_history will still be called even if the condition is true, which leads to exception
        #       "ValueError: slice index -1 of dimension 0 out of bounds." due to that points is empty but we 
        #       try to index a nonexisted element.
        #       However, when points is indexed in a scan loop, this works fine and the numerical result is 
        #       also correct. which is very confused to me. Therefore, I leave this problem here temporarily.
        log_cond_pdf = tf.cond(tf.less(len_points, 2), 
            pdf_no_history,   # if there is only one point in the sequence
            pdf_with_history) # if there is more than one point in the sequence
        return log_cond_pdf

    def log_likelihood(self, points):
        """log likelihood of given points"""
        loglikli  = 0.                                    # loglikelihood initialization
        mask_t    = tf.cast(points[:, 0] > 0, tf.float32) # time mask
        trunc_seq = tf.boolean_mask(points, mask_t)       # truncate the sequence and get the valid part
        seq_len   = tf.shape(trunc_seq)[0]                # length of the sequence
        # term 1: product of lambda
        loglikli += tf.reduce_sum(tf.scan(
            lambda a, i: tf.log(self._lambda(trunc_seq[i, 0], trunc_seq[i, 1:], trunc_seq[:i, 0], trunc_seq[:i, 1:])),
            tf.range(seq_len),
            initializer=np.array(0., dtype=np.float32)))
        # term 2: 1 - F^*(T)
        ti        = points[:, 0]
        zero_ti   = 0 - ti
        T_ti      = self.T[1] - ti
        loglikli -= tf.reduce_sum(tf.scan(
            lambda a, i: self.C * (tf.exp(- self.beta * zero_ti[i]) - tf.exp(- self.beta * T_ti[i])) / \
                tf.clip_by_value(self.beta, 1e-8, 1e+10),
            tf.range(tf.shape(ti)[0]),
            initializer=np.array(0., dtype=np.float32)))
        return loglikli

    def save_params_npy(self, sess, path):
        """save parameters into numpy file."""
        Wss      = sess.run(self.Wss)
        bss      = sess.run(self.bss)
        Wphis    = sess.run(self.Wphis)
        mu, beta = sess.run([self.mu, self.beta])
        print(Wss)
        print(Wphis)
        np.savez(path, Wss=Wss, bss=bss, Wphis=Wphis, mu=mu, beta=beta)



if __name__ == "__main__":
    # Unittest example
    np.random.seed(1)
    tf.set_random_seed(1)

    with tf.Session() as sess:
        hawkes = SpatialTemporalHawkes(
            T=[0., 10.], S=[[-1., 1.], [-1., 1.]], 
            layers=[5], n_comp=3, C=1., maximum=1e+3, verbose=True)

        points = tf.constant([
            [ 1.16898147e-02,  1.45831794e-01, -3.05314839e-01],
            [ 4.81481478e-02, -1.25229925e-01,  8.72766301e-02],
            [ 1.13194443e-01, -3.87020826e-01,  2.80696362e-01],
            [ 1.60300925e-01, -2.42807735e-02, -5.64230382e-01],
            [ 1.64004624e-01,  7.10764453e-02, -1.77927762e-01],
            [ 1.64236113e-01,  6.51166216e-02, -6.82414293e-01],
            [ 2.05671296e-01, -4.48017061e-01,  5.36620915e-01],
            [ 2.12152779e-01, -3.20064761e-02, -2.08911732e-01]], dtype=tf.float32) 

        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        # t = points[-1, 0]
        # s = points[-1, 1:]
        # his_t = points[:-1, 0]
        # his_s = points[:-1, 1:]

        # res = sess.run(hawkes.log_conditional_pdf(points))
        # res = sess.run(hawkes._lambda(t, s, his_t, his_s))
        # res = sess.run(hawkes._softmax(his_s, 0))
        # res = sess.run(hawkes._gaussian_kernel(0, t, s, his_t, his_s))

        # seq_len = tf.shape(points)[0]
        # r = tf.scan(
        #     lambda a, i: hawkes._lambda(points[i, 0], points[i, 1:], points[:i, 0], points[:i, 1:]), 
        #     tf.range(seq_len), # from the first point to the last point
        #     initializer=np.array(0., dtype=np.float32))
        r = hawkes.log_likelihood(points)
        print(sess.run(r))

        # # test sampling
        # seqs = hawkes.sampling(sess, batch_size=10)
        # print(seqs)