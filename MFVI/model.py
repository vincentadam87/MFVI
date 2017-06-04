import tensorflow as tf
import numpy as np
from settings import float_type, jitter_level,std_qmu_init, np_float_type, np_int_type
from functions import eye, variational_expectations
from kullback_leiblers import gauss_kl_white, gauss_kl_white_diag, gauss_kl, gauss_kl_diag
from quadrature import hermgauss



class MFVI(object):

    def __init__(self, X, Y, likelihood,sample=True):
        """
        Mean Field Variational inference with Gaussian Posterior
        :param X: N x D 
        :param Y: N x R
        :param likelihood:  
        :param q_diag: 
        """
        self.likelihood = likelihood
        self.X = X
        self.Y = Y
        self.D = X.get_shape()[1]
        self.num_latent = Y.get_shape()[1]
        self.num_data = Y.get_shape()[0]
        self.initialize_prior()
        self.initialize_inference()
        self.sample=sample


    def initialize_prior(self):

        with tf.variable_scope("prior") as scope:
            self.s = tf.get_variable("s", [1],\
                                initializer=tf.constant_initializer(1., dtype=float_type))

    def initialize_inference(self):

        with tf.variable_scope("inference") as scope:
            self.q_A_mu, self.q_A_sqrt = [], []
            shape= (self.D,self.num_latent)
            q_A_mu = np.random.randn(shape[0],shape[1])*std_qmu_init
            self.q_A_mu = tf.get_variable("q_mu", shape, \
                                          initializer=tf.constant_initializer(q_A_mu, dtype=float_type))
            q_A_sqrt = np.ones(shape)
            self.q_A_sqrt = tf.get_variable("q_sqrt", shape, \
                                            initializer=tf.constant_initializer(q_A_sqrt,dtype=float_type))

    def build_prior_KL(self):
        S = np.square(self.s)*eye(self.D) # diagonal prior v*I
        KL = gauss_kl_diag(self.q_A_mu, self.q_A_sqrt, S)
        return KL



    def build_likelihood(self):
        """
        This gives a variational bound on the model likelihood.
        """
        # Get prior KL.
        KL = self.build_prior_KL()

        if self.sample:
            samples_pred = self.sample_predictor(self.X)
            var_exp = self.likelihood.logp(samples_pred, tf.expand_dims(self.Y, -1))
            var_exp = tf.reduce_mean(var_exp, -1)
        else:
            # Get conditionals
            fmean, fvar = self.build_predictor(self.X)
            # Get variational expectations.
            var_exp = self.likelihood.variational_expectations(fmean, fvar, self.Y)


        return tf.reduce_sum(var_exp)  - KL


    def build_predict(self, Xnew):
        """ Output D x N x R """
        mu = tf.einsum('dr,nd->dnr', self.q_A_mu, Xnew)
        var = tf.einsum('dr,nd->dnr', tf.square(self.q_A_sqrt), tf.square(Xnew))
        return mu,var

    def sample_predict(self, Xnew):
        """ Output D x N x R """
        mu, var = self.build_predict(Xnew)
        num_samples = 5
        shape = mu.get_shape().as_list() + [num_samples]
        return tf.random_normal(shape,tf.expand_dims(mu,-1),tf.expand_dims(var,-1))

    def sample_predictor(self,Xnew):
        """ Output N x R """
        samples = self.sample_predict(Xnew)
        return tf.reduce_sum(samples, 0)

    def build_predictor(self, Xnew):
        """ Output N x R x nsamp """
        e_mean, e_var = self.build_predict(Xnew)
        return tf.reduce_sum(e_mean, 0), tf.reduce_sum(e_var, 0)

class MFVI2(MFVI):

    def __init__(self, X, Y, likelihood, indices=None):

        MFVI.__init__(self, X, Y, likelihood, sample=True)
        self.indices = indices


    def sample_predictor(self, Xnew):
        """ Output N x R x nsamp """
        samples = self.sample_predict(Xnew)
        s = tf.reduce_sum(tf.gather(samples,self.indices[0]),0)
        for i in range(1,len(self.indices)):
            s *= tf.reduce_sum(tf.gather(samples,self.indices[i]),0)
        return s


