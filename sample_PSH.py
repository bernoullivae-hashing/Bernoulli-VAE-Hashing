

import tensorflow as tf
import numpy as np
from utils import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

EPS = 1e-10

def sample_gumbel(shape, eps=1e-20):
    """Sample from Gumbel(0, 1)"""
    U = tf.random_uniform(shape, minval=0, maxval=1)
    return -tf.log(-tf.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    """ Draw a sample from the Gumbel-Softmax distribution"""
    y = logits + sample_gumbel(tf.shape(logits))
    return tf.nn.softmax( y / temperature)


def gumbel_softmax(logits, temperature, hard=False):
    """Sample from the Gumbel-Softmax distribution and optionally discretize"""
    y = gumbel_softmax_sample(logits, temperature)
    if hard:
        k = tf.shape(logits)[-1]
        y_hard = tf.cast(tf.equal(y, tf.reduce_max(y, -1, keepdims=True)), tf.float32)
        y = tf.stop_gradient(y_hard - y) + y
    return y


class pnash(tf.keras.Model):
    
    def __init__(self, latent_dim, n_feas, hidden_dim, nlabels, temperature=1.0,\
                 keep_prob=0.5, learning_rate=0.001, decay_rate=0.96):
        
        super(pnash, self).__init__()
        self.n_feas = n_feas
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.nlabels = nlabels
        self.gamma = 0.5
        self.tau = temperature
        self.keep_prob = keep_prob
        self.use_cross_entropy = True
        # inference network q(z|x)
        self.encoder = tf.keras.Sequential([tf.keras.layers.Dense(self.hidden_dim,\
                    activation = "relu", input_shape = (self.n_feas,),\
                    kernel_initializer = tf.random_uniform_initializer(minval = -np.sqrt(6.0 / (self.n_feas + self.hidden_dim)),\
                    maxval = np.sqrt(6.0 / (self.n_feas + self.hidden_dim))),\
                    bias_initializer = tf.constant_initializer(0.0)),\
                    tf.keras.layers.Dense(self.hidden_dim, activation = "relu",\
                    kernel_initializer = tf.random_uniform_initializer(minval = -np.sqrt(6.0 / (self.hidden_dim + self.hidden_dim)),\
                    maxval = np.sqrt(6.0 / (self.hidden_dim + self.hidden_dim))),\
                    bias_initializer=tf.constant_initializer(0.0)),\
                    tf.keras.layers.Dropout(rate=1-self.keep_prob),\
                    tf.keras.layers.Dense(1 * self.latent_dim,\
                    kernel_initializer = tf.random_uniform_initializer(minval = -np.sqrt(6.0 / (1*self.latent_dim + self.hidden_dim)),\
                    maxval = np.sqrt(6.0 / (1*self.latent_dim + self.hidden_dim))),\
                    bias_initializer=tf.constant_initializer(0.0))])
        
        # noise variance network
        
        self.pred_label = tf.keras.layers.Dense(self.nlabels, activation="sigmoid", input_shape = (1*self.latent_dim,),\
                    kernel_initializer = tf.random_uniform_initializer(minval = -np.sqrt(6.0 / (2*self.latent_dim + self.nlabels)),\
                    maxval = np.sqrt(6.0 / (2*self.latent_dim + self.nlabels))),\
                    bias_initializer=tf.constant_initializer(0.0))
        # decoder
        self.decoder = tf.keras.layers.Dense(self.n_feas, input_shape = (1*self.latent_dim,),\
                    kernel_initializer = tf.random_uniform_initializer(minval = -np.sqrt(6.0 / (1*self.latent_dim + self.n_feas)),\
                    maxval = np.sqrt(6.0 / (1*self.latent_dim + self.n_feas))),\
                    bias_initializer=tf.constant_initializer(0.0))
        
        # optimizer
        lr = 0.0005 #tf.train.exponential_decay(learning_rate, step, 10000, decay_rate, staircase=True, name="lr")
        self.optimizer1 = tf.train.AdamOptimizer(learning_rate = lr)
        self.optimizer2 = tf.train.AdamOptimizer(learning_rate = lr)
    
    def reconst_err(self, logprob_word, docs):
        return -tf.reduce_sum(logprob_word * docs, axis=1, keepdims=True)
    
    def kl_loss(self, q_z, cat_dim=2):
        log_q_z = tf.log(q_z + EPS)
        return tf.reduce_mean(tf.reduce_sum(q_z * (log_q_z - tf.log(self.gamma)), axis=-1))
    
    def pred_loss(self, labels, tag_prob):
        if not self.use_cross_entropy:
            return tf.reduce_sum(tf.pow(tag_prob - labels, 2), axis=1)
        else:
            return -tf.reduce_mean(tf.reduce_sum(labels * tf.log(tf.maximum(tag_prob, 1e-10))\
                           + (1 - labels) * tf.log(tf.maximum(1 - tag_prob, 1e-10)), axis=1))
    
    def arm_grad(self, docs, mu):
        q = self.encoder(docs)
        score1 = tf.sigmoid(-q)
        z1 = 0.5 * (tf.sign(mu - score1) + 1.0)
        logprob_w1 = tf.nn.log_softmax(self.decoder(z1), axis=-1)
        score2 = 1 - score1
        z2 = 0.5 * (tf.sign(score2 - mu) + 1.0)
        logprob_w2 = tf.nn.log_softmax(self.decoder(z2), axis=-1)
        
        F1 = self.reconst_err(logprob_w1, docs)
        F2 = self.reconst_err(logprob_w2, docs)
        g = tf.tile(F1 - F2, [1, self.latent_dim]) * (mu - 0.5)  # 1 X latent_dim
        g_delta = tf.convert_to_tensor(g, dtype=tf.float32)
        return g_delta
    
    def update(self, docs1, docs2, y1, y2, oh_y1, oh_y2, kl_weight, sim_weight, tag_weight):
        
        docs1 = docs1.astype('float32')
        docs1 = docs1.reshape((-1, self.n_feas))
        docs2 = docs2.astype('float32')
        docs2 = docs2.reshape((-1, self.n_feas))
        
        mu = tf.random_uniform([tf.shape(docs1)[0], self.latent_dim])
        g_delta1 = self.arm_grad(docs1, mu)
        g_delta2 = self.arm_grad(docs2, mu)
        
        with tf.GradientTape(persistent=True) as tape:
            
            # first doc
            logits1 = self.encoder(docs1.reshape((-1, self.n_feas)))
            score1 = tf.sigmoid(logits1)
            z1 = 0.5 * (tf.sign(score1 - mu) + 1.0)
            logprob_w1 = tf.nn.log_softmax(self.decoder(z1), axis=-1)
            kl1 = tf.reduce_mean(tf.reduce_sum(score1 * (tf.log(score1 + EPS) - tf.log(self.gamma + EPS))\
                               + (1 - score1) * (tf.log(1 - score1 + EPS) - tf.log(1 - self.gamma + EPS))))
            reconstr_err1 = tf.reduce_mean(self.reconst_err(logprob_w1, docs1))
            
            # second doc
            logits2 = self.encoder(docs2.reshape((-1, self.n_feas)))
            score2 = tf.sigmoid(logits2)
            z2 = 0.5 * (tf.sign(score2 - mu) + 1.0)
            logprob_w2 = tf.nn.log_softmax(self.decoder(z2), axis=-1)
            kl2 = tf.reduce_mean(tf.reduce_sum(score2 * (tf.log(score2 + EPS) - tf.log(self.gamma + EPS))\
                               + (1 - score2) * (tf.log(1 - score2 + EPS) - tf.log(1 - self.gamma + EPS))))
            reconstr_err2 = tf.reduce_mean(self.reconst_err(logprob_w2, docs2))
            
            # build loss
            sgn = 2 * tf.cast(y1==y2, tf.float32) - 1
            tag_prob1 = self.pred_label(tf.stop_gradient(z1-score1)+score1)
            pred_loss1 = self.pred_loss(oh_y1, tag_prob1)
            tag_prob2 = self.pred_label(tf.stop_gradient(z2-score2)+score2)
            pred_loss2 = self.pred_loss(oh_y2, tag_prob2)
            
            reconstr_err = reconstr_err1 + reconstr_err2
            
            loss_fn = tf.reduce_mean(tf.reduce_sum(g_delta1 * logits1 + g_delta2 * logits2, axis=1)) + kl_weight * (kl1 + kl2) +\
                      tag_weight * (pred_loss1 + pred_loss2) + sim_weight * sgn * tf.nn.l2_loss(score1 - score2)
            
            loss_fn2 = reconstr_err + tag_weight * (pred_loss1 + pred_loss2)
        
        grad = tape.gradient(loss_fn , self.encoder.variables) 
        self.optimizer1.apply_gradients(zip(grad, self.encoder.variables)) 
        # decoder variables
        var_list = [v for v in self.decoder.variables]
        for v in self.pred_label.variables:
            var_list.append(v)
        grad2 = tape.gradient(loss_fn2 , var_list) 
        self.optimizer2.apply_gradients(zip(grad2, var_list)) 
        #print(loss_fn)
        
    def transform(self, docs):
        z_data = []
        for i in tqdm(range(len(docs))):
            doc = docs[i]
            doc = doc.astype('float32')
            self.keep_prob = 1.0
            logits = self.encoder(doc.reshape((-1, self.n_feas)))
            score = tf.sigmoid(logits)
            mu = 0.5 #tf.random_uniform([1, self.latent_dim])
            z = 0.5 * (tf.sign(score - mu) + 1.0)
            z_data.append(z.numpy()[0])
        return z_data


if __name__=="__main__":
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.75)
    tf.enable_eager_execution(config=tf.ConfigProto(gpu_options=gpu_options))
    
    filename = './VariationalDeepSemanticHashing-master/dataset/ng20.tfidf.mat'
    data = Load_Dataset(filename)
    
    seedi = 0
    tf.set_random_seed(seedi)
    np.random.seed(seedi)
    
    # initialization
    latent_dim = 32
    hidden_dim = 500
    keep_prob = 0.8
    nlabels = 20
    learner = pnash(latent_dim, data.n_feas, hidden_dim, nlabels, 1.0, keep_prob)
    
    # labels
    ntrain = len(data.train)
    labels = np.argmax(data.gnd_train, axis=1)
    # training
    nepoch = 300
    nbatch = 200
    kl_weight = 0.01
    sim_weight = 0.0009
    tag_weight = 0.1
    tag_inc = 0.1
    max_tag_weight = 10.
    kl_inc = 1 / 5000. # set the annealing rate for KL loss
    idxlist = list(range(ntrain))
    for epoch in range(nepoch):
        print('epoch ', epoch, flush=True)
        idx = np.random.permutation(ntrain)
        np.random.shuffle(idxlist)
        for i in range(0, ntrain, nbatch):
            end_idx = min(i + nbatch, ntrain)
            docs1 = data.train[idxlist[i:end_idx]]
            docs2 = data.train[idx[idxlist[i:end_idx]]]
            y1 = labels[idxlist[i:end_idx]]
            y2 = labels[idx[i:end_idx]]
            learner.update(docs1, docs2, y1, y2, data.gnd_train[idxlist[i:end_idx]], data.gnd_train[idx[i:end_idx]],\
                           kl_weight, sim_weight, tag_weight)
            #kl_weight = min(kl_weight + kl_inc, 1.0)
            tag_weight = min(tag_weight + tag_inc, max_tag_weight)
            learner.tau = max(learner.tau * 0.96, 0.1)
    # run experiment here
    zTrain = learner.transform(data.train)
    zTest = learner.transform(data.test)
    zTrain = np.array(zTrain)
    zTest = np.array(zTest)

    cbTrain = zTrain.astype(int) 
    cbTest = zTest.astype(int) 
    
    TopK=100
    print('Retrieve Top{} candidates using hamming distance'.format(TopK))
    results = run_topK_retrieval_experiment(cbTrain, cbTest, data.gnd_train, data.gnd_test, TopK)
    
