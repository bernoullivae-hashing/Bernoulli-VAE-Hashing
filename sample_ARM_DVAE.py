

import tensorflow as tf
import numpy as np
from utils import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

EPS = 1e-10

class nash(tf.keras.Model):
    
    def __init__(self, latent_dim, n_feas, hidden_dim, keep_prob=0.5, learning_rate=0.001):
        
        super(nash, self).__init__()
        self.n_feas = n_feas
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.gamma = 0.5
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
                    tf.keras.layers.Dense(self.latent_dim,\
                    kernel_initializer = tf.random_uniform_initializer(minval = -np.sqrt(6.0 / (self.latent_dim + self.hidden_dim)),\
                    maxval = np.sqrt(6.0 / (self.latent_dim + self.hidden_dim))),\
                    bias_initializer=tf.constant_initializer(0.0))])
        
        # noise variance network
        self.logsigma = tf.keras.layers.Dense(1, activation="sigmoid", input_shape = (self.latent_dim,),\
                      kernel_initializer = tf.random_uniform_initializer(minval = -np.sqrt(6.0 / (1 + self.latent_dim)),\
                      maxval = np.sqrt(6.0 / (1 + self.latent_dim))),\
                      bias_initializer = tf.constant_initializer(0.0))
        
        
        # decoder variables
        self.decoder = tf.keras.layers.Dense(self.n_feas, input_shape = (self.latent_dim,),\
                    kernel_initializer = tf.random_uniform_initializer(minval = -np.sqrt(6.0 / (self.latent_dim + self.n_feas)),\
                    maxval = np.sqrt(6.0 / (self.latent_dim + self.n_feas))),\
                    bias_initializer=tf.constant_initializer(0.0))
                    
        
        # optimizer
        lr = 0.0005 #tf.train.exponential_decay(learning_rate, step, 10000, decay_rate, staircase=True, name="lr")
        self.optimizer1 = tf.train.AdamOptimizer(learning_rate = lr)
        self.optimizer2 = tf.train.AdamOptimizer(learning_rate = lr)
    
    def reconst_err(self, z, doc, word_indice):
        # decoder
        ze = -tf.matmul(z, self.E) + self.b
        p_x = tf.squeeze(tf.nn.softmax(ze))
        p_x_scores = tf.gather(p_x, word_indice)
        weight_scores = tf.gather(tf.squeeze(doc), word_indice)
        reconstr_err = -tf.reduce_sum(tf.log(tf.maximum(p_x_scores * weight_scores, 1e-10)))
        return reconstr_err
    
    def reconst_err_v2(self, logprob_word, docs):
        return -tf.reduce_sum(logprob_word * docs, axis=1, keepdims=True)
    
    def kl_loss(self, q_z, cat_dim=2):
        log_q_z = tf.log(q_z + EPS)
        return tf.reduce_mean(tf.reduce_sum(q_z * (log_q_z - tf.log(self.gamma)), axis=-1))
    
    def update(self, docs, kl_weight):
        
        docs = docs.astype('float32')
        docs = docs.reshape((-1, self.n_feas))
        q = self.encoder(docs)
        logvar = self.logsigma(q)
        mu = tf.random_uniform([tf.shape(q)[0], self.latent_dim])
        epsilon = tf.random_normal([tf.shape(q)[0], self.latent_dim])
        score1 = tf.sigmoid(-q)
        z1 = 0.5 * (tf.sign(mu - score1) + 1.0)
        z1 = z1 + tf.exp(0.5 * logvar) * epsilon
        logprob_w1 = tf.nn.log_softmax(self.decoder(z1), axis=-1)
        score2 = 1 - score1
        z2 = 0.5 * (tf.sign(score2 - mu) + 1.0)
        z2 = z2 + tf.exp(0.5 * logvar) * epsilon
        logprob_w2 = tf.nn.log_softmax(self.decoder(z2), axis=-1)
        
        F1 = self.reconst_err_v2(logprob_w1, docs)    #nbatch * 1
        F2 = self.reconst_err_v2(logprob_w2, docs)
        g = tf.tile(F1 - F2, [1, self.latent_dim]) * (mu - 0.5)  # nbatch X latent_dim
        mask = tf.cast(tf.math.not_equal(z1, z2), tf.float32)
        g = mask * g
        g_delta = tf.convert_to_tensor(g, dtype=tf.float32)
        
        with tf.GradientTape(persistent=True) as tape:
            
            logits = self.encoder(docs.reshape((-1, self.n_feas)))
            logvar = self.logsigma(logits)
            score = tf.sigmoid(logits)
            z = 0.5 * (tf.sign(score - mu) + 1.0)
            z = z + tf.exp(0.5 * logvar) * epsilon
            logprob_w = tf.nn.log_softmax(self.decoder(z), axis=-1)
            kl = tf.reduce_mean(tf.reduce_sum(score * (tf.log(score + EPS) - tf.log(self.gamma + EPS))\
                               + (1 - score) * (tf.log(1 - score + EPS) - tf.log(1 - self.gamma + EPS)), axis=1))
            reconstr_err = tf.reduce_mean(self.reconst_err_v2(logprob_w, docs))
            
            # build loss
            loss_fn = tf.reduce_sum(g_delta * logits) + kl_weight * kl + reconstr_err
            neg_elbo = reconstr_err + kl_weight * kl
        
        grad = tape.gradient(loss_fn , self.encoder.variables) 
        self.optimizer1.apply_gradients(zip(grad, self.encoder.variables)) 
        # decoder variables
        var_list = [v for v in self.decoder.variables]
        for v in self.logsigma.variables:
            var_list.append(v)
        grad2 = tape.gradient(reconstr_err , var_list) 
        self.optimizer2.apply_gradients(zip(grad2, var_list))
        return neg_elbo
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
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    tf.enable_eager_execution(config=tf.ConfigProto(gpu_options=gpu_options))
    
    filename = './VariationalDeepSemanticHashing-master/dataset/ng20.tfidf.mat'
    data = Load_Dataset(filename)
    
    seedi = 0
    tf.set_random_seed(seedi)
    np.random.seed(seedi)
    
    # initialization
    latent_dim = 32
    hidden_dim = 500
    keep_prob = 1.0
    nlabels = 20
    tau0 = 1.0
    ANNEAL_RATE = 0.00005
    MIN_TEMP = 0.5
    learner = nash(latent_dim, data.n_feas, hidden_dim, keep_prob)
    
    # labels
    ntrain = len(data.train)
    # training
    nepoch = 400
    nbatch = 20
    kl_weight = 0.01
    kl_inc = 1 / 5000. # set the annealing rate for KL loss
    idxlist = list(range(ntrain))
    for epoch in range(nepoch):
        print('epoch ', epoch, flush=True)
        epoch_loss = []
        epoch_err = []
        np.random.shuffle(idxlist)
        for i in range(0, ntrain, nbatch):
            end_idx = min(i + nbatch, ntrain)
            docs = data.train[idxlist[i:end_idx]]
            loss = learner.update(docs, kl_weight)
            #kl_weight = min(kl_weight + kl_inc, 1.0)
            epoch_loss.append(loss)

        print('loss ', np.mean(epoch_loss), flush=True)

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
    
