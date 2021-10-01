import os
import numpy as np
import pandas as pd
from absl import flags, app, logging


import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors

flags.DEFINE_integer("nboot", 400, 'bootstrap')
flags.DEFINE_string("csv", "./do_modelu_grawitacyjnego.csv", "Input file")
flags.DEFINE_string("out", "checkpoint", "Input file")
FLAGS = flags.FLAGS


class PoissonGravitationalModel(tf.Module):
    dt = np.float64
    def __init__(self, name=None):
        super(PoissonGravitationalModel, self).__init__(name=name)
        self.w = tf.Variable(tf.convert_to_tensor([0.7,0.9], dtype=self.dt),name='w')
        self.c = tf.Variable(tf.convert_to_tensor([0,0], dtype=self.dt),name='c')
        self.logits=tf.Variable(np.zeros(3,dtype=self.dt), name='logits')
        self.lograte=tf.Variable(tf.convert_to_tensor(-10, dtype=self.dt),name='lograte')
        #self.lograte=tf.convert_to_tensor(-10, dtype=dt)
    

    def __call__(self, x):
        x=tf.convert_to_tensor(x, dtype=self.dt)
        ydist=tfd.Mixture(
            cat=tfd.Categorical(logits=self.logits+tf.zeros((tf.shape(x)[0],1), dtype=self.dt)),
            components=[
                tfd.Poisson(log_rate=tf.zeros_like(x)+self.lograte), # approximate point mass at 0
                tfd.Poisson(log_rate=self.w[0]*x+self.c[0]),
                tfd.Poisson(log_rate=self.w[1]*x+self.c[1]),
            ]
        )
        return ydist

    def betas(self):
        return np.concatenate([np.array([0]),self.w.numpy()])



def interactions(df,bootstrap=False):
    if bootstrap:
        df = df.sample(df.shape[0], replace=True)
    A=np.array(df)
    C=np.zeros((A.shape[1],A.shape[1]))

    for v in A:
        v = np.sign(v)
        M = v[...,np.newaxis]@v[np.newaxis,...]
        M-=np.diag(np.diag(M))
        C+= M
    S = df.sum().to_numpy()[..., np.newaxis]
    return C,S

class Estimator:
    dt = np.float64

    def __init__(self, data,model, bootstrap=True):

        self.df = data
        self.flat_idx = np.triu_indices(self.df.shape[1],k = 1)
        self.interaction_to_xy(*interactions(self.df,bootstrap))
        self.model = model

    def interaction_to_xy(self, C,S):
        flatc = C[self.flat_idx]
        sst = S@S.T
        sst_flat = sst[self.flat_idx]

        x=np.log1p(sst_flat).astype(self.dt)
        y=flatc.astype(self.dt)

        self.x,self.y = x,y
    
    @tf.function(jit_compile=True)
    def loss_fn(self):
            return -tf.reduce_sum(self.model(self.x).log_prob(self.y))
        
    def fit(self):
        losses = tfp.math.minimize( self.loss_fn, 
            num_steps=5000, 
            optimizer=tf.optimizers.Adam(learning_rate=0.08),
            trainable_variables = self.model.trainable_variables
            )
        return losses

    def plot(self):
        sidx = np.argsort(self.x)
        ydist = self.model(self.x[sidx])

        Z = np.argmax(np.stack([c.prob(self.y[sidx]) for c in ydist.components],axis=1)*ydist.cat.probs_parameter(), axis=1)
        betas = self.model.betas()

        plt.figure(figsize=(11,9))
        plt.scatter(self.x[sidx],np.log1p(self.y[sidx]), 
            alpha=0.8,
            c=betas[Z],
            cmap='Greens'
            )


        # for i1,i2,x_,y_ in zip(*flat_idx,x,y):
        #     if y_ >0:
        #         plt.text(x_,np.log1p(y_),str(set(self.df.columns[[i1,i2]])), fontsize=4)


        for c in ydist.components:
            plt.plot(self.x[sidx],np.log1p(c.mean()))
        #plt.ylim(-0.1,25)
        #plt.yscale('log')
        plt.ylabel(r'$\log(\lambda_{ij}+1)$')
        plt.xlabel(r'$\log(c_{i}c_{j}+1)$')
        plt.title("Model grawitacyjny relacji w funkcji patentów par krajów")
        plt.savefig('gen/pois_grav_oth.pdf')


        ydist = self.model(self.x)

        Z = np.argmax(np.stack([c.prob(self.y) for c in ydist.components],axis=1)*ydist.cat.probs_parameter(), axis=1)


        plt.figure(figsize=(9,11))
        In = np.zeros(2*[self.df.shape[1]])
        betas = self.model.betas()

        In[self.flat_idx]=betas[Z]
        In[(self.flat_idx[1],self.flat_idx[0])]=betas[Z]

        sns.heatmap(In,cmap='Greens',linewidths=0.2, linecolor='black')
        plt.xticks(ticks=np.arange(0,29)+0.5,labels=list(self.df.columns), rotation=90);
        plt.yticks(ticks=np.arange(0,29)+0.5,labels=list(self.df.columns), rotation=0);
        plt.gca().set_aspect('equal')
        plt.savefig('gen/grid_pois_grav_oth.pdf')

    def save(self, directory:str):
        checkpoint = tf.train.Checkpoint(model=self.model)
        checkpoint.write(os.path.join(directory,self.__class__.__name__))

    


def main(_):

    df = pd.read_csv(FLAGS.csv,index_col=0)

    for i in range(FLAGS.nboot):
        try:
            dirname = os.path.join(FLAGS.out,str(i))
            os.makedirs(dirname)
        except:
            pass
        e = Estimator(
            data=df,
            model=PoissonGravitationalModel(),
            bootstrap=True
        )
        e.fit()
        e.save(dirname)



if __name__ == '__main__':
    app.run(main)