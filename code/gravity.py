import os
import numpy as np
import pandas as pd
from absl import flags, app, logging
from enum import Enum

import matplotlib as mpl

import data

mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors

flags.DEFINE_integer("nboot", 1, 'bootstrap')
flags.DEFINE_string("pickle", "dane/clean.pickle", "Input file")
flags.DEFINE_string("out", "checkpoint", "Output dir")
flags.DEFINE_bool('others',False,'Czy kraje z poza UE')
flags.DEFINE_bool('treinablezero',True, "czy rozklad 0 kooperacji jest trenowalny")
flags.DEFINE_integer('nnz',1, "Liczba niezerowych składowych mieszanki")
flags.DEFINE_string('feature_type','ALL','''ONLY_COOPERATION=1
    INDIVIDUAL=2
    ALL=3''')
FLAGS = flags.FLAGS


class PoissonGravitationalModel(tf.Module):
    def __init__(self,trainable_lograte=True,nnz=1,dtype=np.float64, name=None):
        '''

        :param trainable_lograte: Czy rozklad braku interakcji jest trenowlany
        :param nnz: ile skąłdowych w mieszance poza rozkaldem zerowym
        :param dtype: typ dancyh
        :param name: Opcjonalna nazwa modułu
        '''
        super(PoissonGravitationalModel, self).__init__(name=name)
        dt = dtype
        self.dt=dt
        self.nnz = nnz
        self.w = tf.Variable(tf.convert_to_tensor(np.linspace(0.1,1.1,self.nnz, dtype=self.dt)),name='w')
        self.c = tf.Variable(tf.convert_to_tensor(nnz*[0], dtype=dt),name='c')
        self.logits=tf.Variable(np.zeros(nnz+1,dtype=dt), name='logits')
        if trainable_lograte:
            self.lograte=tf.Variable(tf.convert_to_tensor(-1, dtype=dt),name='lograte')
        else:
            self.lograte=tf.convert_to_tensor(-10, dtype=dt)
    

    def __call__(self, x):
        x=tf.convert_to_tensor(x, dtype=self.dt)
        components = [tfd.Poisson(log_rate=tf.zeros_like(x)+self.lograte)]+ \
                     [tfd.Poisson(log_rate=self.w[i]*x+self.c[i]) for i in range(self.nnz)]
        ydist=tfd.Mixture(
            cat=tfd.Categorical(logits=self.logits+tf.zeros((tf.shape(x)[0],1), dtype=self.dt)),
            components=components
        )
        return ydist

    def betas(self):
        return np.concatenate([np.array([0]),self.w.numpy()])


class CountryFeaturesType(Enum):
    ONLY_COOPERATION=1
    INDIVIDUAL=2
    ALL=3

def interactions(df,bootstrap=False, features_type:CountryFeaturesType=CountryFeaturesType.INDIVIDUAL):
    if bootstrap:
        df = df.sample(df.shape[0], replace=True)
    A=np.array(df)
    C=np.zeros((A.shape[1],A.shape[1]))

    for v in A:
        v = np.sign(v)
        M = v[...,np.newaxis]@v[np.newaxis,...]
        M-=np.diag(np.diag(M))
        C+= M

    if features_type == CountryFeaturesType.ALL:
        S = df.sum().to_numpy()[..., np.newaxis]
    elif features_type == CountryFeaturesType.ONLY_COOPERATION:
        S = A[(np.all(A!=1,axis=1)),:].sum(axis=0)[..., np.newaxis]
    elif features_type == CountryFeaturesType.INDIVIDUAL:
        S = A[np.any(A==1, axis=1),:].sum(axis=0)[..., np.newaxis]

    return C,S

class Estimator:
    dt = np.float64

    def __init__(self, data,model, mass:CountryFeaturesType,bootstrap=True):

        self.df = data
        self.flat_idx = np.triu_indices(self.df.shape[1],k = 1)
        self.interaction_to_xy(*interactions(self.df,bootstrap=bootstrap,features_type=mass))
        self.model = model

    def interaction_to_xy(self, C,S):
        flatc = C[self.flat_idx]
        sst = S@S.T
        sst_flat = sst[self.flat_idx]

        x=sst_flat.astype(self.dt)
        y=flatc.astype(self.dt)

        where_x, = np.where(x)
        self.x,self.y = np.log(x[where_x]),y[where_x]
        self.flat_idx = tuple(fi[where_x] for fi in self.flat_idx)
    
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

    def plot(self, dirname=None):
        sidx = np.argsort(self.x)
        ydist = self.model(self.x[sidx])

        Z = np.argmax(np.stack([c.prob(self.y[sidx]) for c in ydist.components],axis=1)*ydist.cat.probs_parameter(), axis=1)
        betas = self.model.betas()
        cmap = plt.get_cmap('viridis')
        plt.figure(figsize=(11,9))
        plt.scatter(self.x[sidx],self.y[sidx],
            alpha=0.8,
            c=betas[Z],
            cmap=cmap
            )

        stats = pd.DataFrame(
            dict(
                x=self.x[sidx],
                y=self.y[sidx],
                Z=Z,
                label=[str(set(self.df.columns[[self.flat_idx[0][si],self.flat_idx[1][si]]])) for si in sidx]
            )
        )
        # for i1,i2,x_,y_ in zip(*flat_idx,x,y):
        #     if y_ >0:
        #         plt.text(x_,np.log1p(y_),str(set(self.df.columns[[i1,i2]])), fontsize=4)

        #ax = plt.gca()
        #colormap = plt.get_cmap(cmap)
        #ax.set_prop_cycle(color=[colormap(k) for k in range(len(ydist.components))])

        for i,c in enumerate(ydist.components):
            plt.plot(self.x[sidx],c.mean(),label=f'$E C_{{ij}}|Z={i}$', color=cmap(betas[i]))
        plt.legend()
        #plt.ylim(-0.1,25)
        log1p_scale = mpl.scale.FuncScale(plt.gca(),(np.log1p,np.expm1))
        plt.yscale(log1p_scale)
        #plt.ylabel(r'$\log(\lambda_{ij}+1)$')
        plt.ylabel('number of interactions')
        plt.yticks([0,1,2,3,4,5,10,25,50,100,250,500,1000])

        plt.xlabel(r'$\log(c_{i}c_{j})$')
        plt.title("Model grawitacyjny relacji w funkcji patentów par krajów")


        if dirname:
            plt.savefig(os.path.join(dirname,'reg.pdf'))
            stats.to_csv(os.path.join(dirname,'summary.csv'))


        ydist = self.model(self.x)

        Z = np.argmax(np.stack([c.prob(self.y) for c in ydist.components],axis=1)*ydist.cat.probs_parameter(), axis=1)


        plt.figure(figsize=(9,11))
        In = np.zeros(2*[self.df.shape[1]])+np.nan
        betas = self.model.betas()

        In[self.flat_idx]=betas[Z]
        In[(self.flat_idx[1],self.flat_idx[0])]=betas[Z]

        sns.heatmap(In,cmap=cmap,linewidths=0.2, linecolor='black')
        ntick = self.df.shape[1]
        plt.xticks(ticks=np.arange(0,ntick)+0.5,labels=list(self.df.columns), rotation=90);
        plt.yticks(ticks=np.arange(0,ntick)+0.5,labels=list(self.df.columns), rotation=0);
        plt.gca().set_aspect('equal')
        if dirname:
            plt.savefig(os.path.join(dirname,'grid.pdf'))
        return stats

    def save(self, directory:str):
        checkpoint = tf.train.Checkpoint(model=self.model)
        checkpoint.write(os.path.join(directory,self.__class__.__name__))

    


def main(_):

    clean_df = data.load_clean(FLAGS.pickle)
    df = data.fractions_countries(clean_df, with_others=FLAGS.others)

    for i in range(FLAGS.nboot):
        try:
            dirname = os.path.join(FLAGS.out,str(i))
            os.makedirs(dirname)
        except:
            pass
        e = Estimator(
            data=df,
            model=PoissonGravitationalModel(
                nnz=FLAGS.nnz,
                trainable_lograte=FLAGS.treinablezero
            ),
            bootstrap=i>0,
            mass=CountryFeaturesType[FLAGS.feature_type]
        )
        e.fit()
        e.save(dirname)
        e.plot(dirname)
    return 0



if __name__ == '__main__':
    app.run(main)