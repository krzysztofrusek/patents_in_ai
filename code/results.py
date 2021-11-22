import os.path
import pickle

import data
import gravity
import bayes

import networkx as nx
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy as np
from absl import flags, app, logging

from util import plot_config

FLAGS= flags.FLAGS
flags.DEFINE_string("mcmcpickle", "../gen/mcmc3/samples.pkl", "Input file")
flags.DEFINE_string("paperdir", "../gen/paper", "out dir")
flags.DEFINE_string('trend_pickle','../plg/trends/16917')


class BayesResults:
    def __init__(self,path:str):
        with open(path, 'br') as f:
            samples = pickle.load(f)
        all_long = np.reshape(np.concatenate(samples, axis=-1), (-1, 8))
        self.all_long_df = pd.DataFrame(all_long,
                                   columns=['$\alpha_1$', '$\alpha_2$', '$\beta_1$', '$\beta_2$', '$\beta_0$', '$l_0$', '$l_1$', '$l_2$'])
        self.samples = samples


    def plot_params(self, save_to=None):

        sns.displot(pd.melt(self.all_long_df),
                    x='value',
                    col='variable',
                    col_wrap=4,
                    height = 1.4,
                    aspect = 1.6,
                    common_bins=False,
                    kde=True,
                    facet_kws=dict(sharex=False, sharey=True)
                    )
        plt.tight_layout()
        if save_to:
            plt.savefig(os.path.join(save_to,'params.pdf'))
        else:
            plt.show()
        return

    def plot_regression(self,save_to=None):
        clean_df = data.load_clean(FLAGS.pickle)
        clean_df = clean_df[clean_df.publication_date.dt.year <= FLAGS.toyear]
        df = data.fractions_countries(clean_df, with_others=True)
        dataset = bayes.Dataset.from_pandas(df, gravity.CountryFeaturesType.ALL)
        _x = dataset.x[..., np.newaxis]

        def reshaper(x):
            s = x.shape
            return np.reshape(x,(-1,)+s[-2:])

        model = bayes.PoissonMixtureRegression(*map(reshaper,self.samples))

        n_batch = model.w.shape[0]

        x = np.broadcast_to(_x,[n_batch]+list(_x.shape))
        y = np.broadcast_to(dataset.y[..., np.newaxis], [n_batch] + list(dataset.y.shape)+[1])
        Z = model.Z(x,y)
        z = np.argmax(np.mean(Z.probs_parameter(), axis=0), axis=-1)

        ydist = model(x)
        _lambda = ydist.distribution.components_distribution.mean()
        E_lambda = np.mean(_lambda, axis=0)
        El_df = pd.DataFrame(np.concatenate([dataset.x[..., np.newaxis], E_lambda], axis=1),
                             columns=['x', '$E Y|x,Z=0$', '$E Y|x,Z=1$', '$E Y|x,Z=2$'])
        Q_lambda = np.quantile(_lambda,q=[0.025,0.975], axis=0)

        xyz_df = pd.DataFrame(dict(x=dataset.x, y=dataset.y,z=z,
                                   label=[str(set(df.columns[[dataset.flat_idx[0][si],dataset.flat_idx[1][si]]])) for si in range(len(dataset.x))]
                                   ))


        palette=sns.color_palette("viridis",3)

        sns.scatterplot(data=xyz_df, x='x',y='y', hue='z', style='z', palette=palette[:3])
        sort_idx = np.argsort(dataset.x)


        for i in range(3):
            plt.plot(dataset.x[sort_idx], E_lambda[sort_idx,i], label=f'$E C_{{ij}}|Z={i}$', alpha=0.8,color=palette[i])
            plt.fill_between(x=dataset.x[sort_idx],
                             y1=Q_lambda[0,sort_idx,i],
                             y2=Q_lambda[1,sort_idx,i],
                             alpha=0.2,
                             color=palette[i])
        plt.legend()

        plt.yscale('function', functions=(np.log1p, np.expm1))
        plt.ylabel('number of interactions')
        plt.yticks([0,1,2,3,4,5,10,25,50,100,250,500,1000])
        plt.yticks([0, 1, 2, 5, 10, 25, 50, 100, 250, 500, 1000])

        plt.xlabel(r'$\log(c_{i}c_{j})$')
        plt.tight_layout()
        if save_to:
            plt.savefig(os.path.join(save_to,'reg.pdf'))
            xyz_df.to_csv(os.path.join(save_to,'summary.csv'))
        else:
            plt.show()

        plt.close()
        In = np.zeros(2*[df.shape[1]])+np.nan
        In[dataset.flat_idx]=z
        In[(dataset.flat_idx[1],dataset.flat_idx[0])]=z

        annot = np.zeros_like(In)
        annot[dataset.flat_idx]=dataset.x
        annot[(dataset.flat_idx[1],dataset.flat_idx[0])]=dataset.x

        ax = sns.heatmap(In, cmap=palette, linewidths=0.03, linecolor='black',annot=annot,fmt='.1f',annot_kws=dict(fontsize=3.5))
        colorbar = ax.collections[0].colorbar
        colorbar.set_ticks([1/3,1,5/3])
        colorbar.set_ticklabels(['0', '1', '2'])

        ntick = df.shape[1]
        plt.xticks(ticks=np.arange(0,ntick)+0.5,labels=list(df.columns), rotation=90, fontsize=5);
        plt.yticks(ticks=np.arange(0,ntick)+0.5,labels=list(df.columns), rotation=0, fontsize=5);
        #plt.gca().set_aspect('equal')
        plt.tight_layout()
        if save_to:
            plt.savefig(os.path.join(save_to,'grid.pdf'))
            plt.savefig(os.path.join(save_to, 'grid.svg'))
        else:
            plt.show()

        return

class PatentCooperationGraph:
    def __init__(self):
        self.clean_df = data.load_clean(FLAGS.pickle)
        self.fractions = data.fractions_countries(self.clean_df, with_others=True)
    def plot(self,save_to=None):
        plt.figure(figsize=(4.7,4.7))
        C,S = gravity.interactions(self.fractions,bootstrap=False,features_type=gravity.CountryFeaturesType.ALL)
        G=nx.Graph(C)
        pos = nx.circular_layout(G,center=(1,1))
        #pos = nx.spectral_layout(G, scale=100,weight=None)

        nodes_df = pd.DataFrame(pos.values(),columns=['x','y'],index=self.fractions.columns)
        nodes_df['s']=np.where(S<0.001,np.nan,S)

        palette = 'YlOrRd'
        palette='viridis'

        ax=sns.scatterplot(data=nodes_df,
                           x='x',
                           y='y',
                           size='s',
                           size_norm=mpl.colors.LogNorm(),
                           sizes=(100,400),
                           hue='s',
                           hue_norm = mpl.colors.LogNorm(),
                           palette=palette,
                           zorder=1
                           )


        def edge_gen():
            for n1, n2, c in G.edges.data('weight'):
                yield dict(x=pos[n1][0],y=pos[n1][1],c=c,edge=(self.fractions.columns[n1],self.fractions.columns[n2]))
                yield dict(x=pos[n2][0], y=pos[n2][1], c=c,
                           edge=(self.fractions.columns[n1], self.fractions.columns[n2]))

        edge_df = pd.DataFrame(list(edge_gen()))

        ax=sns.lineplot(data=edge_df,
                        x='x',
                        y='y',
                        size='c',
                        units='edge',
                        size_norm=mpl.colors.LogNorm(),
                        sizes=(1,3),
                        hue='c',
                        hue_norm=mpl.colors.LogNorm(),
                        estimator=None,
                        palette=palette,
                        alpha=0.8,
                        zorder=0)
        ax.set_aspect('equal')
        #sns.despine(ax=ax,left=True, right=True, top=True, bottom=True)
        # nx.draw_networkx_labels(G,pos, dict(enumerate(self.fractions.columns)),font_size=6,
        #                         font_color='gray'
        #                         )

        for n in G.nodes():
            plt.text(pos[n][0],pos[n][1],s=self.fractions.columns[n],
                     horizontalalignment='center',
                     verticalalignment='center',
                     fontdict={
                'size': 6,
                'color':('black' if S[n]>100 else "white")
            })
        plt.axis(False)

        plt.xlim(-0.1,2.3)
        plt.ylim(-0.1,2.1)
        plt.legend(loc='upper right',
                   title="Patents",
                   prop={'size': 6})

        plt.tight_layout()

        if save_to:
            plt.savefig(os.path.join(save_to,'graph.pdf'))
            Ccopy = C.copy()
            np.fill_diagonal(Ccopy, S)
            Cdf = pd.DataFrame(Ccopy,columns=self.fractions.columns,index=self.fractions.columns)
            Cdf.to_csv(os.path.join(save_to, 'graph.csv'))
            plt.close()
        else:
            plt.show()

        return




def main(_):
    plot_config()

    with sns.plotting_context(rc={
            'figure.figsize': (4.7, 4.7),
            },font_scale=0.8):
        PatentCooperationGraph().plot(FLAGS.paperdir)
    #return 0
    bayes_results = BayesResults(FLAGS.mcmcpickle)
    bayes_results.plot_regression(FLAGS.paperdir)
    bayes_results.plot_params(FLAGS.paperdir)
    return 0

if __name__ == '__main__':
    app.run(main)