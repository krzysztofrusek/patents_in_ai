import matplotlib as mpl
import seaborn as sns

def plot_config():
    try:
        mpl.use('MacOSX')
    except:
        mpl.use('Agg')
    sns.set_theme(
        context='paper',
        style='whitegrid',
        rc={
            'figure.figsize': (4.7, 2.9),
            'font.size': 10,
            'font.family': 'serif',
            'xtick.labelsize':7,
            'ytick.labelsize': 7
            },
        font_scale=1.0
    )
