"""Render comparison figures from measured CSV files; no timings are rerun."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def make_plots(results):
    results=Path(results)
    df=pd.read_csv(results/'summary.csv')
    languages=['Python','C++','Rust','Go','Node.js']
    colors={'Python':'#24445C','C++':'#D97924','Rust':'#9B4334','Go':'#168A8E','Node.js':'#7250A2'}
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for language in languages:
        for ax,mode in zip(axes,['seq','warm']):
            part=df[(df.language==language)&(df.k==256)&(df['mode']==mode)]
            if mode=='warm':part=part[part.p==4]
            part=part.sort_values('n')
            ax.plot(part.n,part.median_ms,'o-',label=language,color=colors[language])
    for ax,title in zip(axes,['Sequential counting sort','Warm pool, p=4']):
        ax.set(xscale='log',yscale='log',xlabel='n',ylabel='Median time (ms)',title=title)
        ax.grid(alpha=.25);ax.legend()
    fig.tight_layout();fig.savefig(results/'absolute_times.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    largest=df.n.max()
    for language in languages:
        part=df[(df.language==language)&(df.k==256)&(df.n==largest)&(df['mode']=='warm')].sort_values('p')
        axes[0].plot(part.p,part.speedup,'o-',label=language,color=colors[language])
        for mode,style in [('warm','-'),('cold','--')]:
            part=df[(df.language==language)&(df.k==256)&(df.p==4)&(df['mode']==mode)].sort_values('n')
            axes[1].plot(part.n,part.median_ms,style,label=f'{language} {mode}',color=colors[language])
    axes[0].set(xlabel='Worker tasks p',ylabel='Speedup relative to own sequential',xticks=[1,2,4],title=f'n={largest:,}, K=256')
    axes[0].axhline(1,color='black',linestyle=':',linewidth=1)
    axes[1].set(xscale='log',yscale='log',xlabel='n',ylabel='Median time (ms)',title='Warm vs worker creation, p=4')
    for ax in axes:ax.grid(alpha=.25);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(results/'scaling_startup.png',dpi=180);plt.close(fig)
    return df

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('results',nargs='?',default=str(Path(__file__).resolve().parent/'results'))
    make_plots(parser.parse_args().results)
