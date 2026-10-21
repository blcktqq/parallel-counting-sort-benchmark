"""Render OpenMP scaling from recorded CSV, without rerunning measurements."""
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def make_plots(results):
    results = Path(results)
    df = pd.read_csv(results / 'summary.csv')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for k, color in [(256, '#24445C'), (65536, '#168A8E')]:
        part = df[(df.n == df.n.max()) & (df.k == k) & (df['mode'] == 'parallel')].sort_values('p')
        axes[0].plot(part.p, part.median_ms, 'o-', color=color, label=f'K={k:,}')
        axes[1].plot(part.p, part.speedup, 'o-', color=color, label=f'K={k:,}')
    axes[0].set(xlabel='OpenMP threads', ylabel='Median time (ms)', title=f'n={df.n.max():,}')
    axes[1].set(xlabel='OpenMP threads', ylabel='Speedup vs Cython sequential', title='OpenMP scaling')
    axes[1].axhline(1, color='black', linestyle=':', linewidth=1)
    for ax in axes:
        ax.set_xticks([1, 2, 4])
        ax.grid(alpha=.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(results / 'openmp_scaling.png', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('results', nargs='?', default=str(Path(__file__).parent / 'results'))
    make_plots(parser.parse_args().results)
