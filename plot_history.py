import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_history(history_path, output_path):
    history_path = Path(history_path)
    output_path = Path(output_path)

    with history_path.open(newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f'No training history found in {history_path}.')

    steps = [int(row['iter_num']) for row in rows]
    train_loss = [float(row['train_loss']) for row in rows]
    val_loss = [float(row['val_loss']) for row in rows]
    best_index = min(range(len(val_loss)), key=val_loss.__getitem__)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(steps, train_loss, marker='o', markersize=4, label='Training loss')
    ax.plot(steps, val_loss, marker='o', markersize=4, label='Validation loss')
    ax.scatter(
        steps[best_index], val_loss[best_index], color='tab:red', zorder=3,
        label=f'Best validation: {val_loss[best_index]:.3f}'
    )
    ax.set(title='Training and Validation Loss', xlabel='Training step', ylabel='Loss')
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return steps[best_index], val_loss[best_index]


if __name__ == '__main__':
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description='Plot saved training history.')
    parser.add_argument(
        '--history', default=project_dir / 'data/tinyshakespeare/history.csv',
        type=Path
    )
    parser.add_argument(
        '--output', default=project_dir / 'data/tinyshakespeare/loss.png',
        type=Path
    )
    args = parser.parse_args()
    best_step, best_loss = plot_history(args.history, args.output)
    print(f'Saved {args.output} (best validation loss {best_loss:.4f} at step {best_step})')
