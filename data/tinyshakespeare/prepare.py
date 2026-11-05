import pickle
from pathlib import Path
import urllib.request

import numpy as np

url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
data_dir = Path(__file__).resolve().parent
input_path = Path(__file__).resolve().parents[2] / 'input.txt'


def main(input_path=input_path, data_dir=data_dir):
    input_path = Path(input_path)
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.is_file():
        print('Downloading input.txt...')
        input_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, input_path)

    text = input_path.read_text(encoding='utf-8')
    chars = sorted(set(text))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    split = int(len(text) * 0.9)
    train_ids = np.array([stoi[c] for c in text[:split]], dtype=np.uint16)
    val_ids = np.array([stoi[c] for c in text[split:]], dtype=np.uint16)
    train_ids.tofile(data_dir / 'train.bin')
    val_ids.tofile(data_dir / 'val.bin')

    with open(data_dir / 'meta.pkl', 'wb') as f:
        pickle.dump((vocab_size, itos, stoi), f)

    print(f'Vocab size is {vocab_size:,}')
    print(f'Train has {len(train_ids):,} tokens')
    print(f'Val has {len(val_ids):,} tokens')


if __name__ == '__main__':
    main()
