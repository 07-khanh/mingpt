import os
import pickle
import argparse

import torch
from model import GPT, GPTConfig
from devices import select_device

data_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'data', 'tinyshakespeare'
)
meta_path = os.path.join(data_dir, 'meta.pkl')
checkpoint_path = os.path.join(data_dir, 'model.pt')


@torch.no_grad()
def generate_text(prompt, checkpoint_path, meta_path, *, max_new_tok=500,
                  temperature=1.0, top_k=40, device=None):
    if not prompt:
        raise ValueError('The prompt must contain at least one character.')
    if device is None:
        device = select_device()

    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f'Missing {meta_path}. Run data/tinyshakespeare/prepare.py first.')
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    if isinstance(meta, dict):
        vocab_size, itos, stoi = meta['vocab_size'], meta['itos'], meta['stoi']
    else:
        vocab_size, itos, stoi = meta

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    config = GPTConfig(**checkpoint['model_args'])
    if config.vocab_size != vocab_size:
        raise ValueError('The checkpoint and tokenizer vocabulary sizes do not match.')
    model = GPT(config)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()

    try:
        tokens = [stoi[c] for c in prompt]
    except KeyError as exc:
        raise ValueError(f'Character {exc.args[0]!r} is not in the vocabulary.') from exc
    idx = torch.tensor([tokens], dtype=torch.long, device=device)
    output = model.generate(
        idx, max_new_tok=max_new_tok, temperature=temperature, top_k=top_k
    )
    return ''.join(itos[i] for i in output[0].cpu().tolist())


def main(argv=None):
    parser = argparse.ArgumentParser(description='Generate text using minGPT.')
    parser.add_argument('--prompt', default='ROMEO:',
                        help='Prompt text to start generation.')
    parser.add_argument('--max_new_tok', type=int, default=500,
                        help='Maximum number of new tokens to generate.')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Temperature for sampling.')
    parser.add_argument('--top_k', type=int, default=40,
                        help='Top-k sampling for generation.')
    args = parser.parse_args(argv)
    text = generate_text(
        args.prompt, checkpoint_path, meta_path,
        max_new_tok=args.max_new_tok,
        temperature=args.temperature,
        top_k=args.top_k
    )
    print(text)
    return text


if __name__ == '__main__':
    main()
