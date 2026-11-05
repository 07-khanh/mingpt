import torch
import torch.nn.functional as F
import pytest

from model import GPT, GPTConfig


@pytest.fixture
def model():
    torch.manual_seed(42)
    config = GPTConfig(
        block_size=16,
        vocab_size=20,
        n_layer=2,
        n_head=2,
        n_embd=32,
        dropout=0.0,
        bias=False,
    )
    return GPT(config).eval()


def test_forward_shapes(model):
    x = torch.randint(0, 20, (2, 8)) 
    y = torch.randint(0, 20, (2, 8)) 
    logits, loss = model(x, y)
    assert logits.shape == (2, 8, 20)
    assert loss.ndim == 0
    assert torch.isfinite(loss)

    inference_logits, inference_loss = model(x)
    assert inference_logits.shape == (2, 20)
    assert inference_loss is None
    torch.testing.assert_close(inference_logits, logits[:, -1, :])


def test_loss_matches_cross_entropy(model):
    x = torch.randint(0, 20, (2, 8))
    y = torch.randint(0, 20, (2, 8))
    logits, loss = model(x, y)
    expected = F.cross_entropy(logits.reshape(-1, 20), y.reshape(-1))
    torch.testing.assert_close(loss, expected)


def test_backward_pass(model):
    model.train()
    x = torch.randint(0, 20, (2, 8))
    y = torch.randint(0, 20, (2, 8))
    _, loss = model(x, y)
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.requires_grad]
    assert gradients and all(g is not None for g in gradients)
    assert all(torch.isfinite(g).all().item() for g in gradients)
    assert any(torch.count_nonzero(g).item() > 0 for g in gradients)


def test_causal_masking(model):
    a = torch.tensor([[1, 2, 3, 4, 5]])
    b = torch.tensor([[1, 2, 3, 9, 8]])
    target = torch.zeros_like(a)
    with torch.no_grad():
        logits_a, _ = model(a, target)
        logits_b, _ = model(b, target)
    torch.testing.assert_close(logits_a[:, :3, :], logits_b[:, :3, :])


def test_generation(model):
    prompt = torch.tensor([[1, 2, 3]])
    with torch.no_grad():
        output = model.generate(prompt, max_new_tok=5, temperature=0.8, top_k=5)
    assert output.shape == (1, 8)
    assert torch.equal(output[:, :3], prompt)
    assert ((output >= 0) & (output < 20)).all()


def test_checkpoint_round_trip(model, tmp_path):
    x = torch.tensor([[1, 2, 3]])
    path = tmp_path / 'model.pt'
    torch.save(model.state_dict(), path)
    restored = GPT(model.config).eval()
    restored.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
    with torch.no_grad():
        original_logits, _ = model(x)
        restored_logits, _ = restored(x)
    torch.testing.assert_close(original_logits, restored_logits)
