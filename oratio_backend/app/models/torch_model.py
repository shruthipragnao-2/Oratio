from __future__ import annotations

# Make torch optional; fall back to deterministic scorer if unavailable
try:  # pragma: no cover - optional
    import torch
    from torch import nn
    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - optional
    torch = None  # type: ignore
    nn = None  # type: ignore
    _TORCH_AVAILABLE = False


class _TinyBiasNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear1 = nn.Linear(32, 16)
        self.linear2 = nn.Linear(16, 1)
        self.act = nn.ReLU()
        self.out = nn.Sigmoid()

    def forward(self, x):  # type: ignore[override]
        x = self.act(self.linear1(x))
        x = self.out(self.linear2(x))
        return x


class TorchBiasScorer:
    def __init__(self, use_gpu: bool = False) -> None:
        if _TORCH_AVAILABLE:
            self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
            self.model = _TinyBiasNet().to(self.device)
            self.model.eval()
        else:
            self.device = None  # type: ignore[assignment]
            self.model = None  # type: ignore[assignment]

    def score(self, text: str) -> float:
        # Deterministic pseudo-score whether or not torch is installed
        if _TORCH_AVAILABLE and self.model is not None:
            seed = abs(hash(text)) % (2**32)
            generator = torch.Generator(device=self.device).manual_seed(seed)
            x = torch.rand((1, 32), generator=generator, device=self.device)
            with torch.no_grad():
                y = self.model(x)
            return float(y.item())
        else:
            return float((abs(hash(text)) % 1000) / 1000.0)


