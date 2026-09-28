import torch
from torch import nn
from torch.nn import functional as F

class VisualRNN(nn.Module):
    def __init__(self, hidden=64, alpha=.5, kind='rnn', normalize=False, matched_init=False):
        super().__init__()
        self.kind, self.hidden, self.alpha = kind, hidden, alpha
        self.freeze_encoder = False
        self.visual = nn.Sequential(nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(),
                                    nn.Conv2d(8, 8, 3, padding=1), nn.ReLU())
        self.project = nn.Linear(8, 16)
        self.encode_norm = nn.BatchNorm1d(16) if normalize else nn.Identity()
        self.input = nn.Linear(16, hidden)
        self.recurrent = nn.Parameter(torch.randn(hidden, hidden) / hidden**.5 * .9)
        self.readout = nn.Linear(hidden, 2)
        signs = torch.ones(hidden)
        signs[int(.8 * hidden):] = -1
        self.register_buffer('signs', signs)
        if kind == 'ei' or matched_init:
            with torch.no_grad():
                w = self.recurrent.abs()
                # Balance row sums at initialization without changing Dale columns.
                w[:, signs < 0] *= (signs > 0).sum() / (signs < 0).sum()
                w *= .9 / torch.linalg.eigvals(w * signs).abs().max()
                self.recurrent.copy_(w if kind == 'ei' else w * signs)
        if kind == 'reservoir':
            for name, p in self.named_parameters():
                if not name.startswith('readout.'):
                    p.requires_grad_(False)

    def weights(self):
        return self.recurrent.abs() * self.signs if self.kind == 'ei' else self.recurrent

    def encode(self, image):
        if self.kind == 'reservoir' or self.freeze_encoder:
            self.encode_norm.eval()
        return self.encode_norm(F.relu(self.project(self.visual(image).sum((2,3)) / 32)))

    def forward(self, sample, test, delay=5, reset=False, return_states=False,
                sample_steps=4, test_steps=4, encoded=False):
        batch = len(sample)
        es, et = (sample, test) if encoded else (self.encode(sample), self.encode(test))
        h = sample.new_zeros((batch, self.hidden))
        w = self.weights()
        states = []
        zero = torch.zeros_like(es)
        for t in range(sample_steps+delay+test_steps):
            if reset and t == sample_steps+delay:
                h = torch.zeros_like(h)
            inp = es if t < sample_steps else (zero if t < sample_steps+delay else et)
            h = (1-self.alpha)*h + self.alpha*F.relu(self.input(inp)+F.linear(h,w))
            states.append(h)
        logits = self.readout(h)
        if return_states:
            return logits, torch.stack(states, 1), es
        return logits
