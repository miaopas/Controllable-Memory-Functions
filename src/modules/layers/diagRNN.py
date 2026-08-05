import torch
from torch import nn


class diagRNN(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
    """

    def __init__(self, input_dim=1, output_dim=1, hidden_dim=16):
        super().__init__()

        self.hidden_dim = hidden_dim

        W_real = nn.Parameter(torch.rand(hidden_dim // 2))
        W_real = torch.repeat_interleave(W_real, repeats=2)

        _W_imag = nn.Parameter(torch.randn(hidden_dim // 2))

        W_imag = torch.zeros(hidden_dim - 1)
        W_imag[::2] = _W_imag

        self.register_buffer("W_real", W_real)
        self.register_buffer("W_imag", W_imag)

        self.U = nn.Parameter(torch.randn(hidden_dim, input_dim))
        self.c = nn.Parameter(torch.randn(output_dim, hidden_dim))

    def forward(self, x):
        batch_size = x.size(0)
        seq_len = x.size(1)

        h = torch.zeros(batch_size, self.hidden_dim, device=x.device)  # Initialize h_{-1}

        self.W = (
            torch.diag(self.W_real)
            + torch.diag(-self.W_imag, diagonal=1)
            + torch.diag(self.W_imag, diagonal=-1)
        )

        # self.W.data = torch.sigmoid(self.W.data)

        # self.W.data = 1/(1+torch.exp(-self.W.data))

        outputs = []
        for t in range(seq_len):
            h = torch.matmul(h, torch.diag(self.W)) + torch.matmul(x[:, t, :], self.U.t())
            y = torch.matmul(self.c, h.t()).unsqueeze(0)
            outputs.append(y)

        return torch.cat(outputs, dim=0).permute(2, 0, 1)


class diagRNN2(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
    """

    def __init__(self, input_dim=1, output_dim=1, hidden_dim=16):
        super().__init__()

        self.hidden_dim = hidden_dim

        self.W1 = nn.Parameter(torch.rand(hidden_dim))  # Diagonal matrix
        self.W2 = nn.Parameter(torch.rand(hidden_dim))  # Diagonal matrix
        self.U1 = nn.Parameter(torch.randn(hidden_dim, input_dim))
        self.U2 = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.c = nn.Parameter(torch.randn(output_dim, hidden_dim))

    def forward(self, x):
        batch_size = x.size(0)
        seq_len = x.size(1)

        # self.W1.data = torch.clamp(self.W1.data, 0.01,0.99)
        # self.W2.data = torch.clamp(self.W2.data, 0.01,0.99)

        self.W1.data = torch.sigmoid(self.W1.data)
        self.W2.data = torch.sigmoid(self.W2.data)

        h = torch.zeros(batch_size, self.hidden_dim, device=x.device)  # Initialize h_{-1}
        g = torch.zeros(batch_size, self.hidden_dim, device=x.device)  # Initialize h_{-1}

        outputs = []
        for t in range(seq_len):
            g = torch.matmul(g, torch.diag(self.W1)) + torch.matmul(x[:, t, :], self.U1.t())
            h = torch.matmul(h, torch.diag(self.W2)) + torch.matmul(g, self.U2.t())

            y = torch.matmul(self.c, h.t()).unsqueeze(0)
            outputs.append(y)

        return torch.cat(outputs, dim=0).permute(2, 0, 1)
