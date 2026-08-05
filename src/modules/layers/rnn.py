import torch
from torch import nn


class gatedRNNCell(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
        (1-g(x_t)) h_t + g(x_t) x_t
    """

    def __init__(self, hid_dim, gate_dim):
        super().__init__()

        self.hidden_ff = nn.Linear(hid_dim, hid_dim)
        # torch.nn.init.orthogonal_(self.hidden_ff.weight.data, gain=0.9)
        # torch.nn.init.xavier_uniform_(self.linear.weight.data)
        # torch.nn.init.xavier_uniform_(self.hidden_ff.weight.data, gain=0.9)

        # register h0
        self.register_buffer("h0", torch.zeros(1, 1, hid_dim))

        # self.gate = nn.Sequential(nn.Linear(hid_dim, gate_dim), nn.Sigmoid(), nn.Linear(gate_dim, 1), nn.Sigmoid())
        self.gate1 = nn.Sequential(nn.Linear(hid_dim, 1), nn.Sigmoid())
        self.gate2 = nn.Sequential(nn.Linear(hid_dim, 1), nn.Sigmoid())

        # self.gate1 = nn.Sequential(nn.Linear(hid_dim, gate_dim), nn.ReLU(), nn.Linear(gate_dim, 1), nn.Sigmoid())
        # self.gate2 = nn.Sequential(nn.Linear(hid_dim, gate_dim), nn.ReLU(), nn.Linear(gate_dim, 1), nn.Sigmoid())

        torch.nn.init.constant_(self.gate1[0].weight.data, 0)
        torch.nn.init.constant_(self.gate1[0].bias.data, 6.0)

        torch.nn.init.constant_(self.gate2[0].weight.data, 0)
        torch.nn.init.constant_(self.gate2[0].bias.data, 6.0)

    def forward(self, x, return_g=False):
        """_summary_

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        # src = [batch size, input len, input dim]
        length = x.shape[1]
        if return_g:
            g1_list = []
            g2_list = []

        hidden = []
        hidden.append(self.h0)

        for i in range(length):
            g1 = self.gate1(hidden[i])
            g2 = self.gate2(hidden[i])

            if return_g:
                g1_list.append(g1)
                g2_list.append(g2)

            h_next = g1 * x[:, i : i + 1, :] + g2 * self.hidden_ff(hidden[i])
            hidden.append(h_next)

        hidden = torch.cat(hidden[1:], dim=1)

        return hidden


class LinearRNNCell(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
    """

    def __init__(
        self,
        hid_dim,
        A_gain,
    ):
        super().__init__()

        self.linear = nn.Linear(hid_dim, hid_dim, bias=False)
        self.hidden_ff = nn.Linear(hid_dim, hid_dim, bias=False)
        torch.nn.init.orthogonal_(self.hidden_ff.weight.data, gain=A_gain)
        torch.nn.init.xavier_uniform_(self.linear.weight.data)
        # torch.nn.init.xavier_uniform_(self.hidden_ff.weight.data, gain=0.9)

        # register h0
        self.register_buffer("h0", torch.zeros(1, 1, hid_dim))

    def forward(self, x):
        """_summary_

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        # src = [batch size, input len, input dim]
        length = x.shape[1]

        hidden = []
        hidden.append(self.h0)
        x = self.linear(x)

        for i in range(length):
            h_next = x[:, i : i + 1, :] + self.hidden_ff(hidden[i])
            hidden.append(h_next)

        hidden = torch.cat(hidden[1:], dim=1)
        return hidden


class LinearRNN(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
    """

    def __init__(self, input_dim, output_dim, hid_dim, n_layer=1, A_gain=0.9):
        super().__init__()

        self.input_ff = nn.Linear(input_dim, hid_dim, bias=False)
        self.rnn_layers = nn.ModuleList([LinearRNNCell(hid_dim=hid_dim, A_gain=A_gain) for _ in range(n_layer)])
        self.output_ff = nn.Linear(hid_dim, output_dim, bias=False)

        torch.nn.init.xavier_uniform_(self.input_ff.weight.data)
        torch.nn.init.xavier_uniform_(self.output_ff.weight.data)

    def forward(self, x):
        """_summary_

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        # src = [batch size, input len, input dim]

        hidden = self.input_ff(x)

        for rnn in self.rnn_layers:
            hidden = rnn(hidden)

        out = self.output_ff(hidden)
        return out


class gatedRNN(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
    """

    def __init__(self, input_dim, output_dim, hid_dim, gate_dim, n_layer=1):
        super().__init__()

        self.input_ff = nn.Linear(input_dim, hid_dim, bias=False)
        self.rnn_layers = nn.ModuleList(
            [gatedRNNCell(hid_dim=hid_dim, gate_dim=gate_dim) for _ in range(n_layer)]
        )
        self.output_ff = nn.Linear(hid_dim, output_dim, bias=False)

        torch.nn.init.xavier_uniform_(self.input_ff.weight.data)
        torch.nn.init.xavier_uniform_(self.output_ff.weight.data)

    def forward(self, x, **kwarg):
        """_summary_

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        # src = [batch size, input len, input dim]

        hidden = self.input_ff(x)

        for rnn in self.rnn_layers:
            hidden = rnn(hidden, **kwarg)

        out = self.output_ff(hidden)
        return out
