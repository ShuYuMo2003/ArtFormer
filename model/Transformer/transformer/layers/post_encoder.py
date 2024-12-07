from torch import nn


class PostEncoder(nn.Module):
    def __init__(self, dim: int, d_model: int, deepth: int, dropout: float):
        super().__init__()
        self.fc0 = nn.Linear(dim, dim * 2)
        self.block = nn.ModuleList([
            ResnetBlockFC(dim * 2, dropout) for _ in range(deepth)
        ])
        self.fc1 = nn.Linear(dim * 2, d_model)

    def forward(self, x):
        x = self.fc0(x)
        for block in self.block:
            x = block(x)
        x = self.fc1(x)
        return x


# Resnet Blocks
class ResnetBlockFC(nn.Module):
    ''' Fully connected ResNet Block class.

    Args:
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
    '''

    def __init__(self, size_in, dropout, size_out=None, size_h=None):
        super().__init__()
        # Attributes
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)
        self.actvn = nn.LeakyReLU(0.2)

        self.droupout = nn.Dropout(dropout)

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)
        # Initialization
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x):
        net = self.fc_0(self.actvn(x))
        net = self.droupout(net)
        dx = self.fc_1(self.actvn(net))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx
