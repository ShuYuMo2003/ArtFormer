from torch import nn

class MLPTokenizer(nn.Module):
    def __init__(self, d_token, d_hidden, d_model, drop_out):
        super().__init__()
        self.fc_0 = nn.Linear(d_token, d_hidden)
        self.acti = nn.Mish()
        self.drop = nn.Dropout(drop_out)
        self.fc_1 = nn.Linear(d_hidden, d_model)

    def forward(self, x):
        x = self.fc_0(x)
        x = self.acti(x)
        x = self.drop(x)
        x = self.fc_1(x)
        return x

class MLPUnTokenizer(nn.Module):
    def __init__(self, d_token, d_hidden, d_model, drop_out):
        super().__init__()
        self.fc_0 = nn.Linear(d_model, d_hidden)
        self.acti = nn.Mish()
        self.drop = nn.Dropout(drop_out)
        self.fc_1 = nn.Linear(d_hidden, d_token)

    def forward(self, x):
        x = self.fc_0(x)
        x = self.acti(x)
        x = self.drop(x)
        x = self.fc_1(x)
        return x