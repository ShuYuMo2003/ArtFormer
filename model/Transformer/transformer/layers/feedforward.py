from torch import nn

# Feed Forward Neural Network
class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, hidden_dim, dropout):
        super().__init__()
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, d_model)
        self.dropout = nn.Dropout(dropout)
        self.acti = nn.Mish()

    def forward(self, x):
        x = self.fc1(x)
        x = self.acti(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
