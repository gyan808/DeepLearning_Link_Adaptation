import torch
import torch.nn as nn

class LSTMModel(nn.Module):

    def __init__(self):

        super(LSTMModel, self).__init__()

        # LSTM Layer
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=50,
            num_layers=1,
            batch_first=True
        )

        # Fully Connected Layer
        self.fc = nn.Linear(50, 1)

    def forward(self, x):

        # LSTM output
        out, _ = self.lstm(x)

        # Take last time step
        out = out[:, -1, :]

        # Final prediction
        out = self.fc(out)

        return out