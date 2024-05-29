import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self) -> None:
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(784, 500)
        self.fc2 = nn.Linear(500, 500)
        self.fc3 = nn.Linear(500, 2000)
        self.fc_mean = nn.Linear(2000, 10)
        self.fc_log_var = nn.Linear(2000, 10)

    def forward(self, x) -> tuple[torch.Tensor]:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc_mean(x), self.fc_log_var(x)