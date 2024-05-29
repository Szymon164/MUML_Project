import torch
import torch.nn as nn
import torch.nn.functional as F
    
class Decoder(nn.Module):
    def __init__(self) -> None:
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(10, 2000)
        self.fc2 = nn.Linear(2000, 500)
        self.fc3 = nn.Linear(500, 500)
        self.fc4 = nn.Linear(500, 784)

    def forward(self, x) -> tuple[torch.Tensor]:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return F.sigmoid(self.fc4(x))