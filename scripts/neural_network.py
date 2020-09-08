import torch
import torch.nn as nn


class PoseToControl(nn.Module):
    def __init__(self, control_dim):
        super(PoseToControl, self).__init__()

        self.in_to_control = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, control_dim),
            nn.ReLU())

    def forward(self, odom_input, batch_size):
        control = self.in_to_control(odom_input)
        return control