import torch
import torch.nn as nn


class PoseToControl(nn.Module):
    def __init__(self, control_dim):
        super(PoseToControl, self).__init__()

        self.in_to_lstm = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU())

        self.lstm_to_control = nn.LSTM(32, control_dim)

    def forward(self, odom_input, laser_scan, batch_size, seq_len, h0, c0):
        total_input = torch.cat((odom_input, laser_scan), dim=1)
        h = self.in_to_lstm(total_input)
        h = torch.reshape(h, (batch_size,seq_len,32))
        h = h.permute(1,0,2)
        control, (hn, cn) = self.lstm_to_control(h, (h0,c0))

        return control[-1,:,:]