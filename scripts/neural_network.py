import torch
import torch.nn as nn


class PoseToControl(nn.Module):
    def __init__(self, control_dim):
        super(PoseToControl, self).__init__()

        self.in1_to_lstm = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32))

        self.in2_to_lstm = nn.Sequential(
            nn.Linear(360, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32))

        self.in_to_lstm = nn.Sequential(
            nn.Linear(364, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU())

        self.lstm_to_control = nn.LSTM(32*2, control_dim)

    def forward(self, odom_input, laser_scan, batch_size, seq_len, h0, c0):
        total_input = torch.cat((odom_input, laser_scan), dim=1)
        h = self.in_to_lstm(total_input)
        h = torch.reshape(h, (batch_size,seq_len,64))
        h = h.permute(1,0,2)



        # =======================================
        '''h1 = self.in1_to_lstm(odom_input)
        h1 = torch.reshape(h1, (batch_size,seq_len,32))
        h2 = self.in2_to_lstm(laser_scan)
        h2 = torch.reshape(h2, (batch_size,seq_len,32))
        h1, h2 = h1.permute(1,0,2), h2.permute(1,0,2)
        h = torch.cat((h1,h2), dim=2)'''
        control, (hn, cn) = self.lstm_to_control(h, (h0,c0))

        return control[-1,:,:]