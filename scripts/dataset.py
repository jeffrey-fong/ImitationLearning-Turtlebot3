from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from builtins import range
import numpy as np
import torch
from torch.utils.data import Dataset
import copy
import matplotlib.pyplot as plt
import os
import json



class ImitationDataset(Dataset):
    def __init__(self, device='cpu', is_val=False, is_test=False):
        super(ImitationDataset, self).__init__()

        # Initializing attributes
        self.poses = None
        self.scans = None
        self.velocities = None
        self.device = device

        file_path = "/home/jeffrey/catkin_ws/src/cs6244/data/"

        # Iterate through all data files
        i = 0
        while True:
            # check if any data files exists
            if not os.path.exists(file_path+'trajectory_'+str(i)+'.json'):
                break

            # open each data file
            with open(file_path+'trajectory_'+str(i)+'.json', 'r') as file:
                traj_dict = json.load(file)

            # Convert the relevant parts to corresponding numpy arrays
            traj_pos = np.array(list(traj_dict.items())[0][-1])
            traj_scan = np.array(list(traj_dict.items())[1][-1])
            traj_vel = np.array(list(traj_dict.items())[2][-1])

            # Concatenate new data
            if i == 0:
                self.poses, self.scans = traj_pos, traj_scan
                self.velocities = traj_vel
            else:
                self.poses = np.concatenate((self.poses, traj_pos), axis=0)
                self.scans = np.concatenate((self.scans, traj_scan), axis=0)
                self.velocities = np.concatenate((self.velocities, traj_vel), 
                                                    axis=0)

            i += 1
        # Just consider the angular velocity only
        #self.velocities = self.velocities[:,-1]
        #self.velocities = np.expand_dims(self.velocities, axis=1)
        try:
            self.scans[self.scans == np.inf] = 0
        except:
            print('No trajectory files')

        if not is_test:
            # Normalize data
            # (House dimension: x{-7.5 to 7.5}, y{-5 to 5})
            # Max Linear Vel: 0.5   Max Angular Vel: 2.84
            self.poses[:,:,0] = self.poses[:,:,0] / 7.5
            self.poses[:,:,1] = self.poses[:,:,1] / 5
            self.scans = self.scans / np.max(self.scans)
            self.velocities[:,0] = self.velocities[:,0] / np.max(np.abs(self.velocities[:,0]))
            self.velocities[:,1] = (self.velocities[:,1] + np.abs(np.min(self.velocities[:,1])))
            self.velocities[:,1] = self.velocities[:,1] / np.max(self.velocities[:,1])

            # Separate into train or test set (80%-20%)
            if is_val:
                start, end = -int(self.poses.shape[0]*0.2), -1
            else:
                start, end = 0, -int(self.poses.shape[0]*0.2)

            # Convert numpy to tensors
            self.poses = torch.tensor(self.poses[start:end]).detach().to(self.device).type(torch.float32)
            self.scans = torch.tensor(self.scans[start:end]).detach().to(self.device).type(torch.float32)
            self.velocities = torch.tensor(self.velocities[start:end]).detach().to(self.device).type(torch.float32)

    def __len__(self):
        return self.poses.shape[0]

    def __getitem__(self, i):
        return self.poses[i, :], self.scans[i, :], self.velocities[i, :]


# Unit test
def unitTest():
    train_data = ImitationDataset()
    test_data = ImitationDataset(is_val=True)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=50, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=50, shuffle=True)
    # iterate through example data
    '''for pose, scan, vel in train_loader:
        print('train:', pose.size())
    for pose, scan, vel in test_loader:
        print('test:', pose.size())'''


if __name__ == "__main__":
    unitTest()
