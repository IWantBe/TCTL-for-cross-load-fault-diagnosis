import torch
import torch.nn as nn
import torch.nn.functional as F
import mmd as mmd

# Define model--CNN
class CNN1(nn.Module):
    def __init__(self):
        super(CNN1, self).__init__()
        self.conv1 = nn.Sequential(         # input shape (10,1,401)
            nn.Conv1d(
                in_channels=1,              # input height
                out_channels=64,            # n_filters
                kernel_size=20,              # filter size
                stride=1,
                padding='same'  # filter movement/step
            ),                              # output shape (16, 1, 2048)
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),                      # activation
            nn.MaxPool1d(kernel_size=2),    # choose max value in 2x2 area, output shape (16, 1, 1024)
        )
        self.conv2 = nn.Sequential(         # input shape (16, 14, 14)
            nn.Conv1d(64, 32, 20, 1, 'same'),     # output shape (32, 14, 14)
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),                 # activation
            # nn.MaxPool1d(2),                # output shape (32, 7, 7)
        )
        self.conv3 = nn.Sequential(         # input shape (16, 14, 14)
            nn.Conv1d(32, 32, 20, 1, 'same'),     # output shape (32, 14, 14)
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),                 # activation
            # nn.MaxPool1d(2),                # output shape (32, 7, 7)
        )
        self.conv4 = nn.Sequential(         # input shape (16, 14, 14)
            nn.Conv1d(32, 32, 20, 1, 'same'),     # output shape (32, 14, 14)
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),                 # activation
            nn.MaxPool1d(2),                # output shape (32, 7, 7)
        )
        self.fc1 = nn.Linear(3360, 100)  # 适合随机取样
        self.fc2 = nn.Linear(100, 10)   # fully connected layer, output 10 classes
        self.relu = nn.ReLU()

    def forward(self, x_src, x_tar):
        x_src = self.conv1(x_src)
        x_tar = self.conv1(x_tar)
        x_src = self.conv2(x_src)
        x_tar = self.conv2(x_tar)
        x_src = self.conv3(x_src)
        x_tar = self.conv3(x_tar)
        x_src = self.conv4(x_src)
        x_tar = self.conv4(x_tar)
        x_src = x_src.view(x_src.size(0), -1)  # flatten the output of conv2 to (64, 3360)
        # print(len(x_src[0]))
        x_tar = x_tar.view(x_tar.size(0), -1)  # flatten the output of conv2 to (64, 3360)
        x_src_mmd = self.fc1(x_src)   # [64,100]
        x_src_mmd1 = self.relu(x_src_mmd)
        x_tar_mmd = self.fc1(x_tar)   # [64,100]
        x_tar_mmd1 = self.relu(x_tar_mmd)
        y_src = self.fc2(x_src_mmd1)  # [64,10]
        y_tar = self.fc2(x_tar_mmd1)
        return y_src, y_tar, x_src_mmd, x_tar_mmd
