import torch
import torch.nn as nn
import torch.nn.functional as F


class Chomp1d(nn.Module):
    """移除卷積後多餘的 Padding，確保時間序列的因果性 (不偷看未來)"""

    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """單層 TCN 區塊"""

    def __init__(self,
                 n_inputs,
                 n_outputs,
                 kernel_size,
                 stride,
                 dilation,
                 padding,
                 dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs,
                               n_outputs,
                               kernel_size,
                               stride=stride,
                               padding=padding,
                               dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs,
                               n_outputs,
                               kernel_size,
                               stride=stride,
                               padding=padding,
                               dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1,
                                 self.dropout1, self.conv2, self.chomp2,
                                 self.relu2, self.dropout2)

        self.downsample = nn.Conv1d(n_inputs, n_outputs,
                                    1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNBranch(nn.Module):
    """單一時間框架的 TCN 通道"""

    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2):
        super(TCNBranch, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2**i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [
                TemporalBlock(in_channels,
                              out_channels,
                              kernel_size,
                              stride=1,
                              dilation=dilation_size,
                              padding=(kernel_size - 1) * dilation_size,
                              dropout=dropout)
            ]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # x shape: (batch_size, num_features, sequence_length)
        return self.network(x)


class ParallelTCNAlphaHunter(nn.Module):
    """核心模型：三通道平行 TCN"""

    def __init__(self,
                 input_features=5,
                 tcn_channels=[16, 32],
                 kernel_size=3,
                 dropout=0.3,
                 num_classes=3):
        super(ParallelTCNAlphaHunter, self).__init__()

        # 三個獨立的通道分別提取 1H, 4H, 1D 特徵
        self.tcn_1h = TCNBranch(input_features, tcn_channels, kernel_size,
                                dropout)
        self.tcn_4h = TCNBranch(input_features, tcn_channels, kernel_size,
                                dropout)
        self.tcn_1d = TCNBranch(input_features, tcn_channels, kernel_size,
                                dropout)

        # 將三者特徵融合 (Concat)，然後經過 MLP 輸出
        final_tcn_dim = tcn_channels[-1]
        self.fc1 = nn.Linear(final_tcn_dim * 3, 64)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(
            64, num_classes)  # 輸出: [Hold(0), Buy(1), Sell(-1)] 的 Logits

    def forward(self, x_1h, x_4h, x_1d):
        # 注意: PyTorch 的 Conv1d 預期輸入形狀為 (Batch, Channels/Features, Length)
        # 我們取最後一個時間點的輸出作為該時間序列的特徵
        out_1h = self.tcn_1h(x_1h)[:, :, -1]
        out_4h = self.tcn_4h(x_4h)[:, :, -1]
        out_1d = self.tcn_1d(x_1d)[:, :, -1]

        # 特徵融合
        combined = torch.cat([out_1h, out_4h, out_1d], dim=1)

        # 決策層
        x = F.relu(self.fc1(combined))
        x = self.dropout(x)
        logits = self.fc2(x)
        return logits
