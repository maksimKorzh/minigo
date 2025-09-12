import torch
import torch.nn as nn
import torch.nn.functional as F

BLOCKS = 10
FILTERS = 128
INPUT_FEATURES = 16
BOARD_SIZE = 19

class ResidualBlock(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(FILTERS, FILTERS, kernel_size=3, padding=1, bias=False)
    self.bn1 = nn.BatchNorm2d(FILTERS)
    self.conv2 = nn.Conv2d(FILTERS, FILTERS, kernel_size=3, padding=1, bias=False)
    self.bn2 = nn.BatchNorm2d(FILTERS)

  def forward(self, x):
    shortcut = x
    out = self.conv1(x)
    out = self.bn1(out)
    out = F.relu(out)
    out = self.conv2(out)
    out = self.bn2(out)
    out = out + shortcut
    out = F.relu(out)
    return out

class MinigoNet(nn.Module):
  def __init__(self):
    super().__init__()
    self.board_size = BOARD_SIZE
    self.conv_in = nn.Conv2d(INPUT_FEATURES, FILTERS, kernel_size=3, padding=1, bias=False)
    self.bn_in = nn.BatchNorm2d(FILTERS)
    self.res_blocks = nn.Sequential(*[ResidualBlock() for _ in range(BLOCKS)])
    self.conv_policy = nn.Conv2d(FILTERS, 2, kernel_size=1, bias=False)
    self.bn_policy = nn.BatchNorm2d(2)
    self.fc_policy = nn.Linear(2 * BOARD_SIZE * BOARD_SIZE, BOARD_SIZE * BOARD_SIZE)
    self.conv_value = nn.Conv2d(FILTERS, 1, kernel_size=1, bias=False)
    self.bn_value = nn.BatchNorm2d(1)
    self.fc_value1 = nn.Linear(BOARD_SIZE * BOARD_SIZE, 64)
    self.fc_value2 = nn.Linear(64, 1)

  def forward(self, x):
    batch_size = x.size(0)
    x = x.view(batch_size, INPUT_FEATURES, self.board_size, self.board_size)
    x = self.conv_in(x)
    x = self.bn_in(x)
    x = F.relu(x)
    x = self.res_blocks(x)
    policy = self.conv_policy(x)
    policy = self.bn_policy(policy)
    policy = F.relu(policy)
    policy = policy.view(batch_size, -1)
    policy = self.fc_policy(policy)
    policy = F.log_softmax(policy, dim=1)
    value = self.conv_value(x)
    value = self.bn_value(value)
    value = F.relu(value)
    value = value.view(batch_size, -1)
    value = F.relu(self.fc_value1(value))
    value = torch.tanh(self.fc_value2(value))
    return policy, value
