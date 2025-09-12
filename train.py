import sys
import os
import torch
import torch.optim as optim
import torch.nn as nn
from model import *
import numpy as np

torch.manual_seed(1337)

if len(sys.argv) < 5:
  print('Usage: python train.py <DEVICE> <BATCH_SIZE> <LEARNING_RATE> <DATASET_PATH>')
  sys.exit()

device = torch.device(sys.argv[1])
batch_size = int(sys.argv[2])
learning_rate = float(sys.argv[3])
dataset_path = sys.argv[4]
checkpoint_path = 'minigo_checkpoint.pth'

model = MinigoNet()
model.to(device)

optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
policy_criterion = nn.CrossEntropyLoss()
value_criterion = nn.MSELoss()

states, moves, values = torch.load(dataset_path)
states = states.float()
moves = moves.long()
values = values.float()

dataset_size = states.size(0)
print(f'Loaded dataset with {dataset_size} samples.')

if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    print(f'Loaded checkpoint')

indices = torch.randperm(dataset_size)
states = states[indices].to(device)
moves = moves[indices].to(device)
values = values[indices].to(device)

for start in range(0, dataset_size, batch_size):
  end = min(start + batch_size, dataset_size)
  batch_states = states[start:end]
  batch_moves = moves[start:end]
  batch_values = values[start:end]
  optimizer.zero_grad()
  policy_logits, value_pred = model(batch_states)
  policy_loss = policy_criterion(policy_logits, batch_moves)
  value_loss = value_criterion(value_pred.squeeze(-1), batch_values)
  loss = policy_loss + value_loss
  loss.backward()
  optimizer.step()
  print(f'Iter {end}/{dataset_size} | Loss: {loss.item():.4f} | Policy: {policy_loss.item():.4f} | Value: {value_loss.item():.4f}')

torch.save({
  'model_state': model.state_dict(),
  'optimizer_state': optimizer.state_dict()
}, checkpoint_path)
torch.save(model.state_dict(), 'minigo.pth')

try:
  from google.colab import files
  files.download(checkpoint_path)
  files.download('minigo.pth')
except: pass

print(f'Model checkpoint saved to {checkpoint_path}')
