import torch
from model import MinigoNet
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MinigoNet().to(device)
checkpoint = torch.load('./test/minigo.pth', map_location=device)
model.load_state_dict(checkpoint)
model.eval()

folder = 'test'
for file_name in os.listdir(folder):
  if file_name.endswith(".pt"):
    states, moves, values = torch.load(os.path.join(folder, file_name))
    states = states.float().to(device)
    moves = moves.long().to(device)
    values = values.float().to(device)
    with torch.no_grad():
      policy_logits, value_pred = model(states)
    pred_moves = policy_logits.argmax(dim=1)
    policy_acc = (pred_moves == moves).float().mean().item()
    value_mse = torch.mean((value_pred.squeeze(-1) - values) ** 2).item()
    print(f'{file_name} | Policy accuracy: {policy_acc*100:.2f}% | Value MSE: {value_mse:.4f}')
