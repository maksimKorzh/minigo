import sys
import torch
import goban
from model import MinigoNet

BOARD_SIZE = 19

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MinigoNet()
checkpoint = torch.load('minigo.pth', map_location=device)
model.load_state_dict(checkpoint)
model.to(device)
model.eval()

def search(color):
  move, value = nn_move(goban.board_to_tensor(), color)
  print(move, file=sys.stderr)
  if move != [goban.NONE, goban.NONE]:
    goban.play(move[0], move[1], color)
    print(f'winrate: {value}', file=sys.stderr)
    return 'ABCDEFGHJKLMNOPQRST'[move[0]-1] + str(BOARD_SIZE - move[1]+1)
  else: return 'pass'

def nn_move(board_array, color):
  board_tensor = torch.tensor(board_array, dtype=torch.float32).unsqueeze(0).to(device)
  with torch.no_grad():
    policy_logits, value = model(board_tensor)
    probs = torch.exp(policy_logits).squeeze(0).cpu().numpy()
  move_indices = probs.argsort()[::-1]
  for i, best_move_idx in enumerate(move_indices):
    row, col = divmod(best_move_idx, BOARD_SIZE)
    if goban.board[row+1][col+1] == goban.EMPTY and (col+1, row+1) != goban.ko:
      if not goban.is_suicide(col+1, row+1, color):
        return [int(col+1), int(row+1)], value.item()
    if i > 5: return [goban.NONE, goban.NONE], value.item()
  return [goban.NONE, goban.NONE], value.item()
