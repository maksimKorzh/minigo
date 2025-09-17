import gc
import sys
import torch
import goban
import numpy as np
from copy import deepcopy
from model import MinigoNet

MCTS = True

analysis = {'is': False}
first_out = True
last_out = False
info_str = {'val': ''}

BOARD_SIZE = 19
CPUCT = 1.5
NUM_SIMULATIONS = 10
TOP_K = 5
Q = {}
N = {}
P = {}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MinigoNet()
checkpoint = torch.load('minigo.pth', map_location=device)
model.load_state_dict(checkpoint)
model.to(device)
model.eval()

def is_legal(move, color):
  c, r = move
  if move == (goban.NONE, goban.NONE): return True
  if goban.board[r][c] != goban.EMPTY: return False
  if (c, r) == tuple(goban.ko): return False
  if goban.is_suicide(c, r, color): return False
  return True

def nn_topk_moves(board_array, color, k=TOP_K):
  board_tensor = torch.tensor(board_array, dtype=torch.float32).unsqueeze(0).to(device)
  with torch.no_grad():
    policy_logits, value = model(board_tensor)
    policy_logits = policy_logits.squeeze(0).cpu().numpy()
    probs = np.exp(policy_logits)
    probs /= probs.sum()
  move_indices = probs.argsort()[::-1]
  top_moves = []
  illegal_count = 0
  for idx in move_indices:
    row, col = divmod(idx, BOARD_SIZE)
    r, c = row+1, col+1
    if is_legal((c, r), color): top_moves.append(((c, r), float(probs[idx])))
    else: illegal_count += 1
    if len(top_moves) >= k or illegal_count >= k: break
  while len(top_moves) < k:
    top_moves.append(((goban.NONE, goban.NONE), 0.0))
  return top_moves, float(value.item())

def top_k_moves():
  moves, value = nn_topk_moves(goban.board_to_tensor(), goban.side, TOP_K)
  return moves, value

def mcts(color, ponder):
  global first_out, last_out, info_str
  first_out = True
  last_out = False
  info_str['val'] = ''
  Q.clear()
  N.clear()
  P.clear()
  gc.collect()
  old_side = goban.side
  goban.side = color
  root_moves, _ = top_k_moves()
  legal_root_moves = [m for m,_ in root_moves if is_legal(m, goban.side)]
  if legal_root_moves[-1] == (goban.NONE, goban.NONE): return legal_root_moves[-1]
  if not analysis['is']:
    for _ in range(NUM_SIMULATIONS): simulate(color, ponder)
  else:
    while analysis['is'] == True:
      simulate(color, ponder)
      if analysis['is'] == False: break
  best = max(legal_root_moves, key=lambda m: N.get(m, 0))
  goban.side = old_side
  return best

def simulate(color, ponder):
  global first_out, last_out, info_str
  path = []
  board_copy = deepcopy(goban.board)
  side_copy = goban.side
  ko_copy = goban.ko[:]
  groups_copy = deepcopy(goban.groups)
  history_copy = deepcopy(goban.move_history)
  while True:
    moves, _ = top_k_moves()
    unexplored = [m for m,_ in moves if m not in P]
    if unexplored:
      move_to_expand = unexplored[0]
      for m,p in moves:
        if is_legal(m, goban.side): P[m] = p
      break
    total_n = sum(N.get(m,0) for m,_ in moves)
    ucb_values = []
    for m, p in moves:
      q = Q.get(m, 0)
      n = N.get(m, 0)
      u = CPUCT * p * np.sqrt(total_n + 1) / (1 + n)
      ucb_values.append(q + u)
    best_move = moves[np.argmax(ucb_values)][0]
    if not is_legal(best_move, goban.side): break
    path.append(best_move)
    if best_move == (goban.NONE, goban.NONE): goban.pass_move()
    else: goban.play(best_move[0], best_move[1], goban.side)
  _, value = top_k_moves()
  for m in path:
    old_q = Q.get(m, 0)
    old_n = N.get(m, 0)
    Q[m] = (old_q * old_n + value) / (old_n + 1)
    N[m] = old_n + 1
  root_moves, _ = top_k_moves()
  for move, prior in root_moves:
    if not is_legal(move, goban.side): continue
    visits = N.get(move, 0)
    p_side = 0.5 * (Q.get(move, 0) + 1.0)
    if color == goban.BLACK: winrate = p_side
    else: winrate = 1.0 - p_side
    if ponder:
      if analysis['is'] == True:
        m = goban.coords_to_move(move)
        pv = ' pv ' + m + ' '
        if visits < 10: pv = ' '
        info_str['val'] += 'info move ' + m + ' visits ' + str(visits) + ' winrate ' + str(winrate) + ' prior ' + str(prior) + pv
      else: last_out = True
  if ponder:
    if first_out == True and last_out == False: print('=', file=sys.stdout, flush=True)
    print(info_str['val'], file=sys.stdout, flush=True)
    first_out = False
  goban.board = board_copy
  goban.side = side_copy
  goban.ko = ko_copy
  goban.history = history_copy
  goban.groups = groups_copy

def nn_move(board_array, color):
  board_tensor = torch.tensor(board_array, dtype=torch.float32).unsqueeze(0).to(device)
  with torch.no_grad():
    policy_logits, value = model(board_tensor)
    probs = torch.exp(policy_logits).squeeze(0).cpu().numpy()
  move_indices = probs.argsort()[::-1]
  for i, best_move_idx in enumerate(move_indices):
    row, col = divmod(best_move_idx, BOARD_SIZE)
    if is_legal((col+1,row+1), color):
      return (int(col+1), int(row+1)), value.item()
    if i > 5: return (goban.NONE, goban.NONE), value.item()
  return (goban.NONE, goban.NONE), value.item()

def search(color, ponder):
  if MCTS: move = mcts(color, False)
  else:
    move, value = nn_move(goban.board_to_tensor(), color)
  if move != (goban.NONE, goban.NONE):
    goban.play(move[0], move[1], color)
    net_move, nn_value = nn_move(goban.board_to_tensor(), color)
    return 'ABCDEFGHJKLMNOPQRST'[move[0]-1] + str(BOARD_SIZE - move[1]+1)
  else: return 'pass'
