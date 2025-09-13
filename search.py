import sys
import torch
import goban
import numpy as np
from copy import deepcopy
from model import MinigoNet

BOARD_SIZE = 19

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MinigoNet()
checkpoint = torch.load('minigo.pth', map_location=device)
model.load_state_dict(checkpoint)
model.to(device)
model.eval()




# --- MCTS parameters ---
CPUCT = 1.5
NUM_SIMULATIONS = 500
TOP_K = 5

# --- Global MCTS tables ---
Q = {}  # Q[move_tuple] = mean value
N = {}  # N[move_tuple] = visit count
P = {}  # P[move_tuple] = NN prior probability

# --- NN wrapper to get top-K moves and value ---
def nn_topk_moves(board_array, color, k=TOP_K):
  board_tensor = torch.tensor(board_array, dtype=torch.float32).unsqueeze(0).to(device)
  with torch.no_grad():
    policy_logits, value = model(board_tensor)
    policy_logits = policy_logits.squeeze(0).cpu().numpy()
    probs = np.exp(policy_logits)
    probs /= probs.sum()  # softmax

  move_indices = probs.argsort()[::-1]
  top_moves = []

  for idx in move_indices:
    row, col = divmod(idx, BOARD_SIZE)
    r, c = row+1, col+1  # goban uses 1-based indices
    if goban.board[r][c] == goban.EMPTY and (c, r) != tuple(goban.ko):
      if not goban.is_suicide(c, r, color):
        top_moves.append(((c, r), float(probs[idx])))
    if len(top_moves) >= k:
      break

  # fill remaining slots with pass
  while len(top_moves) < k:
    top_moves.append(((goban.NONE, goban.NONE), 0.0))

  return top_moves, float(value.item())

# --- Helper: get top-K legal moves for current board ---
def top_k_moves():
  board_copy = deepcopy(goban.board)
  side_copy = goban.side
  ko_copy = goban.ko[:]

  moves, value = nn_topk_moves(goban.board_to_tensor(), goban.side, TOP_K)

  goban.board = board_copy
  goban.side = side_copy
  goban.ko = ko_copy

  return moves, value

# --- MCTS selection & simulation ---
def select_action():
  for _ in range(NUM_SIMULATIONS):
    simulate()
  # pick move with highest visit count
  best_move = max(N, key=N.get)
  return best_move

def simulate():
  path = []
  board_copy = deepcopy(goban.board)
  side_copy = goban.side
  ko_copy = goban.ko[:]

  while True:
    moves, _ = top_k_moves()
    unexplored = [m for m,_ in moves if m not in P]
    if unexplored:
      move_to_expand = unexplored[0]
      # store NN priors for this node
      for m,p in moves:
        P[m] = p
      break

    # PUCT selection
    total_n = sum(N.get(m,0) for m,_ in moves)
    ucb_values = []
    for m, p in moves:
      q = Q.get(m, 0)
      n = N.get(m, 0)
      u = CPUCT * p * np.sqrt(total_n + 1) / (1 + n)
      ucb_values.append(q + u)
    best_move = moves[np.argmax(ucb_values)][0]
    path.append(best_move)

    if best_move == (goban.NONE, goban.NONE):
      goban.pass_move()
    else:
      goban.play(best_move[0], best_move[1], goban.side)

  # Expansion & evaluation
  _, value = top_k_moves()

  # Backpropagation
  for m in path:
    old_q = Q.get(m, 0)
    old_n = N.get(m, 0)
    Q[m] = (old_q * old_n + value) / (old_n + 1)
    N[m] = old_n + 1

  for move in P:
    visits = N.get(move, 0)
    winrate = Q.get(move, 0)
    prior = P.get(move, 0)
    print(f'info move {goban.coords_to_move(move)} visits {visits} winrate {winrate:.6f} prior {prior:.6f}', end=' ')

  # Restore board
  goban.board = board_copy
  goban.side = side_copy
  goban.ko = ko_copy


#info move Q16 visits 0 edgeVisits 0 utility -0.12444 winrate 0.439639 scoreMean -1.0947 scoreStdev 24.0794 scoreLead -1.0947 scoreSelfplay -1.41354 prior 0.123982 lcb -4.56036 utilityLcb -14 weight 0 order 0 pv Q16 info move Q4 visits 0 edgeVisits 0 utility -0.12444 winrate 0.439639 scoreMean -1.0947 scoreStdev 24.0794 scoreLead -1.0947 scoreSelfplay -1.41354 prior 0.123982 lcb -4.56036 utilityLcb -14 weight 0 isSymmetryOf Q16 order 1 pv Q4 info move D16 visits 0 edgeVisits 0 utility -0.12444 winrate 0.439639 scoreMean -1.0947 scoreStdev 24.0794 scoreLead -1.0947 scoreSelfplay -1.41354 prior 0.123982 lcb -4.56036
#info move R8 visits 0 winrate 0.000 prior 0.013 info move C5 visits 0 winrate 0.000 prior 0.033 info move L17 visits 0 winrate 0.000 prior 0.021 info move D6 visits 0 winrate 0.000 prior 0.052 info move H3 visits 0 winrate 0.000 prior 0.090 info move S4 visits 0 winrate 0.000 prior 0.024 info move D10 visits 0 winrate 0.000 prior 0.090 info move C9 visits 0 winrate 0.000 prior 0.062 info move R16 visits 13 winrate -0.080 prior 0.096 info move Q16 visits 10 wi



















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

#goban.width = 19+2
#goban.init_board()
#move = select_action()
#col, row = move
#if move == (goban.NONE, goban.NONE): goban.pass_move()
#else: goban.play(col, row, goban.side)
#goban.print_board()
