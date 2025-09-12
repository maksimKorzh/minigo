import os
import sys
import torch
import goban
import requests
import numpy as np

if not os.path.exists('games.txt'):
  print('Downloading dataset...', end='', flush=True)
  response = requests.get('https://github.com/maksimKorzh/minigo/releases/download/minigo/games.txt')
  with open('games.txt', 'w') as f: f.write(response.text)
  print(' Done')

with open('games.txt') as f: games = f.read().splitlines()
game_count = 0
total_samples = 0

def build_training_batch(from_index, to_index):
  global game_count, total_samples
  states_batch = []
  moves_batch = []
  values_batch = []
  batch_games = games[from_index:min(to_index, len(games))]
  for game in batch_games:
    game_count += 1
    print(f'Encoding game {game_count}')
    goban.width=19+2
    goban.init_board()
    game_result = 1 if game.split('|')[-1] == 'B' else -1
    moves = game.split('|')[0].split(';')
    for move_num, move in enumerate(moves):
      if len(move):
        state_tensor = goban.board_to_tensor()
        move_index = goban.load_sgf_move(move)
        current_player = 1 if (move_num % 2 == 0) else -1
        value_target = game_result * current_player
        states_batch.append(state_tensor)
        moves_batch.append(move_index)
        values_batch.append(value_target)
  states_tensor = torch.tensor(np.array(states_batch), dtype=torch.uint8)
  moves_tensor = torch.tensor(np.array(moves_batch), dtype=torch.int16)
  values_tensor = torch.tensor(np.array(values_batch), dtype=torch.float32)
  torch.save((states_tensor, moves_tensor, values_tensor), f'games_{from_index}-{to_index}.pt')
  print(f"Saved batch with {len(states_batch)} samples")
  states_batch.clear()
  moves_batch.clear()

if len(sys.argv) < 3: print('Usage: python build_dataset.py FIRST_GAME_INDEX LAST_GAME_INDEX')
else:
  try:
    first = int(sys.argv[1])
    last = int(sys.argv[2])
    build_training_batch(first, last)
  except: print('Usage: python build_dataset.py FIRST_GAME_INDEX LAST_GAME_INDEX')
