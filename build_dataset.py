import os
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
  batch_games = games[from_index:min(to_index, len(games))]
  for game in batch_games:
    game_count += 1
    print(f'Encoding game {game_count}')
    goban.width=19+2
    goban.init_board()
    for move in game.split('|')[0].split(';'):
      if len(move):
        state_tensor = goban.board_to_tensor()
        move_index = goban.load_sgf_move(move)
        states_batch.append(state_tensor)
        moves_batch.append(move_index)
  states_tensor = torch.tensor(np.array(states_batch), dtype=torch.uint8)
  moves_tensor = torch.tensor(np.array(moves_batch), dtype=torch.int16)
  torch.save((states_tensor, moves_tensor), f'games_{from_index}-{to_index}.pt')
  print(f"Saved batch with {len(states_batch)} samples")
  states_batch.clear()
  moves_batch.clear()
build_training_batch(0, 10)
