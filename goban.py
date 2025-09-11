import sys
import copy
import random
from copy import deepcopy

NONE = -1
EMPTY = 0
BLACK = 1
WHITE = 2
FENCE = 3
ESCAPE = 4

width = NONE
board = [[]]
side = NONE
ko = [NONE, NONE]
groups = []

def init_board():
  global board, side, ko, groups
  board = [[0 for _ in range(width)] for _ in range(width)]
  for row in range(width):
    for col in range(width):
      if row == 0 or row == width-1 or col == 0 or col == width-1:
        board[row][col] = FENCE
  side = BLACK
  ko = [NONE, NONE]
  groups = [[], []]

def print_board():
  for row in range(width):
    for col in range(width):
      if col == 0 and row != 0 and row != width-1:
        rown = width-row-1
        print((' ' if rown < 10 else ''), rown, end=' ')
      if board[row][col] == FENCE: continue
      if col == ko[0] and row == ko[1]: print('#', end=' ')
      else: print(['.', 'X', 'O', '#'][board[row][col]], end=' ')
    if row < width-1: print()
  print('   ', 'A B C D E F G H J K L M N O P Q R S T'[:width*2-4])
  print('\n    Side to move:', ('BLACK' if side == 1 else 'WHITE'), file=sys.stderr)
  print('\n              Ko:', ('None' if move_to_string(ko)[0] == '?' else move_to_string(ko)), file=sys.stderr)
  print()

def print_groups():
  print('    Black groups:')
  for group in groups[BLACK-1]: print('      ', group)
  print('\n    White groups:')
  for group in groups[WHITE-1]: print('      ', group)
  print()

def count(col, row, color, marks):
  stone = board[row][col]
  if stone == FENCE: return
  if stone and (stone & color) and marks[row][col] == EMPTY:
    marks[row][col] = stone
    count(col+1, row, color, marks)
    count(col-1, row, color, marks)
    count(col, row+1, color, marks)
    count(col, row-1, color, marks)
  elif stone == EMPTY:
    marks[row][col] = ESCAPE

def add_stones(marks, color):
  group = {'stones': [], 'liberties' :[]}
  for row in range(width):
    for col in range(width):
      stone = marks[row][col]
      if stone == FENCE or stone == EMPTY: continue
      if stone == ESCAPE: group['liberties'].append((col, row))
      else: group['stones'].append((col, row))
  return group

def make_group(col, row, color):
  marks = [[EMPTY for _ in range(width)] for _ in range(width)]
  count(col, row, color, marks)
  return add_stones(marks, color)

def update_groups():
  global groups
  groups = [[], []]
  for row in range(width):
    for col in range(width):
      stone = board[row][col]
      if stone == FENCE or stone == EMPTY: continue

      if stone == BLACK:
        group = make_group(col, row, BLACK)
        if group not in groups[BLACK-1]: groups[BLACK-1].append(group)
      if stone == WHITE:
        group = make_group(col, row, WHITE)
        if group not in groups[WHITE-1]: groups[WHITE-1].append(group)

def is_clover(col, row):
  clover_color = -1
  other_color = -1
  for stone in [board[row][col+1], board[row][col-1], board[row+1][col], board[row-1][col]]:
    if stone == FENCE: continue
    if stone == EMPTY: return EMPTY
    if clover_color == -1:
      clover_color = stone
      other_color = (3-clover_color)
    elif stone == other_color: return EMPTY
  return clover_color

def is_suicide(col, row, color):
  suicide = False
  board[row][col] = color
  marks = [[EMPTY for _ in range(width)] for _ in range(width)]
  count(col, row, color, marks)
  group = add_stones(marks, color)
  if len(group['liberties']) == 0: suicide = True
  board[row][col] = EMPTY
  return suicide

def is_atari(col, row, color):
  atari = False
  board[row][col] = color
  marks = [[EMPTY for _ in range(width)] for _ in range(width)]
  count(col, row, color, marks)
  group = add_stones(marks, color)
  if len(group['liberties']) == 1: atari = True
  board[row][col] = EMPTY
  return atari

def play(col, row, color):
  global ko, side
  ko = [NONE, NONE]
  board[row][col] = color
  update_groups()
  for group in groups[(3-color-1)]:
    if len(group['liberties']) == 0:
      if len(group['stones']) == 1 and is_clover(col, row) == (3-color):
        ko = group['stones'][0]
      for stone in group['stones']:
        board[stone[1]][stone[0]] = EMPTY
  side = (3-color)

def is_ladder(col, row, color, first_run):
  group = make_group(col, row, color)
  if len(group['liberties']) == 0: return 1
  if len(group['liberties']) == 1:
    if board[row][col] != EMPTY and first_run == False:
      if len(group['liberties']) <= 1: return 1
      else: return 0
    board[row][col] = color
    new_col = group['liberties'][0][0]
    new_row = group['liberties'][0][1]
    if is_ladder(new_col, new_row, color, False): return 1
    board[row][col] = EMPTY
  if len(group['liberties']) == 2:
    for move in group['liberties']:
      board[move[1]][move[0]] = (3-color)
      group = make_group(col, row, color)
      new_col = group['liberties'][0][0]
      new_row = group['liberties'][0][1]
      if is_ladder(new_col, new_row, color, False): return move
      board[move[1]][move[0]] = EMPTY
  return 0

def check_ladder(col, row, color):
  global board
  current_board = copy.deepcopy(board)
  ladder = is_ladder(col, row, color, True)
  board = copy.deepcopy(current_board)
  return ladder

def move_to_string(move):
  global width
  col = chr(move[0]-(1 if move[0]<=8 else 0)+ord('A'))
  row = str(width-move[1]-1)
  return col+row
