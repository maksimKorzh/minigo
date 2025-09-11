import sys
import torch
import goban

goban.width = 19+2
goban.init_board()

while True:
  command = input()
  if 'name' in command: print('= minigo\n')
  elif 'protocol_version' in command: print('= 2\n');
  elif 'version' in command: print('=', 'by Code Monkey King\n')
  elif 'list_commands' in command: print('= protocol_version\n')
  elif 'boardsize' in command: goban.width = int(command.split()[1])+2; print('=\n')
  elif 'clear_board' in command: goban.init_board(); print('=\n')
  elif 'showboard' in command: print('= ', end=''); goban.print_board()
  elif 'play' in command:
    if 'pass'.upper() not in command:
      params = command.split()
      color = goban.BLACK if params[1] == 'B' else goban.WHITE
      col, row = goban.move_to_coords(params[2])
      goban.play(col, row, color)
      print('=\n')
    else:
      goban.side = (3-goban.side)
      goban.ko = [goban.NONE, goban.NONE]
      print('=\n')
  elif 'genmove' in command: print('= pass\n')
  elif 'quit' in command: sys.exit()
  else: print('=\n')
