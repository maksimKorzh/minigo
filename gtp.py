import sys
import torch
import goban
import threading
from search import *

goban.width = 19+2
goban.init_board()
thread = None

while True:
  command = input()
  info_str['val'] = ''
  analysis['is'] = False
  try: thread.join()
  except: pass
  if 'name' in command: print('= minigo\n')
  elif 'protocol_version' in command: print('= 2\n');
  elif 'version' in command: print('=', 'by Code Monkey King\n')
  elif 'list_commands' in command: print('= protocol_version\nanalyze\n')
  elif 'boardsize' in command: goban.width = int(command.split()[1])+2; print('=\n')
  elif 'clear_board' in command: goban.init_board(); print('=\n')
  elif 'showboard' in command: print('= ', end=''); goban.print_board()
  elif 'play' in command:
    try:
      analysis['is'] = False
      thread.join()
    except: pass
    if 'pass' not in command and 'pass'.upper() not in command:
      params = command.split()
      color = goban.BLACK if params[1] == 'B' else goban.WHITE
      col, row = goban.move_to_coords(params[2])
      goban.play(col, row, color)
      print('=\n')
    else:
      goban.pass_move()
      print('=\n')
  elif 'genmove' in command:
    params = command.split()
    color = goban.BLACK if params[1].lower() == 'b' else goban.WHITE
    move = search(color, False)
    print(f'= {move}\n')
  elif 'analyze' in command:
    analysis['is'] = True
    params = command.split()
    color = goban.side
    thread = threading.Thread(target=mcts, args=(color, True))
    thread.start()
  elif 'stop' in command:
    print('=\n')
  elif 'quit' in command:
    print('=\n')
    sys.exit()
  else: print('=\n')
