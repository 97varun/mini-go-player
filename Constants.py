EMPTY = 0
BLACK = 1
WHITE = 2

CELL_TO_REP = {0: '.', 1: 'X', 2: 'O'}

OTHER_STONE = {1: 2, 2: 1, 0: 0}

WIN_REWARD = 1
LOSS_REWARD = -1

WIN_SCORE = 50
LOSS_SCORE = -50

MASK = 3
PLAYER_POS = 50
CURR_PLAYER_BLACK = BLACK << PLAYER_POS
CURR_PLAYER_WHITE = WHITE << PLAYER_POS

MAX_DEPTH = 4

NO_ACTION = -2

BOARD_SIZE = 5

MAX_MOVES = 24

NUM_MOVES_FILENAME = 'num_moves.txt'
Q_TABLE_FILENAME = 'qtable.pkl'
EPSILON_FILENAME = 'epsilon.txt'

MAX_QTABLE_SIZE = 2000000

MAX_POSSIBLE_ACTIONS = 10