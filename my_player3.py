import logging
import numpy as np
import sys
from agent import AlphaBetaAgent, RLAgent
import constants
from game import Game

sys.stdin = open('input.txt', 'r')
sys.stdout = open('output.txt', 'w')
sys.stderr = open('error.txt', 'w')

logger = logging.getLogger(__name__)
logger.addHandler(logging.FileHandler('my_player3.log'))
logger.setLevel(logging.INFO)
logger.propagate = False


def to_state(board: np.array) -> int:
    size = len(board)
    state = 0

    for cord, stone in np.ndenumerate(board):
        stone = int(stone)
        flat_index = cord[0] * size + cord[1]
        flat_index <<= 1
        state |= (stone << flat_index)

    return state


def get_board(curr_player: int):
    board = []

    def color_correction(stone): return constants.OTHER_STONE[int(
        stone)] if curr_player == constants.WHITE else int(stone)

    for _ in range(constants.BOARD_SIZE):
        board.append(
            list(
                map(color_correction, list(input()))
            )
        )

    return np.array(board)


def get_num_moves() -> int:
    with open(constants.NUM_MOVES_FILENAME, 'r') as fp:
        num_moves = int(fp.read())

    return num_moves


def put_num_moves(num_moves: int) -> None:
    with open(constants.NUM_MOVES_FILENAME, 'w') as fp:
        fp.write(str(num_moves))


def get_next_move_from_rl_agent(curr_state: int):
    game = Game(constants.BOARD_SIZE, game_state=curr_state)
    rl_agent = RLAgent(epsilon=0.0, game=game)

    if curr_state not in rl_agent.q:
        # I don't know, taking help from alpha-beta agent
        return get_next_move_from_ab_agent(game)
    
    action = rl_agent.get_next_action()
    return action


def get_next_move_from_ab_agent():
    ab_agent = AlphaBetaAgent(max_depth=constants)
    action = ab_agent.search(game)
    return action


if __name__ == '__main__':
    curr_player = int(input())

    prev_board = get_board(curr_player)
    prev_state = to_state(prev_board)

    curr_board = get_board(curr_player)
    curr_state = to_state(curr_board)

    curr_player = constants.OTHER_STONE[curr_player]

    if curr_state == 0:
        put_num_moves(0)

    print(curr_state)

    game = Game(constants.BOARD_SIZE, curr_state |
                curr_player << constants.PLAYER_POS)

    num_moves = get_num_moves()

    action = get_next_move_from_rl_agent(curr_state)

    logging.info(f'action: {action}')
    logging.info(f'board: {game.board}')

    put_num_moves(num_moves + 1)

    if action == -1:
        print("PASS")
    else:
        print(action // constants.BOARD_SIZE,
              action % constants.BOARD_SIZE, sep=',')