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
logger.setLevel(logging.DEBUG)
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

    for _ in range(constants.BOARD_SIZE):
        board.append(
            list(
                map(
                    lambda stone:
                        constants.OTHER_STONE[int(stone)] if curr_player == constants.WHITE
                        else int(stone), 
                    list(input())
                )
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


def get_next_move_from_rl_agent(game: Game):
    rl_agent = RLAgent(epsilon=0.0, game=game)

    if (curr_state not in rl_agent.q) or len(rl_agent.q[curr_state]) < 4:
        logger.debug('alpha-beta')
        # I don't know, taking help from alpha-beta agent
        return get_next_move_from_ab_agent(game)

    logger.debug('q-learning')
    
    action = rl_agent.get_next_action()
    return action


def get_next_move_from_ab_agent(game):
    ab_agent = AlphaBetaAgent(max_depth=constants.MAX_DEPTH)
    action = ab_agent.search(game)
    return action


if __name__ == '__main__':
    curr_player = int(input())

    prev_board = get_board(curr_player)
    prev_state = to_state(prev_board)

    curr_board = get_board(curr_player)
    curr_state = to_state(curr_board)

    curr_player = constants.BLACK

    logger.debug(f'{curr_state}: curr_state')

    if curr_state == 0:
        put_num_moves(0)

    game = Game(constants.BOARD_SIZE, curr_state |
                curr_player << constants.PLAYER_POS)

    num_moves = get_num_moves()

    action = get_next_move_from_rl_agent(game)

    logger.debug(f'action: {action}')
    logger.debug(f'board: {game.board}')

    put_num_moves(num_moves + 1)

    if action == -1:
        print("PASS")
    else:
        print(action // constants.BOARD_SIZE,
              action % constants.BOARD_SIZE, sep=',')
