import logging
import numpy as np
import sys
from agent import AlphaBetaAgent, RLAgent
import constants
from game import Game
import time

sys.stdin = open('input.txt', 'r')
sys.stdout = open('output.txt', 'w')
sys.stderr = open('error.txt', 'w')

logger = logging.getLogger(__name__)
logger.addHandler(logging.FileHandler('my_player3.txt'))
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


def get_next_move_from_rl_agent(game: Game, last_move: int) -> int:
    rl_agent = RLAgent(epsilon=0.0, game=game)

    if (curr_state not in rl_agent.q) or len(rl_agent.q[curr_state]) < 4:
        # I don't know, taking help from alpha-beta agent
        return get_next_move_from_ab_agent(game, last_move)

    action = rl_agent.get_next_action()
    return action


def get_next_move_from_ab_agent(game: Game) -> int:
    if game.num_moves == 0 or game.num_moves == 1:
        if game.board[2, 2] == constants.EMPTY:
            return 12
        elif game.board[1, 1] == constants.EMPTY:
            return 6
    
    max_depth = 6 if game.num_moves >= 18 else 4
    
    ab_agent = AlphaBetaAgent(max_depth=max_depth)
    
    start_time = time.time()
    
    action = ab_agent.search(game)
    
    logger.info(f'num_moves: {num_moves}, max_depth: {max_depth}')
    logger.info(f'alpha-beta agent took time {time.time() - start_time} with depth {max_depth}')
    
    return action


def get_last_move(prev_board, curr_board):
    for i in range(constants.BOARD_SIZE):
        for j in range(constants.BOARD_SIZE):
            if prev_board[i, j] != curr_board[i, j]:
                return (i * constants.BOARD_SIZE + j)

    return -1


if __name__ == '__main__':
    curr_player = int(input())

    prev_board = get_board(curr_player)
    prev_state = to_state(prev_board)

    if prev_state == 0:
        prev_state = -1

    curr_board = get_board(curr_player)
    curr_state = to_state(curr_board)

    logger.debug(f'{curr_state}: curr_state')

    if prev_state == -1:
        put_num_moves(0 if curr_state == 0 else 1)

    curr_state |= (constants.BLACK << constants.PLAYER_POS)

    num_moves = get_num_moves()

    first_player = constants.WHITE if curr_player == constants.WHITE else constants.BLACK

    game = Game(constants.BOARD_SIZE, game_state=curr_state,
                prev_board_state=prev_state, num_moves=num_moves, first_player=first_player)

    action = get_next_move_from_ab_agent(game)

    logger.debug(f'action: {action}')
    logger.debug(f'board: {game.board}')

    put_num_moves(num_moves + 2)

    if action == -1:
        print("PASS")
    else:
        print(action // constants.BOARD_SIZE,
              action % constants.BOARD_SIZE, sep=',')
