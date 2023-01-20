import enum
import numpy as np
import constants
from game import Game
import pickle
from copy import deepcopy
import os
import random
import time

from math import floor, ceil

from qlearningagent import RLAgent

import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.FileHandler('agent.log'))
logger.setLevel(logging.INFO)
logger.propagate = False


class AlphaBetaAgent:
    def __init__(self, max_depth=constants.MAX_DEPTH):
        self.pos_inf = int(1e9)
        self.neg_inf = -self.pos_inf
        self.max_depth = max_depth
        self.values = {}

    def load_values(self):
        if os.path.exists(constants.AB_VALUES_FILENAME):
            with open(constants.AB_VALUES_FILENAME, 'rb') as fp:
                self.values = pickle.load(fp)

    def save_values(self):
        with open(constants.AB_VALUES_FILENAME, 'wb') as fp:
            pickle.dump(self.values, fp)

    def search(self, game: Game):
        if game.num_moves == 0 or game.num_moves == 1:
            if game.board[2, 2] == constants.EMPTY:
                return 12
            elif game.board[1, 1] == constants.EMPTY:
                return 7

        self.player = game.curr_player

        _, action = self.get_max_value(game, self.neg_inf, self.pos_inf, 0)

        return action

    def get_next_state(self, game: Game, action: int):
        new_game = deepcopy(game)
        new_game.move(action)
        return new_game

    def get_max_value(self, game: Game, alpha: int, beta: int, depth: int):
        curr_state = game.get_game_state()

        if curr_state in self.values:
            return self.values[curr_state]

        logger.info(
            f'max_value: state: {curr_state}, alpha: {alpha}, beta: {beta}')

        if game.game_over or depth == self.max_depth:
            logger.info('max_value: game over!')
            logger.info(f'state: {game}\n get_max_value {game.get_score(self.player)}')

            return (game.get_score(self.player), constants.NO_ACTION)

        possible_next_actions = game.get_possible_moves()

        logger.debug(f'possible_next_actions: {possible_next_actions}')

        possible_next_actions = list(self.prune(possible_next_actions, game))

        logger.info(f'state: {game}\n possible_next_actions(pruned): {possible_next_actions}')

        best_action = -1

        for action in possible_next_actions:
            next_state = self.get_next_state(game, action)

            min_value = self.get_min_value(
                next_state, alpha, beta, depth + 1)[0]

            logger.info(f'state: {game}\n  action: {action}, min_value: {min_value}')
            
            if min_value > alpha:
                alpha = min_value
                best_action = action

            if alpha >= beta:
                self.values[curr_state] = (beta, best_action)
                return (beta, best_action)

        logger.info(f'state: {game}\n get_max_value: best_action: {best_action}')

        self.values[curr_state] = (alpha, best_action)
        return (alpha, best_action)

    def get_min_value(self, game: Game, alpha: int, beta: int, depth: int):
        curr_state = game.get_game_state()

        if curr_state in self.values:
            return self.values[curr_state]

        logger.info(
            f'min_value: state: {curr_state}, alpha: {alpha}, beta: {beta}')

        if game.game_over or depth == self.max_depth:
            logger.info('min_value: game over!')
            logger.info(f'state: {game}\n get_min_value, {game.get_score(self.player)}')

            return (game.get_score(self.player), constants.NO_ACTION)

        possible_next_actions = game.get_possible_moves()

        logger.debug(f'possible_next_actions: {possible_next_actions}')

        possible_next_actions = list(self.prune(possible_next_actions, game))

        logger.info(f'state: {game}\n possible_next_actions(pruned): {possible_next_actions}')

        best_action = -1

        for action in possible_next_actions:
            next_state = self.get_next_state(game, action)

            max_value = self.get_max_value(
                next_state, alpha, beta, depth + 1)[0]

            logger.info(f'state: {game}\n action: {action}, max_value: {max_value}')
            
            if max_value < beta:
                beta = max_value
                best_action = action

            if beta <= alpha:
                self.values[curr_state] = (alpha, best_action)
                return (alpha, best_action)

        logger.info(f'state: {game}\n get_min_value: best_action: {best_action}')

        self.values[curr_state] = (beta, best_action)
        return (beta, best_action)

    def prune(self, possible_actions: np.array, game: Game) -> np.array:
        def non_isolated(action) -> bool:
            if action == -1:
                return True

            x, y = action // constants.BOARD_SIZE, action % constants.BOARD_SIZE

            return any(map(
                lambda neigbor: game.board[neigbor] != constants.EMPTY, self.get_neighbors(x, y)))

        possible_actions = filter(non_isolated, possible_actions)

        return np.fromiter(possible_actions, dtype=np.int64)

    def get_neighbors(self, x: int, y: int) -> list:
        neighbors = []
        
        if x > 0:
            neighbors.append((x - 1, y))
        if x < constants.BOARD_SIZE - 1:
            neighbors.append((x + 1, y))
        if y > 0:
            neighbors.append((x, y - 1))
        if y < constants.BOARD_SIZE - 1:
            neighbors.append((x, y + 1))

        return neighbors

    def manhattan_distance(self, x: int, y: int, p: int, q: int) -> int:
        return abs(x - p) + abs(y - q)



def test_flip_colors():
    state = 141750845600

    game = Game(constants.BOARD_SIZE, game_state=state)

    print(game)


def test_minimax_play_game():
    agent = AlphaBetaAgent()
    game = Game(constants.BOARD_SIZE)

    while not game.game_over:
        a = agent.search(game)
        logger.info(a)
        game.move(a)
        logger.info(game.board)


def test_deep_rl_model_creation():
    agent = RLAgent()
    agent.learn()


if __name__ == "__main__":
    test_deep_rl_model_creation()
    # train_rl_agent()
    # test_flip_colors()
