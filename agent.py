import numpy as np
import constants
from game import Game
import pickle
from copy import deepcopy
import os

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

    def search(self, game: Game):
        self.player = game.curr_player

        _, action = self.get_max_value(game, self.neg_inf, self.pos_inf, 0)

        return action

    def get_next_state(self, game: Game, action: int):
        new_game = deepcopy(game)
        new_game.move(action)
        return new_game

    def get_max_value(self, game: Game, alpha: int, beta: int, depth: int):
        logger.debug(
            f'max_value: state: {game.get_game_state()}, alpha: {alpha}, beta: {beta}')

        if game.game_over or depth == self.max_depth:
            logger.debug('max_value: game over!')

            return (game.get_reward(self.player), constants.NO_ACTION)

        possible_next_actions = game.get_possible_moves()

        best_action = -1

        for action in possible_next_actions:
            next_state = self.get_next_state(game, action)

            min_value = self.get_min_value(
                next_state, alpha, beta, depth + 1)[0]
            if min_value > alpha:
                alpha = min_value
                best_action = action

            if alpha >= beta:
                return (beta, best_action)

        return (alpha, best_action)

    def get_min_value(self, game: Game, alpha: int, beta: int, depth: int):
        logger.debug(
            f'min_value: state: {game.get_game_state()}, alpha: {alpha}, beta: {beta}')

        if game.game_over or depth == self.max_depth:
            logger.debug('min_value: game over!')

            return (game.get_reward(self.player), constants.NO_ACTION)

        possible_next_actions = game.get_possible_moves()

        best_action = -1

        for action in possible_next_actions:
            next_state = self.get_next_state(game, action)

            max_value = self.get_max_value(
                next_state, alpha, beta, depth + 1)[0]
            if max_value < beta:
                beta = max_value
                best_action = action

            if beta <= alpha:
                return (alpha, best_action)

        return (beta, best_action)


class RLAgent:
    def __init__(self, epsilon=1.0, game=None):
        # q = {(s: int, a: int): q(s, a): float}
        self.q = {}
        self.load_q_values()

        self.game = game

        self.alpha = 0.1
        self.epsilon_cutoff = 0.05
        self.epsilon = epsilon
        self.epsilon_decay = 0.9999

        self.last_game = []

    def play_episode(self, init_game_state: int):
        self.game = Game(constants.BOARD_SIZE, game_state=init_game_state)
        curr_state = self.game.get_board_state()

        # select next action epsilon-greedy based on q(s, a)
        curr_action = self.get_next_action()

        while not self.game.game_over:
            logger.debug(f'board:-\n {self.game.board}')

            logger.debug(
                f'curr_state: {curr_state}, curr_action: {curr_action}')

            player = self.game.curr_player

            # Take action and observe reward and next_state
            self.game.move(curr_action)
            reward = self.game.get_reward(player)
            next_state = self.game.get_board_state()

            # select next action a' epsilon-greedy based on q(s, a)
            next_action = self.get_next_action()

            logger.debug(
                f'next_state: {next_state}, next_action: {next_action}')
            
            self.last_game.append(
                (curr_state, curr_action, next_state, next_action, reward))

            curr_state = next_state
            curr_action = next_action

    def update_q_values(self):
        self.last_game.reverse()

        for curr_state, curr_action, next_state, next_action, reward in self.last_game:
            # Update q value -> q(s, a) = (1 - alpha) * q(s, a) + alpha * (r + q(s', a'))
            self.q[curr_state] = self.q.get(curr_state, {})
            self.q[next_state] = self.q.get(next_state, {})

            self.q[curr_state][curr_action] = (1 - self.alpha) * self.q[curr_state].get(curr_action, 0.0) + \
                self.alpha * \
                (reward + self.q[next_state].get(next_action, 0.0))

        self.last_game = []

    def get_next_action(self) -> int:
        curr_state = self.game.get_board_state()

        possible_actions = self.game.get_possible_moves()

        if np.random.random() < self.epsilon:
            return int(np.random.choice(possible_actions, 1)[0])

        logger.debug(f'possible_actions: {possible_actions}')

        possible_q_values = np.array(
            map(lambda action: self.q.get(curr_state, {}).get(action, 0.0), possible_actions))

        return int(possible_actions[np.argmax(possible_q_values)])

    def learn(self, num_episodes=1):
        self.load_epsilon()
        self.load_q_values()

        curr_episode = 1
        init_game_states = [constants.CURR_PLAYER_WHITE,
                            constants.CURR_PLAYER_BLACK]

        while curr_episode <= num_episodes and len(self.q) <= constants.MAX_QTABLE_SIZE:
            self.play_episode(init_game_states[curr_episode % 2])
            self.update_q_values()

            logger.info(
                f'curr_episode: {curr_episode}, len(self.q): {len(self.q)}, self.epsilon: {self.epsilon}')

            self.epsilon *= self.epsilon_decay
            curr_episode += 1

        self.save_epsilon()
        self.save_q_values()

    def save_q_values(self) -> None:
        with open(constants.Q_TABLE_FILENAME, 'wb') as fp:
            pickle.dump(self.q, fp)

    def load_q_values(self) -> None:
        if not os.path.exists(constants.Q_TABLE_FILENAME):
            with open(constants.Q_TABLE_FILENAME, 'wb') as fp:
                empty_dict = {}
                pickle.dump(empty_dict, fp)
        
        with open(constants.Q_TABLE_FILENAME, 'rb') as fp:
            self.q = pickle.load(fp)

    def save_epsilon(self) -> None:
        with open(constants.EPSILON_FILENAME, 'w') as fp:
            fp.write(str(self.epsilon))

    def load_epsilon(self) -> None:
        with open(constants.EPSILON_FILENAME, 'r') as fp:
            self.epsilon = float(fp.read())


def train_rl_agent():
    agent = RLAgent()

    import time
    start_time = time.time()

    agent.learn()

    print(time.time() - start_time)


def test_minimax_play_game():
    agent = AlphaBetaAgent()
    game = Game(constants.BOARD_SIZE)

    while not game.game_over:
        a = agent.search(game)
        logger.info(a)
        game.move(a)
        logger.info(game.board)


if __name__ == "__main__":
    train_rl_agent()
