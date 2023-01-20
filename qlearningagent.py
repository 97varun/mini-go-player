
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

import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.FileHandler('agent.log'))
logger.setLevel(logging.INFO)
logger.propagate = False

class RLAgent:
    def __init__(self, epsilon=1.0, game=None):
        self.q = {}
        self.load_q_values()

        self.game = game

        self.alpha = 0.7
        self.epsilon = epsilon
        self.epsilon_decay = 0.99985
        self.num_episodes = 30000

        self.last_game = []

    def play_episode(self, first_player: int):
        self.game = Game(constants.BOARD_SIZE, first_player=first_player)
        curr_state = self.game.get_board_state()

        logger.info(f'self.game.curr_player: {self.game.curr_player}')

        # select next action epsilon-greedy based on q(s, a)
        curr_action = self.get_next_action()

        while not self.game.game_over:
            logger.debug(f'board:-\n {self.game.board}')

            logger.debug(
                f'curr_state: {curr_state}, curr_action: {curr_action}')

            player = self.game.curr_player

            # Take action and observe reward and next_state
            self.game.move(curr_action)
            reward = self.game.get_score(player)
            next_state = self.game.get_board_state()

            # select next action a' epsilon-greedy based on q(s, a)
            next_action = self.get_next_action()

            logger.debug(
                f'next_state: {next_state}, next_action: {next_action}')

            if player == constants.WHITE:
                last_observation = (self.flip_colors(curr_state), curr_action,
                                    next_state, next_action, reward, self.game.game_over)
            else:
                last_observation = (curr_state, curr_action,
                                    self.flip_colors(next_state), next_action, reward, self.game.game_over)

            self.last_game.append(last_observation)

            curr_state = next_state
            curr_action = next_action

    def update_q_values(self):
        self.last_game.reverse()

        for curr_state, curr_action, next_state, next_action, reward, game_over in self.last_game:
            # Update q value -> q(s, a) = (1 - alpha) * q(s, a) + alpha * (r + q(s', a'))
            self.q[curr_state] = self.q.get(curr_state, {})
            self.q[next_state] = self.q.get(next_state, {})

            if game_over:
                self.q[curr_state][curr_action] = (1 - self.alpha) * self.q[curr_state].get(curr_action, 0.0) + \
                self.alpha * reward

            self.q[curr_state][curr_action] = (1 - self.alpha) * self.q[curr_state].get(curr_action, 0.0) + \
                self.alpha * \
                (reward + self.q[next_state].get(next_action, 0.0))

        self.last_game = []

    def get_next_action(self) -> int:
        curr_state = self.game.get_board_state()

        possible_actions = self.game.get_possible_moves()

        logger.debug(f'possible_actions: {possible_actions}')

        if np.random.random() < self.epsilon:
            rand_idx = np.random.randint(0, len(possible_actions))
            return int(possible_actions[rand_idx])

        if self.game.curr_player == constants.WHITE:
            curr_state = self.flip_colors(curr_state)

        possible_q_values = np.fromiter(
            map(lambda action: self.q.get(curr_state, {}).get(
                action, 0.0), possible_actions),
            dtype=float
        )

        return int(possible_actions[np.argmax(possible_q_values)])

    def learn(self):
        self.load_epsilon()
        self.load_q_values()

        curr_episode = 1
        init_game_states = [constants.WHITE, constants.BLACK]

        while curr_episode <= self.num_episodes and len(self.q) <= constants.MAX_QTABLE_SIZE:
            self.play_episode(init_game_states[curr_episode % 2])
            self.update_q_values()

            logger.info(
                f'curr_episode: {curr_episode}, len(self.q): {len(self.q)}, self.epsilon: {self.epsilon}')

            self.epsilon *= self.epsilon_decay
            curr_episode += 1

        self.save_epsilon()
        self.save_q_values()

    def flip_colors(self, state: int) -> int:
        flipped_state = 0

        for cell in range(0, 2 * (constants.BOARD_SIZE ** 2), 2):
            stone = (state & (constants.MASK << cell)) >> cell

            if stone != constants.EMPTY:
                flipped_state |= (constants.OTHER_STONE[stone] << cell)

        return flipped_state

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

