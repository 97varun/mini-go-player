import numpy as np
import constants
from functools import reduce
from copy import deepcopy

import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.FileHandler('game.log'))
logger.setLevel(logging.INFO)
logger.propagate = False


class Game:
    def __init__(self, N: int, 
        game_state: int = 0, 
        prev_board_state: int = -1,
        num_moves: int = 0,
        first_player: int = constants.BLACK):
        
        self.size = N

        self.parent_board_state = prev_board_state

        self.board_state = game_state & ~(
            constants.MASK << constants.PLAYER_POS)
        self.board = self.from_state(self.board_state)
        self.captured_stones = []

        self.curr_player = (game_state >> constants.PLAYER_POS)
        if self.curr_player == 0:
            self.curr_player = constants.BLACK

        self.first_player = first_player

        self.komi = N / 2
        self.num_moves = num_moves
        self.game_over = False

        self.num_captured_stones = {constants.BLACK: 0, constants.WHITE: 0}
        self.cell_score = {constants.BLACK: [0, 0], constants.WHITE: [0, 0]}
        self.liberty_score = {constants.BLACK: [0, 0], constants.WHITE: [0, 0]}
        self.component_score = {constants.BLACK: [0, 0], constants.WHITE: [0, 0]}

    def place_stone(self, x: int, y: int, stone: int) -> None:
        self.board[x][y] = stone

        self.captured_stones = []
        self.find_captured_stones(stone)

        self.num_captured_stones[stone] = len(self.captured_stones)

        logger.debug(f'self.captured_stones: {self.captured_stones}')

        self.remove_captured_stones()

        self.parent_board_state = self.board_state
        self.board_state = self.to_state()

    def find_captured_stones(self, played_stone) -> None:
        self.liberty_score = {constants.BLACK: [0, 0], constants.WHITE: [0, 0]}

        for cord, stone in np.ndenumerate(self.board):
            if self.board[cord] != constants.EMPTY:
                logger.debug(f'{cord}')
                self.has_liberty(*cord, stone)
                logger.debug(f'{cord}')

            if not self.liberty[cord]:
                self.captured_stones.append(cord)

        logger.debug(f'self.liberty: {self.liberty}')

        self.captured_stones = list(filter(
            lambda cord: self.board[cord] == constants.OTHER_STONE[played_stone], self.captured_stones))

    def remove_captured_stones(self) -> None:
        for stone in self.captured_stones:
            self.board[stone] = constants.EMPTY

    def legal_placement(self, x: int, y: int, stone: int) -> bool:
        if self.board[x, y] != constants.EMPTY:
            return False

        # has liberty initially
        self.board[x, y] = stone
        if self.has_liberty(x, y, stone):
            self.board[x, y] = constants.EMPTY
            return True
        self.board[x, y] = constants.EMPTY

        # place stone in copy of board and check liberty and ko rule
        board_copy = deepcopy(self)
        board_copy.place_stone(x, y, stone)
        logger.debug(f'board_copy: {board_copy}')
        if board_copy.has_liberty(x, y, stone) and board_copy.board_state != self.parent_board_state:
            return True

        return False

    def has_liberty(self, x: int, y: int, stone: int):
        self.liberty = np.zeros((self.size, self.size), dtype=bool)
        self.visited = np.zeros((self.size, self.size), dtype=bool)

        return self.find_liberty(x, y)

    def find_liberty(self, x: int, y: int) -> int:
        curr_stone = self.board[x, y]

        if curr_stone == constants.EMPTY:
            return True

        if self.visited[x, y]:
            return self.liberty[x, y]

        self.visited[x, y] = True

        neighbors = list(self.get_friend_neighbors(x, y, curr_stone))

        neighbors_liberty = list(map(
            lambda neighbor: self.find_liberty(*neighbor), neighbors))

        self.liberty_score[curr_stone][0] += sum(neighbors_liberty)
        self.liberty_score[curr_stone][1] += len(neighbors_liberty)

        self.liberty[x, y] = reduce(
            lambda x, y: (x or y), neighbors_liberty, False)

        logger.debug(
            f'x: {x}, y: {y}, neighbors: {neighbors}, liberty: {self.liberty[x, y]}')

        return self.liberty[x, y]

    def get_friend_neighbors(self, x: int, y: int, stone: int):
        neighbors = []
        if x > 0:
            neighbors.append((x - 1, y))
        if y > 0:
            neighbors.append((x, y - 1))
        if x < self.size - 1:
            neighbors.append((x + 1, y))
        if y < self.size - 1:
            neighbors.append((x, y + 1))

        neighbors = filter(
            lambda neighbor: self.board[neighbor] != constants.OTHER_STONE[stone], neighbors)

        return neighbors

    def get_game_state(self):
        return self.board_state | (self.curr_player << constants.PLAYER_POS)

    def get_board_state(self):
        return self.board_state

    def to_state(self) -> int:
        state = 0

        for cord, stone in np.ndenumerate(self.board):
            flat_index = cord[0] * self.size + cord[1]
            flat_index <<= 1
            state |= (int(stone) << flat_index)

        return state

    def from_state(self, state: int):
        board = []

        for cell in range(0, 2 * (self.size ** 2), 2):
            board.append((state & (constants.MASK << cell)) >> cell)

        return np.reshape(board, (self.size, self.size)).astype(np.int64)

    def __str__(self) -> str:
        line = '-----\n'
        rep = line

        for i in range(self.size):
            for j in range(self.size):
                rep += constants.CELL_TO_REP[self.board[i][j]]
            rep += '\n'

        rep += line

        return rep

    def get_num_stones(self, stone: int) -> int:
        return len(np.where(self.board == stone)[0])

    def move(self, a: int) -> None:
        # Pass
        if a == -1:
            # Two consecutive passes
            logger.debug(
                f'self.board_state: {self.board_state}, self.parent_board_state: {self.parent_board_state}')
            if self.board_state == self.parent_board_state:
                self.game_over = True

            self.parent_board_state = self.board_state
        else:
            x, y = a // self.size, a % self.size
            if self.legal_placement(x, y, self.curr_player):
                self.place_stone(x, y, self.curr_player)

        self.curr_player = constants.OTHER_STONE[self.curr_player]
        self.num_moves += 1

        # max moves
        if self.num_moves == constants.MAX_MOVES:
            self.game_over = True

    def get_reward(self, curr_player):
        if self.game_over:
            players = [constants.BLACK, constants.WHITE]
            score = {player: self.get_num_stones(player) for player in players}

            score[constants.OTHER_STONE[self.first_player]] += self.komi
            winner = np.argmax(
                [score[constants.BLACK], score[constants.WHITE]]) + 1
            return constants.WIN_REWARD if curr_player == winner else constants.LOSS_REWARD

        return 0

    def get_score(self, curr_player):
        players = [constants.BLACK, constants.WHITE]
        num_stones = {player: self.get_num_stones(player) for player in players}
        score = deepcopy(num_stones)
        
        score[constants.OTHER_STONE[self.first_player]] += self.komi

        print(self)
        print(score, self.game_over)

        if self.game_over:
            winner = np.argmax(
                [score[constants.BLACK], score[constants.WHITE]]) + 1
            return constants.WIN_SCORE if curr_player == winner else constants.LOSS_SCORE

        self.calculate_connected_component_score()
        self.calculate_cell_score()
        
        for player in players:
            logger.debug(f'num_moves: {self.num_moves}, score[player]: {score[player]}')
            
            score[player] /= (self.num_moves // 2 + (0 if player == self.first_player else self.komi))
            score[player] *= 2

            if num_stones[player] == 0:
                num_stones[player] = 1
            
            logger.debug(f'cell_score: {self.cell_score[player][0]} / {constants.CELL_SCORE_PREFIX_SUM[num_stones[player]]}')
            logger.debug(f'cell_score: {self.cell_score[player][0] / constants.CELL_SCORE_PREFIX_SUM[num_stones[player]]}')
            
            score[player] += 0.25 * self.cell_score[player][0] / constants.CELL_SCORE_PREFIX_SUM[num_stones[player]]
            
            if self.liberty_score[player][1] == 0:
                self.liberty_score[player][1] = 1

            logger.debug(f'lib_score: {self.liberty_score[player][0] / (self.liberty_score[player][1])}')

            score[player] += 0.5 * self.liberty_score[player][0] / self.liberty_score[player][1]
            
            logger.debug(f'comp_score: {(self.component_score[player][0] / (self.component_score[player][1])) / (num_stones[player])}')
            
            score[player] += (self.component_score[player][0] / (self.component_score[player][1])) / (num_stones[player])
            
            # score[player] += 2 * self.num_captured_stones[player] / (self.num_captured_stones[player] + num_stones[constants.OTHER_STONE[player]])

        score_diff = score[constants.BLACK] - score[constants.WHITE]

        return score_diff if curr_player == constants.BLACK else -score_diff

    def calculate_connected_component_score(self):
        self.visited = np.zeros((self.size, self.size), dtype=bool)
        
        self.component_score = {constants.BLACK: [0, 0], constants.WHITE: [0, 0]}

        for i in range(self.size):
            for j in range(self.size):
                if self.visited[i, j] or (self.board[i, j] == constants.EMPTY):
                    continue

                stone = self.board[i, j]
                component_size = self.find_connected_components(i, j, stone)

                
                self.component_score[stone][0] += component_size
                self.component_score[stone][1] += 1

        for player in [constants.BLACK, constants.WHITE]:
            if self.component_score[player][1] == 0:
                self.component_score[player][1] = 1

    def find_connected_components(self, x: int, y: int, stone: int) -> int:
        if self.board[x, y] != stone:
            return 0

        if self.visited[x, y]:
            return 0

        self.visited[x, y] = True

        num_stone = 1

        neighbors = list(self.get_friend_neighbors(x, y, stone))
        for neighbor in neighbors:
            num_stone += self.find_connected_components(*neighbor, stone)

        return num_stone

    def calculate_cell_score(self):
        self.cell_score = {constants.BLACK: [0, 0], constants.WHITE: [0, 0]}

        edges = [0, 4]

        for x in range(self.size):
            for y in range(self.size):
                stone = self.board[x, y]
                if stone == constants.EMPTY:
                    continue

                self.cell_score[stone][0] += constants.CELL_SCORES[x][y]
                self.cell_score[stone][1] += 1

    def get_cell_score(self, x: int, y: int) -> int:
        edges = [0, 4]

        if x == 2 and y == 2:
            return 3
        elif x in edges or y in edges:
            return 1
        else:
            return 2

    def get_possible_moves(self):
        actions = np.array(
            [12, 6, 7, 8, 11, 13, 16, 17, 18, 0, 1, 2, 3,
                4, 5, 9, 10, 14, 15, 19, 20, 21, 22, 23, 24]
        )
        legal_actions = np.fromiter(
            filter(lambda action: self.legal_placement(action // self.size, action % self.size, self.curr_player), actions), dtype=int)
        legal_actions = np.append(legal_actions, -1)
        return legal_actions


def test_game_to_state():
    game = Game(constants.BOARD_SIZE)

    game.move(1)
    game.move(0)

    assert game.get_game_state() == (6 | (constants.BLACK << constants.PLAYER_POS))


def get_game_from_moves(moves) -> Game:
    game = Game(constants.BOARD_SIZE)

    for mov in moves:
        game.move(constants.BOARD_SIZE * mov[0] + mov[1])

    return game


def test_has_liberty():
    game1 = Game(N=constants.BOARD_SIZE, game_state=44179170368 | (
        constants.WHITE << constants.PLAYER_POS))
    assert game1.legal_placement(2, 0, constants.WHITE)

    game1.move(10)
    game2 = Game(N=5, game_state=44160296000)
    assert game1.board_state == game2.board_state


def test_from_state():
    assert Game(N=constants.BOARD_SIZE,
                game_state=33555969).board_state == 33555969


def test_liberty():
    state = 99327865629014
    g = Game(N=constants.BOARD_SIZE, game_state=state |
             (constants.WHITE << constants.PLAYER_POS))
    assert not g.legal_placement(4, 4, 2)

    state = 591711615346005
    g = Game(N=constants.BOARD_SIZE, game_state=state |
             (constants.BLACK << constants.PLAYER_POS))
    assert not g.legal_placement(4, 3, 1)


def test_ko_rule():
    moves = [[1, 2], [1, 3], [2, 1], [2, 4], [2, 3], [3, 3], [3, 2]]
    game = get_game_from_moves(moves)

    legal = game.legal_placement(2, 2, constants.WHITE)
    game.move(12)
    assert legal

    legal = game.legal_placement(2, 3, constants.BLACK)
    assert not legal


def test_game_over():
    game = Game(constants.BOARD_SIZE)

    game.move(-1)
    game.move(-1)

    assert game.game_over


def test_reward():
    game = Game(5)
    game.move(-1)
    game.move(-1)

    assert game.game_over

    assert game.get_reward(constants.WHITE) == constants.WIN_REWARD
    assert game.get_reward(constants.BLACK) == constants.LOSS_REWARD

    game = Game(5)
    game.move(5)
    game.move(0)
    game.move(1)

    assert game.get_reward(constants.WHITE) == 0.5

    assert game.get_reward(constants.BLACK) == -0.5


def test_get_possible_moves():
    moves = [[1, 2], [1, 3], [2, 1], [2, 4], [2, 3], [3, 3], [3, 2]]
    game = get_game_from_moves(moves)

    expected_possible_moves = np.array(
        [-1, 0, 1, 2, 3, 4, 5, 6, 9, 10, 12, 15, 16, 19, 20, 21, 22, 23, 24])

    assert np.array_equal(game.get_possible_moves(), expected_possible_moves)


if __name__ == "__main__":
    test_game_to_state()
    test_has_liberty()
    test_from_state()
    test_liberty()
    test_ko_rule()
    test_game_over()
    test_reward()
    test_get_possible_moves()
