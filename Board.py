import numpy as np
import constants
import sys
from functools import reduce

import logging
logger = logging.getLogger(__name__)

logging.basicConfig(filename='board.log', level=logging.DEBUG)


class Board:
    def __init__(self, N: int, state: int=0):
        self.size = N

        self.state = state
        self.board = self.from_state(state)

        self.curr_player = 0
        self.captured_stones = []

    def place_stone(self, x: int, y: int, stone: int) -> None:
        self.board[x][y] = stone

        self.captured_stones = []

        self.find_captured_stones(x, y)

        self.captured_stones = list(filter(
            lambda cord: self.board[cord] == constants.OTHER_STONE[stone], self.captured_stones))

        self.remove_captured_stones()

        self.last_move = (x, y, stone)

    def remove_captured_stones(self) -> None:
        for stone in self.captured_stones:
            self.remove_stone(*stone)

    def find_captured_stones(self, x: int, y: int) -> None:
        self.liberty = np.zeros((self.size, self.size), dtype=bool)
        self.visited = np.zeros((self.size, self.size), dtype=bool)

        for cord, stone in np.ndenumerate(self.board):
            if not (self.visited[cord] or (self.board[cord] == constants.EMPTY)):
                self.liberty_dfs(*cord, stone)

        logger.debug(f'self.libery:-\n{self.liberty}')

    def remove_stone(self, x: int, y: int) -> None:
        self.board[x][y] = constants.EMPTY

    def remove_last_stone(self) -> bool:
        if self.last_move is None:
            return False

        self.board[self.last_move[0]][self.last_move[1]] = constants.EMPTY

        removed_stone = constants.OTHER_STONE[self.last_move[2]]

        for stone in self.captured_stones:
            self.board[stone[0]][stone[1]] = removed_stone

        return True

    def valid(self, cord: int) -> bool:
        return cord >= 0 and cord <= self.size - 1

    def valid2(self, x: int, y: int) -> bool:
        return self.valid(x) and self.valid(y)

    def has_liberty(self, x: int, y: int, stone: int):
        self.liberty = np.zeros((self.size, self.size), dtype=bool)
        self.visited = np.zeros((self.size, self.size), dtype=bool)

        self.board[x][y] = stone

        neighbors = [(x - 1, y), (x, y - 1), (x + 1, y), (x, y + 1)]

        neighbors = filter(lambda x: self.valid2(*x), neighbors)

        neighbors = map(lambda x: (
            self.board[x], self.liberty_dfs(*x, self.board[x])), neighbors)

        neighbors_liberty = map(
            lambda x: not x[1] if x[0] == constants.OTHER_STONE[stone] else x[1], neighbors)

        liberty = reduce(lambda x, y: (x or y), neighbors_liberty)

        self.board[x][y] = constants.EMPTY

        logger.debug(f'self.libery:-\n{self.liberty}')

        return liberty

    def liberty_dfs(self, x: int, y: int, stone: int) -> int:
        if not (self.valid(x) and self.valid(y)):
            return None

        curr_stone = self.board[x][y]

        if curr_stone == constants.EMPTY:
            return True

        if curr_stone != stone:
            return False

        if self.visited[x, y]:
            return self.liberty[x, y]

        self.visited[x, y] = True

        neighbors_liberty = [
            self.liberty_dfs(x - 1, y, stone),
            self.liberty_dfs(x, y - 1, stone),
            self.liberty_dfs(x + 1, y, stone),
            self.liberty_dfs(x, y + 1, stone),
        ]

        lib = reduce(lambda x, y: (x or y), neighbors_liberty)

        if not lib:
            self.captured_stones.append((x, y))

        self.liberty[x, y] = lib

        return lib

    def to_state(self) -> int:
        state = 0

        for cord, stone in np.ndenumerate(self.board):
            stone = int(stone)

            flat_index = cord[0] * self.size + cord[1]

            flat_index <<= 1

            state |= (stone << flat_index)

        return state

    def from_state(self, state: int):
        mask = 3
        board = []

        for cell in range(0, 2 * (self.size ** 2), 2):
            board.append((state & (mask << cell)) >> cell)

        return np.reshape(board, (self.size, self.size))

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

    def empty(self, x: int, y: int) -> bool:
        return self.board[x, y] == constants.EMPTY


def test_board_to_state():
    board = Board(5)

    board.place_stone(0, 0, constants.WHITE)
    board.place_stone(0, 1, constants.BLACK)

    assert board.to_state() == 6


def get_board_with_pieces(black_stones, white_stones) -> Board:
    board = Board(5)

    for stone in black_stones:
        board.place_stone(*stone, constants.BLACK)

    for stone in white_stones:
        board.place_stone(*stone, constants.WHITE)

    return board


def test_has_liberty():
    board1 = Board(N=5, state=44179170368)

    assert board1.has_liberty(2, 0, constants.WHITE)

    board1.place_stone(2, 0, constants.WHITE)

    board2 = Board(N=5, state=44160296000)

    assert board1.to_state() == board2.to_state()


def test_num_stones():
    black_stones = [[0, 3], [1, 0], [1, 3]]
    white_stones = [[1, 1], [1, 2], [1, 4], [2, 3], [3, 1], [3, 2]]

    board1 = get_board_with_pieces(black_stones, white_stones)

    assert board1.get_num_stones(constants.BLACK) == 3
    assert board1.get_num_stones(constants.WHITE) == 6


def test_from_state():
    assert Board(N=5, state=33555969).to_state() == 33555969


def test_liberty():
    state = 99327865629014
    b = Board(N=5, state=state)
    print(b.has_liberty(4, 4, 2))


if __name__ == "__main__":
    test_board_to_state()
    test_has_liberty()
    test_num_stones()
    test_from_state()
    test_liberty()
