import numpy as np
import Constants


class Board:
    def __init__(self, N):
        self.size = N
        self.board = np.full((N, N), fill_value=Constants.EMPTY, dtype=int)
        self.curr_player = 0

    def place_stone(self, x: int, y: int, stone: int) -> None:
        self.board[x][y] = stone
        self.last_move = (x, y)

    def remove_stone(self, x: int, y: int) -> None:
        self.board[x][y] = Constants.EMPTY

    def remove_last_stone(self) -> bool:
        if self.last_move is None:
            return False

        self.board[self.last_move[0]][self.last_move[1]] = Constants.EMPTY
        
        return True

    def valid(self, cord: int) -> bool:
        return cord >= 0 and cord <= self.size - 1

    def has_liberty(self, x: int, y: int, stone: int):
        self.visited = np.zeros((self.size, self.size), dtype=bool)

        for cord, stone in np.ndenumerate(self.board):
            if not self.visited[cord] and not stone == Constants.EMPTY:
                self.has_liberty_inner(*cord, stone)

        return self.has_liberty_inner(x, y, stone)

    def has_liberty_inner(self, x: int, y: int, stone: int) -> int:
        if not (self.valid(x) and self.valid(y)):
            return False
        
        if self.board[x][y] == Constants.EMPTY:
            return True

        if self.board[x][y] != stone:
            return False

        left_lib = self.has_liberty_inner(x - 1, y, stone)
        top_lib = self.has_liberty_inner(x, y - 1, stone)
        right_lib = self.has_liberty_inner(x + 1, y, stone)
        bottom_lib = self.has_liberty_inner(x, y + 1, stone)

        lib = left_lib or top_lib or right_lib or bottom_lib

        if not lib:
            self.remove_stone(x, y)

        return 

    def to_state(self) -> int:
        state: int = 0

        for cord, stone in np.ndenumerate(self.board):
            flat_index = cord[0] * self.size + cord[1]

            flat_index <<= 1

            state |= stone << flat_index

        return state

    def __str__(self):
        rep = ''

        for i in range(self.size):
            for j in range(self.size):
                rep += Constants.CELL_TO_REP[self.board[i][j]]
            rep += '\n'

        return rep        

def test_board_to_state():
    board = Board(5)

    board.place_stone(0, 0, Constants.WHITE)
    board.place_stone(0, 1, Constants.BLACK)

    assert board.to_state() == 6


def test_has_liberty():
    board = Board(5)

    black_stones = [[0, 3], [1, 0], [1, 3], [3, 4]]
    white_stones = [[1, 1], [1, 2], [1, 4], [2, 0], [2, 3], [3, 1], [3, 2]]

    for stone in black_stones:
        board.place_stone(*stone)
    
    for stone in white_stones:
        board.place_stone(*stone)


if __name__ == "__main__":
    test_board_to_state()
    test_has_liberty()
    
