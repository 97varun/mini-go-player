import numpy as np
import constants
from board import *
import sys

sys.stdout = open("output.txt", "w")


class Game:
    def __init__(self, N):
        self.state = 0
        self.parent_state = 0
        self.curr_player = constants.BLACK
        self.board = Board(N)
        self.komi = N / 2
        self.num_moves = 0
        self.size = N
        self.game_over = False

    def move(self, a: int) -> bool:
        # pass
        if a < 0:
            return self.move2(None, None)

        return self.move2(a // self.size, a % self.size)

    def move2(self, x: int, y: int) -> bool:
        # two consecutive pass
        if (x == None or y == None) and self.state == self.parent_state:
            self.game_over = True
        
        if not self.legal_move(x, y):
            success = False
        else:
            success = True

            if not(x == None or y == None):
                self.board.place_stone(x, y, self.curr_player)

        self.parent_state = self.state
        self.state = self.board.to_state()

        self.curr_player = constants.OTHER_STONE[self.curr_player]
        self.num_moves += 1

        # max moves
        if self.num_moves == self.size ** 2 - 1:
            self.game_over = True

        return success

    def legal_move(self, a: int) -> bool:
        if a < 0:
            return self.legal_move2(None, None)
        
        return self.legal_move2(a // self.size, a % self.size)


    def legal_move2(self, x: int, y: int):
        # pass
        if x == None or y == None:
            return True

        liberty = self.check_liberty_rule(x, y)

        self.board.place_stone(x, y, self.curr_player)

        ko = self.check_ko_rule(x, y)

        self.board.remove_last_stone()

        return liberty and not ko

    def check_liberty_rule(self, x: int, y: int) -> bool:
        return self.board.has_liberty(x, y, self.curr_player)

    def check_ko_rule(self, x: int, y: int) -> bool:
        return self.board.to_state() == self.parent_state

    def get_reward(self, player):
        black_score = self.board.get_num_stones(constants.BLACK)
        white_score = self.board.get_num_stones(constants.WHITE) + self.komi

        if self.game_over:  
            winner = np.argmax([black_score, white_score]) + 1

            return constants.WIN_REWARD if player == winner else constants.LOSS_REWARD

        score_diff = black_score - white_score
        
        return score_diff if player == constants.BLACK else -score_diff

        

def test_ko_rule():
    black_stones = [[1, 2], [2, 1], [2, 3], [3, 2]]
    white_stones = [[1, 3], [2, 4], [3, 3]]

    board1 = get_board_with_pieces(black_stones, white_stones)

    game = Game(5)
    game.board = board1
    game.state = board1.to_state()
    game.curr_player = constants.WHITE

    legal = game.move(2, 2)

    assert legal

    legal = game.move(2, 3)

    assert not legal


def test_game_over():
    game = Game(5)

    game.move(None, None)

    game.move(None, None)

    assert game.game_over

def test_reward():
    game = Game(5)
    game.move(None, None)
    game.move(None, None)

    assert game.game_over

    assert game.get_reward(constants.WHITE) == constants.WIN_REWARD
    assert game.get_reward(constants.BLACK) == constants.LOSS_REWARD

    game = Game(5)
    game.move(1, 0)
    game.move(0, 0)
    game.move(0, 1)

    assert game.get_reward(constants.WHITE) == 0.5

    assert game.get_reward(constants.BLACK) == -0.5

if __name__ == "__main__":
    test_ko_rule()
    test_game_over()
    test_reward()
