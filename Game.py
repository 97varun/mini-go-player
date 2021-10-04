import numpy as np
import Constants
import Board

class Game:
    def __init__(self):
        self.state = 0
        self.parent_state = 0
        self.curr_player = Constants.BLACK
        self.board = Board()
    
    def move(self, x: int, y: int) -> True:
        if self.legal_move(x, y):
            self.board.place_stone(x, y, self.curr_player)

        self.parent_state = self.board.to_state()

    def legal_move(self, x: int, y: int):
        # pass
        if x == None or y == None:
            return True
        
        self.board.place_stone(x, y, self.curr_player)

        liberty = self.check_liberty_rule(x, y)

        ko = self.check_ko_rule(x, y)

        self.board.remove_last_stone()

        return liberty and not ko

    def check_liberty_rule(self, x: int, y: int) -> bool:
        return self.board.has_liberty()

    def check_ko_rule(self, x: int, y: int) -> bool:
        return self.board.to_state() == self.parent_state

    def state_to_board():
        pass