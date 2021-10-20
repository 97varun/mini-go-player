import json
import pickle

from game import Game

state = 2147483648

g = Game(N=5, game_state=state)

print(g)

print(f'player: {state >> 50}')


# with open('qtable.json', 'r') as fp:
#     q = json.load(fp)

# with open('qpickle.pkl', 'wb') as fp:
#     pickle.dump(q, fp)


