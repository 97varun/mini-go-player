import pickle
import constants

from game import Game

def examine_state():
    state = 2147483648

    g = Game(N=5, game_state=state)

    print(f'player: {state >> 50}')
    print(g)

def get_q_values_for_state(state):
    with open(constants.Q_TABLE_FILENAME, 'rb') as fp:
        q = pickle.load(fp)

    if state in q:
        print(q[state])
    else:
        print('not found')

get_q_values_for_state(0)

# with open('qtable.json', 'r') as fp:
#     q = json.load(fp)

# with open('qpickle.pkl', 'wb') as fp:
#     pickle.dump(q, fp)


