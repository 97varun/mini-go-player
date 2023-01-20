class NNRLAgent():
    def __init__(self, epsilon=1.0, game=None):
        self.game = game

        self.alpha = 0.7
        self.epsilon = epsilon
        self.epsilon_decay = 0.9996

        self.create_model()

        self.history = []

        self.game_helper = Game(constants.BOARD_SIZE)

        self.train_set_size = 4
        self.training_gap = 4
        self.num_episodes = 10
    
    def create_model(self):
        self.model = Sequential()
        # self.model.add(Conv2D(25, kernel_size=(3, 3), activation='relu', input_shape=(constants.BOARD_SIZE, constants.BOARD_SIZE, 1)))
        # self.model.add(Flatten())

        self.model.add(Dense(units=512, activation='relu', input_dim=constants.BOARD_SIZE * constants.BOARD_SIZE))
        self.model.add(Dense(units=512, activation='relu'))
        self.model.add(Dense(units=512, activation='relu'))
        self.model.add(Dense(units=512, activation='relu'))
        self.model.add(Dense(units=26, activation='linear'))
        self.model.compile(loss='mse', optimizer='adam')

        self.model.summary()

    def play(self, first_player: int):
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

            possible_actions = self.game.get_possible_moves()

            if curr_action in possible_actions:
                # Take action and observe reward and next_state
                self.game.move(curr_action)
                reward = self.game.get_reward(player)
                next_state = self.game.get_board_state()

                # select next action a' epsilon-greedy based on q(s, a)
                next_action = self.get_next_action()
            else:
                # If illegal move then game is over and reward is -1
                next_state = curr_state if player == constants.BLACK else self.flip_colors(curr_state)
                next_action = curr_action
                reward = -1

                self.game.game_over = True

            logger.debug(
                f'next_state: {next_state}, next_action: {next_action}')

            if player == constants.WHITE:
                last_observation = (self.flip_colors(curr_state), curr_action,
                                    next_state, next_action, reward, self.game.game_over)
            else:
                last_observation = (curr_state, curr_action,
                                    self.flip_colors(next_state), next_action, reward, self.game.game_over)

            self.history.append(last_observation)

            curr_state = next_state
            curr_action = next_action


    def learn(self):
        curr_episode = 1
        first_players = [constants.WHITE,
                            constants.BLACK]

        while curr_episode <= self.num_episodes:
            self.play(first_players[curr_episode % 2])

            logger.info(
                f'curr_episode: {curr_episode}, self.epsilon: {self.epsilon}')

            self.epsilon *= self.epsilon_decay
            curr_episode += 1

            if len(self.history) > self.train_set_size and curr_episode % self.training_gap == 0:
                print(curr_episode)
                self.update_q_values()

                self.play_game_against_ab()

        print(len(self.history))

    def update_q_values(self):
        train_set_x, train_set_y = self.get_train_set()

        self.model.fit(train_set_x, train_set_y, epochs=1)

    def get_next_action(self) -> int:
        curr_state = self.game.get_board_state()

        possible_actions = np.arange(-1, constants.BOARD_SIZE ** 2, 1)

        logger.debug(f'possible_actions: {possible_actions}')

        if np.random.random() < self.epsilon:
            rand_idx = np.random.randint(0, len(possible_actions))
            return int(possible_actions[rand_idx])

        if self.game.curr_player == constants.WHITE:
            curr_state = self.flip_colors(curr_state)

        curr_state2d = self.game_helper.from_state(curr_state)
        q_curr_state = self.model.predict(np.reshape(curr_state2d, (1, constants.BOARD_SIZE * constants.BOARD_SIZE)))[0]

        return int(np.argmax(q_curr_state) - 1)

    def flip_colors(self, state: int) -> int:
        flipped_state = 0

        for cell in range(0, 2 * (constants.BOARD_SIZE ** 2), 2):
            stone = (state & (constants.MASK << cell)) >> cell

            if stone != constants.EMPTY:
                flipped_state |= (constants.OTHER_STONE[stone] << cell)

        return flipped_state

    def get_train_set(self, train_set_size=4):
        start_time = time.time()

        train_set_x, train_set_y = [], []

        random_sample = random.sample(self.history, train_set_size)

        curr_states2d = list(map(lambda x: 
            self.game_helper.from_state(x[0]), random_sample))
        next_states2d = list(map(lambda x: 
            self.game_helper.from_state(x[2]), random_sample))

        curr_state_predictions = self.model.predict(np.reshape(curr_states2d, (train_set_size, constants.BOARD_SIZE * constants.BOARD_SIZE)))
        next_state_predictions = self.model.predict(np.reshape(next_states2d, (train_set_size, constants.BOARD_SIZE * constants.BOARD_SIZE)))

        logger.info(f'curr_state_predictions.shape: {curr_state_predictions.shape}')
        logger.info(f'next_state_predictions.shape: {next_state_predictions.shape}')

        for idx, (curr_state, curr_action, next_state, next_action, reward, game_over) in enumerate(random_sample):
            train_set_x.append(curr_states2d[idx])

            q_next_state_next_action = next_state_predictions[idx][next_action]
            y = curr_state_predictions[idx]
            y[curr_action] = reward if game_over else reward + q_next_state_next_action

            train_set_y.append(y)


        train_set_x = np.reshape(train_set_x, (train_set_size, constants.BOARD_SIZE * constants.BOARD_SIZE))
        train_set_y = np.array(train_set_y)

        logger.info(f'get_train_set: {time.time() - start_time}')

        logger.info(f'train_set_x.shape: {train_set_x.shape}, train_set_y.shape: {train_set_y.shape}')

        return train_set_x, train_set_y

    def get_next_best_legal_action(self, game: Game) -> int:
        legal_actions = game.get_possible_moves()

        curr_state2d = self.game_helper.from_state(game.get_board_state())
        actions = self.model.predict(np.reshape(curr_state2d, (1, constants.BOARD_SIZE * constants.BOARD_SIZE)))[0]

        actions = zip(actions, range(-1, constants.BOARD_SIZE ** 2, 1))
        actions = list(filter(lambda x: x[1] in legal_actions, actions))

        print(f'zipped_actions: {actions}')

        best_action = max(actions, key=lambda x: x[0])
        return best_action[1]

    def play_game_against_ab(self):
        ab_agent = AlphaBetaAgent(max_depth=4)

        for player in [constants.WHITE, constants.BLACK]:
            game = Game(constants.BOARD_SIZE, num_moves=0)
            
            print(f'rl: {player}, ab: {constants.OTHER_STONE[player]}')

            while not game.game_over:
                if game.curr_player == player:
                    game_copy = Game(
                        constants.BOARD_SIZE, 
                        game_state=self.flip_colors(game.get_board_state()), 
                        first_player=constants.WHITE,
                        num_moves=game.num_moves) if player == constants.WHITE else game

                    action = self.get_next_best_legal_action(game_copy)
                    print(f'rl played {action}')
                else:
                    game_copy = Game(
                        constants.BOARD_SIZE, 
                        game_state=self.flip_colors(game.get_board_state()), 
                        first_player=constants.WHITE,
                        num_moves=game.num_moves) if player == constants.BLACK else game

                    action = ab_agent.search(game_copy)
                    print(f'ab played {action}')
                
                game.move(action)
                print(game)

            print(f'game over: {game.get_reward(player)}')
