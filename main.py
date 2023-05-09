import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, Flatten, InputLayer, MaxPooling2D
from keras.callbacks import TensorBoard
from keras.optimizers import Adam

from collections import deque
import random
import os
import time

from tqdm import tqdm

REPLAY_MEMORY_SIZE = 10_000
MIN_REPLAY_MEMORY_SIZE = 1_000
MODEL_NAME = 'V10-256x2'
MINIBATCH_SIZE = 64
DISCOUNT = 0.99
UPDATE_TARGET_EVERY = 5

MIN_REWARD = -60_000

EPISODES = 100_000

epsilon = 0
EPSILON_DECAY = 0.9950
MIN_EPSILON = 0.001

AGGREGATE_STATS_EVERY = 50
SHOW_PREVIEW = False


class DQNAgent:
    def __init__(self, weights=None):
        self.model = self.create_model()  # main model

        self.target_model = self.create_model()  # target model
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        self.tensorboard = ModifiedTensorBoard(log_dir=f'logs/{MODEL_NAME}-{int(time.time())}')

        self.target_update_counter = 0

        if weights:
            self.model.set_weights(weights)
            self.target_model.set_weights(weights)

    def create_model(self):
        """Creates both the target model and the main model"""
        model = Sequential()
        model.add(InputLayer(input_shape=(6, 7, 1)))  # Dimension of the 2d list with 1 for greyscale

        model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
        # model.add(MaxPooling2D(2, 2))
        # model.add(Dropout(0.2))

        model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D(2, 2))
        # model.add(Dropout(0.2))

        # model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
        # model.add(Dropout(0.2))

        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(7, activation='softmax'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
        return model

    def update_replay_memory(self, transition):
        """Updates replay memory"""
        self.replay_memory.append(transition)

    def get_qs(self, state):
        # print(BOARD)
        # print(self.model.predict(np.array(state).reshape(-1, *state.shape) / 255)[0])
        return self.model.predict(np.array(state).reshape(-1, *state.shape) / 255)[0]

    def train(self, terminal_state):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:  # Do not train if small sample size
            return
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)
        current_states = np.array([transition[0] for transition in minibatch]) / 255
        current_qs_list = self.model.predict(current_states)

        new_current_states = np.array([transition[3] for transition in minibatch]) / 255
        future_qs_list = self.target_model.predict(new_current_states)

        X, y = list(), list()

        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):
            if not done:  # episode incomplete
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            # if type(action) == np.ndarray:
            #     action = np.argmax(action)
            try:
                current_qs = current_qs_list[index]
                current_qs[action] = new_q
            except IndexError:
                print(f'Current State: {current_state}\nAction: {action}\nReward: {reward}\n'
                      f'N. Current State: {new_current_state}\nDone: {done}')
                continue


            X.append(current_state)
            y.append(current_qs)
        try:
            self.model.fit(np.array(X) / 255, np.array(y), batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False,
                           callbacks=[self.tensorboard] if terminal_state else None)
        except tf.errors.ResourceExhaustedError as e:
            print(f'Error is {e}')
            print(f'Relevant info:\n'
                  f'X - {X}\n'
                  f'y - {y}')

        # Updating to determine if it is time to update target model
        if terminal_state:
            self.target_update_counter += 1

        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0


class ModifiedTensorBoard(TensorBoard):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.create_file_writer(self.log_dir)
        self._log_write_dir = self.log_dir

    def set_model(self, model):
        self.model = model

        self._train_dir = os.path.join(self._log_write_dir, 'train')
        self._train_step = self.model._train_counter

        self._val_dir = os.path.join(self._log_write_dir, 'validation')
        self._val_step = self.model._test_counter

        self._should_write_train_graph = False

    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    def on_batch_end(self, batch, logs=None):
        pass

    def on_train_end(self, _):
        pass

    def update_stats(self, **stats):
        with self.writer.as_default():
            for key, value in stats.items():
                tf.summary.scalar(key, value, step=self.step)
                self.writer.flush()


class Board:
    def __init__(self):
        """Create the base board and turn counter"""
        self.board = [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ]
        self.size = 0
        self.valid_cols = {x for x in range(7)}

    def clear(self):
        self.board = [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ]
        self.size = 0
        self.valid_cols = {x for x in range(7)}

    def __str__(self):
        base = str()
        for row in self.board:
            base += f'{row} \n'
        return base

    def valid_move_check(self, x, y):
        """Returns if the current move is possible"""
        if x > 5 or y > 6:
            return False
        if x == 5:  # Bottom piece is always a valid move
            return True
        if self.board[x + 1][y] != 0:  # If there is a piece in the spot below, it is a valid move
            return True
        return False

    def get_valid_moves(self):
        return self.valid_cols

    def board_state(self):
        """Return a copy of the board"""
        return self.board.copy()

    def col_full(self, col):
        """Lets us know if a column is full"""
        try:
            if self.board[0][col] != 0:
                self.valid_cols.remove(col)
                return True
        except TypeError:
            print(f'New State: {new_state}\nReward: {reward}\nDone: {done}\nL. Block Lock: {last_block_loc}')
        return False

    def drop(self, col, player_id):
        """
        Drop the piece into the proper column
        -2 - Draw
        -1 - Illegal move
        1 - Winner Found
        2 - 3 in a row detected
        3 - Basic Move
        """
        block_loc = None
        if self.size == 42:  # Board full, draw
            return -2, (None, None), block_loc
        if self.col_full(col):  # If full, return -1
            return -1, (None, None), block_loc

        safe_row = -1
        for x in range(5, -1, -1):  # Finds the first spot open to drop a piece
            if self.board[x][col] == 0:
                safe_row = x  # Represents the first open spot
                break

        self.board[safe_row][col] = player_id  # Turn that spot for the player
        three_in_a_row = self.three_in_a_row(safe_row, col)
        if safe_row == 0:  # Removes columns as they are filled
            self.valid_cols.remove(col)
        self.size += 1  # Increase the piece count

        if self.four_in_a_row(safe_row, col):
            return 1, (safe_row, col), block_loc
        elif self.size == 42:  # Board is now filled
            return -2, (None, None), block_loc
        elif three_in_a_row[0]:
            block_loc = three_in_a_row[1]
            # print(f'returning {2, (safe_row, col), block_loc}')
            return 2, (safe_row, col), block_loc
        else:
            return 3, (safe_row, col), block_loc

    def four_in_a_row(self, x, y):
        def vertical():
            row = self.board[x]
            for loc in range(3, 7):
                if row[loc - 3] == row[loc - 2] == row[loc - 1] == row[loc] and row[loc] != 0:
                    return True
            return False

        def horizontal():
            for loc in range(3, 6):
                if self.board[loc - 3][y] == self.board[loc - 2][y] == self.board[loc - 1][y] == self.board[loc][y] and \
                        self.board[loc][y] != 0:
                    return True
            return False

        def diagonal_1():
            chip = self.board[x][y]

            if chip == 0:
                return False

            x_down, y_down = x, y  # Top left corner
            count = 0
            while x_down > -1 and y_down > -1 and self.board[x_down][y_down] == chip:
                count += 1
                x_down -= 1
                y_down -= 1

            x_up, y_up = x, y  # Bottom right corner
            count -= 1  # Decrementing for counting the dropped piece twice
            while x_up < 6 and y_up < 7 and self.board[x_up][y_up] == chip:
                count += 1
                x_up += 1
                y_up += 1

            # 1, 1 = x down y down; 3, 3 = x up y up
            if count >= 4:
                return True
            return False

        def diagonal_2():
            chip = self.board[x][y]

            if chip == 0:
                return False

            x_down, y_up = x, y  # Top right corner
            count = 0
            while x_down > -1 and y_up < 7 and self.board[x_down][y_up] == chip:
                count += 1
                x_down -= 1
                y_up += 1

            x_up, y_down = x, y  # Bottom left corner
            count -= 1  # Decrementing for counting the dropped piece twice
            while x_up < 6 and y_down < 7 and self.board[x_up][y_down] == chip:
                count += 1
                x_up += 1
                y_up -= 1

            # 1, 1 = x down y down; 3, 3 = x up y up
            if count >= 4:
                return True

            return False

        return vertical() or horizontal() or diagonal_1() or diagonal_2()

    def three_in_a_row(self, x, y):
        block_pos = list()

        def vertical():
            is_three = False
            row = self.board[x]
            for loc in range(2, 7):
                if row[loc - 2] == row[loc - 1] == row[loc] and row[loc] != 0:
                    if self.valid_move_check(x, loc - 3):
                        block_pos.append((x, loc - 3))
                    if self.valid_move_check(x, loc + 1):
                        block_pos.append((x, loc + 1))
                    is_three = True

            return is_three

        def horizontal():
            is_three = False
            for loc in range(2, 6):
                if self.board[loc - 2][y] == self.board[loc - 1][y] == self.board[loc][y] and self.board[loc][y] != 0:
                    # print(f'CHECKING {loc - 3, y} if it a valid move')
                    if self.valid_move_check(loc - 3, y):
                        # print(f'{loc - 3, y} IS VALID')
                        block_pos.append((loc - 3, y))
                    if self.valid_move_check(loc + 1, y):
                        block_pos.append((loc + 1, y))

            return is_three

        def diagonal_1():
            is_three = False
            chip = self.board[x][y]

            if chip == 0:
                return False

            x_down, y_down = x, y  # Top left corner
            count = 0
            while x_down > -1 and y_down > -1 and self.board[x_down][y_down] == chip:
                count += 1
                x_down -= 1
                y_down -= 1

            x_up, y_up = x, y  # Bottom right corner
            count -= 1  # Decrementing for counting the dropped piece twice
            while x_up < 6 and y_up < 7 and self.board[x_up][y_up] == chip:
                count += 1
                x_up += 1
                y_up += 1

            # 1, 1 = x down y down; 3, 3 = x up y up
            if count >= 3:
                is_three = True
                if self.valid_move_check(x_down - 1, y_down - 1):
                    block_pos.append((x_down - 1, y_down - 1))
                if self.valid_move_check(x_up + 1, y_up + 1):
                    block_pos.append((x_up + 1, y_up + 1))

            return is_three

        def diagonal_2():
            is_three = False
            chip = self.board[x][y]

            if chip == 0:
                return False

            x_down, y_up = x, y  # Top right corner
            count = 0
            while x_down > -1 and y_up < 7 and self.board[x_down][y_up] == chip:
                count += 1
                x_down -= 1
                y_up += 1

            x_up, y_down = x, y  # Bottom left corner
            count -= 1  # Decrementing for counting the dropped piece twice
            while x_up < 6 and y_down < 7 and self.board[x_up][y_down] == chip:
                count += 1
                x_up += 1
                y_up -= 1

            # 1, 1 = x down y down; 3, 3 = x up y up
            if count >= 3:
                is_three = True
                if self.valid_move_check(x_down - 1, y_up + 1):
                    block_pos.append((x_down - 1, y_up + 1))
                if self.valid_move_check(x_up + 1, y_down - 1):
                    block_pos.append((x_up + 1, y_down - 1))

            return is_three

        return_bool = vertical()
        return_bool = horizontal() or return_bool
        return_bool = diagonal_1() or return_bool
        return_bool = diagonal_2() or return_bool
        return return_bool, block_pos


class RandomBot:
    def reset(self):
        self.possible_moves = [x for x in range(7)]

    def action(self):
        """Makes a completely random move"""
        col = random.choice(list(BOARD.get_valid_moves()))
        return col


class User:
    def action(self):
        """A method which allows the user to make a move"""
        print(BOARD)
        move = int(input("You're move is: "))
        return move if move in BOARD.get_valid_moves() else random.choice(list(BOARD.get_valid_moves()))


class Connect4Env:
    SIZE_ROWS = 6
    SIZE_COLUMNS = 7
    RETURN_IMAGES = True

    GAME_WIN_REWARD = 70_000
    BLOCK_ENEMY_REWARD = 500
    THREE_IN_A_ROW_REWARD = 30
    MOVE_PENALTY = 1
    # All Below this comment will be turned negative
    THREE_IN_A_ROW_PENALTY = 40_000
    GAME_DRAW_PENALTY = 60_000
    GAME_LOSS_PENALTY = 100_000
    ILLEGAL_MOVE_PENALTY = 500_000

    OBSERVATION_SPACE_VALUES = (SIZE_COLUMNS, SIZE_ROWS, 3)
    ACTION_SPACE_SIZE = 7
    player_1 = [1, None]
    player_2 = [-1, None]
    turn = 1
    episode_step = 0

    def __init__(self):
        pass

    def reset(self):
        """Function to reset the episode to base state"""
        BOARD.clear()
        if training_mode == 1:  # Only for training against random bot
            bot.reset()
        self.episode_step = 0
        return BOARD.board_state()  # Return the current state (Which is empty)

    def step(self, turn_action, user, block_loc):
        # print(block_loc, user)
        if user == self.player_1[1]:  # Finds if this move was the 1st player or 2nd
            result = BOARD.drop(turn_action, self.player_1[0])  # 1 is the piece
        else:
            result = BOARD.drop(turn_action, self.player_2[0])  # -1 is the piece

        # print(result)
        if result[0] == -2:
            move_reward = -self.GAME_DRAW_PENALTY  # Draw
        elif result[0] == -1:
            move_reward = -self.ILLEGAL_MOVE_PENALTY  # Illegal Move || THIS CODE SHOULD NEVER BE REACHED
        elif result[0] == 1 and user == bot:  # Loss
            move_reward = -self.GAME_LOSS_PENALTY
        elif result[0] == 1 and user == agent:  # Win
            move_reward = self.GAME_WIN_REWARD
        elif (result[0] == 3 or result[0] == 2) and block_loc and result[1] not in block_loc:
            # Failed to block 3 in a row
            move_reward = -self.THREE_IN_A_ROW_PENALTY
        elif result[0] == 2 and user == agent:  # Agent 3 in a row
            move_reward = self.THREE_IN_A_ROW_REWARD
        else:
            move_reward = -self.MOVE_PENALTY

        if move_reward == self.GAME_WIN_REWARD or move_reward == -self.GAME_LOSS_PENALTY \
                or move_reward == -self.GAME_DRAW_PENALTY:
            complete = True
        else:
            complete = False

        self.episode_step += 1
        return np.array(BOARD.board_state()), move_reward, complete, result[2]


agent_training = str(
    input("Please give the path to the model you want to train, or write 'NEW' if you wish to train a new model:  "))
if agent_training == 'NEW':
    agent = DQNAgent()
else:
    agent = DQNAgent(
        tf.keras.models.load_model(agent_training).get_weights())

training_mode = str(input("Please give path to the model you wish to train against. If you wish to train against "
                          "the random bot please enter 'RANDOM'. If you wish to manually train the bot please enter"
                          " 'USER':  "))
if training_mode == 'RANDOM':
    bot = RandomBot()
elif training_mode == 'USER':
    bot = User()
else:
    bot = tf.keras.models.load_model(training_mode)

env = Connect4Env()
BOARD = Board()
results = {70_000: 'Agent Win', 500: 'Block Enemy', 30: '3 in a row', -1: 'Move',
           -4_000: 'Unblocked 3 in a row', -60_000: 'Draw', -100_000: 'Bot Win'}

ep_rewards = [0]

if not os.path.isdir('models'):  # Creates model directory
    os.makedirs('models')

for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episode'):
    agent.tensorboard.step = episode
    current_state = np.array(env.reset())
    last_player_state = None
    last_move = 0
    episode_reward = 0
    step = 1
    turn = 1
    done = False
    last_block_loc = None
    last_reward = 0
    before_loss = [current_state, None, None]

    if random.randint(0, 2) == 0:  # Choosing who goes first
        player_1, player_2, env.player_1[1], env.player_2[1] = agent, bot, agent, bot
        players = {player_1: 'Agent', player_2: 'Bot'}
        player_turn = 1
    else:
        player_1, player_2, env.player_1[1], env.player_2[1] = bot, agent, bot, agent
        players = {player_1: 'Bot', player_2: 'Agent'}
        player_turn = -1
    #
    # player_1, player_2, env.player_1[1], env.player_2[1] = bot, agent, bot, agent
    # players = {player_1: 'USER', player_2: 'AGENT'}
    # print(players)
    while not done:
        if turn == 1:  # Odd Moves
            player = players[player_1]
            q_dict = dict()
            if player_1 == agent:  # Agent goes 1st
                if np.random.random() > epsilon:
                    action = agent.get_qs(current_state)
                    q_dict = {action[0]: 0, action[1]: 1, action[2]: 2, action[3]: 3,
                              action[4]: 4, action[5]: 5, action[6]: 6}
                    is_epsi = True
                else:
                    action = random.choice(list(BOARD.get_valid_moves()))
                    is_epsi = False

                counter = 0
                if is_epsi:  # epsilon gives a invalid move
                    while counter < 7 and q_dict[action[counter]] not in BOARD.get_valid_moves():
                        # While we keep doing invalid moves, punish and find new input
                        agent.update_replay_memory((current_state, q_dict[action[counter]], -env.ILLEGAL_MOVE_PENALTY,
                                                    current_state, done))
                        agent.train(done)
                        counter += 1  # Move to the next index
                    try:
                        action = q_dict[action[counter]]  # Assigning action its index/col
                    except IndexError:
                        print(f'Action: {action}\n{BOARD}')

                else:  # Random move is invalid
                    agent.update_replay_memory((current_state, action, -env.ILLEGAL_MOVE_PENALTY,
                                                current_state, done))
                    agent.train(done)
                    action = random.choice(list(BOARD.get_valid_moves()))

                new_state, reward, done, last_block_loc = env.step(action, agent, last_block_loc)

                before_loss[0], before_loss[1], before_loss[2] = current_state, action, new_state
                episode_reward += reward

                last_reward = reward
                last_move = action
                if type(action) == np.ndarray:
                    print(f'Agent Spot 1 action: {action}\nCurrent State:\n{BOARD}')

                agent.update_replay_memory((current_state, last_move, reward, new_state, done))
                agent.train(done)

            else:  # Bot goes 1st
                if training_mode == 'RANDOM' or training_mode == 'USER':
                    action = bot.action()
                else:
                    action = np.argmax(bot.predict(current_state.reshape((1, 6, 7, 1))))
                    if action not in BOARD.get_valid_moves():
                        action = random.choice(list(BOARD.get_valid_moves()))

                new_state, reward, done, last_block_loc = env.step(action, bot, last_block_loc)

                if reward == -env.GAME_LOSS_PENALTY:
                    agent.update_replay_memory((last_player_state, last_move, -env.GAME_LOSS_PENALTY,
                                                new_state, done))
                    agent.train(done)
                elif reward == -env.GAME_DRAW_PENALTY:
                    agent.update_replay_memory((last_player_state, last_move, -env.GAME_DRAW_PENALTY,
                                                new_state, done))
                    agent.train(done)

                last_reward = reward
                if type(action) == np.ndarray:
                    print(f'Bot Spot 1 action: {action}\nCurrent State:\n{BOARD}\n'
                          f'Previous Agent move: {last_move}\nLast Agent State:\n{last_player_state}')

        else:  # Even Moves
            player = players[player_2]
            q_dict = dict()
            if player_2 == agent:  # Agent goes 2nd
                if np.random.random() > epsilon:
                    action = agent.get_qs(current_state)

                    q_dict = {action[0]: 0, action[1]: 1, action[2]: 2, action[3]: 3,
                              action[4]: 4, action[5]: 5, action[6]: 6}
                    is_epsi = True
                else:
                    action = random.choice(list(BOARD.get_valid_moves()))
                    is_epsi = False

                counter = 0
                if is_epsi:  # epsilon gives a invalid move
                    while counter < 7 and q_dict[action[counter]] not in BOARD.get_valid_moves():
                        # While we keep doing invalid moves, punish and find new input
                        agent.update_replay_memory((current_state, q_dict[action[counter]], -env.ILLEGAL_MOVE_PENALTY,
                                                    current_state, done))
                        agent.train(done)
                        counter += 1  # Move to the next index
                    try:
                        action = q_dict[action[counter]]  # Assigning action its index/col
                    except IndexError:
                        print(f'Action: {action}\n{BOARD}')
                else:  # Random move is invalid
                    agent.update_replay_memory((current_state, action, -env.ILLEGAL_MOVE_PENALTY,
                                                current_state, done))
                    action = random.choice(list(BOARD.get_valid_moves()))
                    agent.train(done)

                new_state, reward, done, last_block_loc = env.step(action, agent, last_block_loc)

                before_loss[0], before_loss[1], before_loss[2] = current_state, action, new_state
                episode_reward += reward

                last_reward = reward
                last_move = action
                if type(action) == np.ndarray:
                    print(f'Agent Spot 2 action: {action}\nCurrent State:\n{BOARD}')

                agent.update_replay_memory((current_state, action, reward, new_state, done))
                agent.train(done)

            else:  # Bot goes 2nd
                if training_mode == 'RANDOM' or training_mode == 'USER':
                    action = bot.action()
                else:
                    action = np.argmax(bot.predict(current_state.reshape((1, 6, 7, 1))))
                    if action not in BOARD.get_valid_moves():
                        action = random.choice(list(BOARD.get_valid_moves()))
                new_state, reward, done, last_block_loc = env.step(action, bot, last_block_loc)

                if reward == -env.GAME_LOSS_PENALTY:
                    agent.update_replay_memory((last_player_state, last_move, -env.GAME_LOSS_PENALTY,
                                                new_state, done))
                    agent.train(done)
                elif reward == -env.GAME_DRAW_PENALTY:
                    agent.update_replay_memory((last_player_state, last_move, -env.GAME_DRAW_PENALTY,
                                                new_state, done))
                    agent.train(done)

                last_reward = reward
                if type(action) == np.ndarray:
                    print(f'Bot Spot 2 action: {action}\nCurrent State:\n{BOARD}\n'
                          f'Previous Agent move: {last_move}\nLast Agent State:\n{last_player_state}')

        if player_turn == turn:
            last_player_state = current_state.copy()
        turn *= -1
        current_state = new_state
        step += 1
        # print(f'\n{BOARD}')
        print(f'Current Move Reward {reward}')
    # Printing episode results
    print(f'Episode #{episode}, Result: {results[reward]}\nReward: {episode_reward}, Steps: {step}\nEpsilon {epsilon}')

    ep_rewards.append(episode_reward)
    if not episode % AGGREGATE_STATS_EVERY or episode == 1:
        average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:]) / len(ep_rewards[-AGGREGATE_STATS_EVERY:])
        min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
        max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
        agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward,
                                       reward_max=max_reward, epsilon=epsilon)

        if min_reward >= MIN_REWARD:
            agent.model.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}'
                             f'avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

        if epsilon > MIN_EPSILON:
            epsilon *= EPSILON_DECAY
            epsilon = max(MIN_EPSILON, epsilon)

    if episode % 15000 == 0:  # Reset epsilon every so often
        epsilon = 1
