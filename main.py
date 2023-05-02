import numpy
import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, Flatten, InputLayer
from keras.callbacks import TensorBoard
from keras.optimizers import Adam

from collections import deque
import random
import os
import time

from tqdm import tqdm

REPLAY_MEMORY_SIZE = 100_000
MIN_REPLAY_MEMORY_SIZE = 1_000
MODEL_NAME = '256x2_V2'
MINIBATCH_SIZE = 64
DISCOUNT = 0.99
UPDATE_TARGET_EVERY = 5

MIN_REWARD = -200

EPISODES = 20_000

epsilon = 1
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.001

AGGREGATE_STATS_EVERY = 50
SHOW_PREVIEW = False


class DQNAgent:
    def __init__(self, weights=None):
        # print(weights)
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
        model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
        # model.add(MaxPooling2D(2, 2))
        model.add(Dropout(0.3))
        # model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
        # # model.add(MaxPooling2D(2, 2))
        # model.add(Dropout(0.3))

        model.add(Flatten())
        model.add(Dense(256))
        model.add(Dropout(0.3))
        model.add(Dense(128))
        model.add(Dropout(0.3))
        model.add(Dense(7, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
        return model

    def update_replay_memory(self, transition):
        """Updates replay memory"""
        self.replay_memory.append(transition)

    def get_qs(self, state):
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

            if type(action) == np.ndarray:
                action = np.argmax(action)
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            X.append(current_state)
            y.append(current_qs)

        self.model.fit(np.array(X) / 255, np.array(y), batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False,
                       callbacks=[self.tensorboard] if terminal_state else None)

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
        if x == 5:  # Bottom piece is always a valid move
            return True
        if self.board[x - 1][y] != 0:  # If there is a piece in the spot below, it is a valid move
            return True
        return False

    def get_valid_moves(self):
        return self.valid_cols

    def board_state(self):
        """Return a copy of the board"""
        return self.board.copy()

    def col_full(self, col):
        """Lets us know if a column is full"""
        if self.board[0][col] != 0:
            self.valid_cols.remove(col)
            return True
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

        if self.size == 42:  # Board full, draw
            return -2, None, (None, None)
        if self.col_full(col):  # If full, return -1
            return -1, None, (None, None)

        safe_row = -1
        for x in range(0, 6):  # Finds the first spot open to drop a piece
            if self.board[x][col] != 0:
                safe_row = x - 1  # Represents the first open spot
                break

        self.board[safe_row][col] = player_id  # Turn that spot for the player

        if safe_row == 0:  # Removes columns as they are filled
            self.valid_cols.remove(col)
        self.size += 1  # Increase the piece count

        if self.four_in_a_row(safe_row, col):
            return 1, None, (safe_row, col)
        elif self.three_in_a_row(safe_row, col)[0]:
            return 2, self.three_in_a_row(safe_row, col)[1], (safe_row, col)
        else:
            return 3, None, (safe_row, col)

    # def found_winner(self, x, y):
    #     """
    #     Checks if the most recent move found a winner
    #     1 - Winner found
    #     2 - No winner found
    #     """
    #
    #     def across():  # Row check
    #         row = self.board[x]
    #         for loc in range(3, 7):
    #             cell_1, cell_2, cell_3, cell_4 = row[loc - 3], row[loc - 2], row[loc - 1], row[loc]
    #             if cell_1 == cell_2 == cell_3 == cell_4 and cell_1 != 0:
    #                 return True
    #             return False
    #
    #     def up():  # Col check
    #         for loc in range(3, 6):
    #             cell_1, cell_2, cell_3, cell_4 = self.board[loc - 3][y], self.board[loc - 2][y], \
    #                                              self.board[loc - 1][y], self.board[loc][y]
    #             # print(f'{cell_1}, {cell_2}, {cell_3}, {cell_4}')
    #             if cell_1 == cell_2 == cell_3 == cell_4 and cell_1 != 0:
    #                 return True
    #         return False
    #
    #     def diagonal_1():  # 0,0 to 5,6
    #         chip = self.board[x][y]
    #         if chip == 0:  # Edge case
    #             return False
    #
    #         x_down, y_down = x - 1, y - 1  # To top left corner
    #         count = 1
    #         while x_down > -1 and y_down > -1 and self.board[x_down][y_down] == chip:
    #             # print(f'{board[x_down][y_down]}: {x_down}, {y_down}')
    #             count += 1
    #             x_down -= 1
    #             y_down -= 1
    #
    #         x_up, y_up = x + 1, y + 1  # To bottom right corner
    #         while x_up < 6 and y_up < 7 and self.board[x_up][y_up] == chip:
    #             # print(f'{board[x_up][y_up]}: {x_up}, {y_down}')
    #             count += 1
    #             x_up += 1
    #             y_up += 1
    #
    #         return count >= 4
    #
    #     def diagonal_2():  # 0,6 to 5,0
    #         chip = self.board[x][y]
    #         if chip == 0:  # edge case
    #             return False
    #
    #         x_down, y_up = x - 1, y + 1  # To top left corner
    #         count = 1
    #         while x_down > -1 and y_up < 7 and self.board[x_down][y_up] == chip:
    #             # print(f'{board[x_down][y_down]}: {x_down}, {y_down}')
    #             count += 1
    #             x_down -= 1
    #             y_up += 1
    #
    #         x_up, y_down = x + 1, y - 1  # To bottom right corner
    #         while x_up < 6 and y_down > -1 and self.board[x_up][y_down] == chip:
    #             # print(f'{board[x_up][y_down]}: {x_up}, {y_down}')
    #             count += 1
    #             x_up += 1
    #             y_down -= 1
    #
    #             return count >= 4
    #
    #     if across() or up() or diagonal_1() or diagonal_2():
    #         return 1  # Winner found
    #     elif self.size == 42:
    #         return -2  # Draw
    #     else:
    #         return 2  # Continue playing

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
                    if self.valid_move_check(loc - 3, y):
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
        print('The current board state is:')
        print(current_state)
        move = int(input("You're move is: "))
        if move not in BOARD.get_valid_moves():
            print('INVALID MOVE: MOVE BEING RANDOMIZED')
        return move if move in BOARD.get_valid_moves() else random.choice(list(BOARD.get_valid_moves()))


class Connect4Env:
    SIZE_ROWS = 6
    SIZE_COLUMNS = 7
    RETURN_IMAGES = True

    GAME_WIN_REWARD = 70
    BLOCK_ENEMY_REWARD = 50
    THREE_IN_A_ROW_REWARD = 30
    MOVE_PENALTY = 0
    # All Below this comment will be turned negative
    THREE_IN_A_ROW_PENALTY = 40
    GAME_DRAW_PENALTY = 60
    GAME_LOSS_PENALTY = 100
    ILLEGAL_MOVE_PENALTY = 500

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

    def step(self, turn_action, user):
        if user == self.player_1[1]:  # Finds if this move was the 1st player or 2nd
            result = BOARD.drop(turn_action, self.player_1[0])  # 1 is the piece
        else:
            result = BOARD.drop(turn_action, self.player_2[0])  # -1 is the piece

        # print(result)

        last_move = result[2]  # Updates the last move

        if result[0] == -2:
            move_reward = -self.GAME_DRAW_PENALTY  # Draw
        elif result[0] == -1:
            print('ERROR: INVALID MOVE')
            move_reward = -self.ILLEGAL_MOVE_PENALTY  # Illegal Move || THIS CODE SHOULD NEVER BE REACHED
        elif result[0] == 1 and user == bot:  # Loss
            move_reward = -self.GAME_LOSS_PENALTY
        elif result[0] == 1 and user == agent:  # Win
            move_reward = self.GAME_WIN_REWARD
        elif result[0] == 2 and user == bot and block_loc and last_move not in block_loc:  # Failed to block 3 in a row
            move_reward = self.THREE_IN_A_ROW_PENALTY
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
        return np.array(BOARD.board_state()), move_reward, complete, result[1]


agent_training = int(
    input("Please indicate which model you wish to train.\n1: New Model\n2: V1 141avg\n3: V2 143 avg\n"))
if agent_training == 1:
    agent = DQNAgent()
elif agent_training == 2:
    agent = DQNAgent(
        tf.keras.models.load_model('models/256x2___200.00max__141.99avg_-100.02min__1682878860.model').get_weights())
else:
    agent = DQNAgent(
        tf.keras.models.load_model('models/256x2-V2___200.00max__143.99avg___-0.02min__1682896420.model').get_weights())

training_mode = int(input("Please give which training you want.\n1: Randomized bot\n2: V1 110avg\n3: V1 141avg\n4: V2 "
                          "110avg\n5: V2 143avg\nAny other number is user\n"))
if training_mode == 1:
    bot = RandomBot()
elif training_mode == 2:
    bot = tf.keras.models.load_model('models/256x2___200.00max__111.99avg___-0.02min__1682878784.model')
elif training_mode == 3:
    bot = tf.keras.models.load_model('models/256x2___200.00max__141.99avg_-100.02min__1682878860.model')
elif training_mode == 4:
    bot = tf.keras.models.load_model('models/256x2-V2___200.00max__109.99avg_-100.02min__1682895238.model')
elif training_mode == 5:
    bot = tf.keras.models.load_model('models/256x2-V2___200.00max__143.99avg___-0.02min__1682896420.model')
else:
    bot = User()

env = Connect4Env()
BOARD = Board()
results = {70: 'Agent Win', 50: 'Block Enemy', 30: '3 in a row', 0: 'Move',
           -40: 'Unblocked 3 in a row', -60: 'Draw', -100: 'Bot Win'}
ep_rewards = [0]

if not os.path.isdir('models'):  # Creates model directory
    os.makedirs('models')

for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episode'):
    agent.tensorboard.step = episode
    current_state = np.array(env.reset())
    episode_reward = 0
    step = 1
    turn = 1
    done = False
    block_loc = None
    last_move = None
    last_reward = 0
    before_loss = [current_state, None, None]

    if random.randint(0, 2) == 0:  # Choosing who goes first
        player_1, player_2, env.player_1[1], env.player_2[1] = agent, bot, agent, bot
        players = {player_1: 'Agent', player_2: 'Bot'}
    else:
        player_1, player_2, env.player_1[1], env.player_2[1] = bot, agent, bot, agent
        players = {player_1: 'Bot', player_2: 'Agent'}

    print(players)

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
                        agent.update_replay_memory((current_state, action, -env.ILLEGAL_MOVE_PENALTY,
                                                    current_state, done))
                        counter += 1  # Move to the next index
                    action = q_dict[action[counter]]  # Assigning action its index/col
                else:  # Random move is invalid
                    agent.update_replay_memory((current_state, action, -env.ILLEGAL_MOVE_PENALTY,
                                                current_state, done))
                    action = random.choice(list(BOARD.get_valid_moves()))

                new_state, reward, done, block_loc = env.step(action, agent)

                before_loss[0], before_loss[1], before_loss[2] = current_state, action, new_state
                episode_reward += reward

                agent.update_replay_memory((current_state, action, reward, new_state, done))
                agent.train(done)
                last_reward = reward

            else:  # Bot goes 1st
                if training_mode == 1 or training_mode > 5:
                    action = bot.action()
                else:
                    action = np.argmax(bot.predict(current_state.reshape((1, 6, 7, 1))))
                    if action not in BOARD.get_valid_moves():
                        action = random.choice(list(BOARD.get_valid_moves()))

                new_state, reward, done, block_loc = env.step(action, bot)

                if reward == -env.GAME_LOSS_PENALTY:
                    agent.update_replay_memory((current_state, action, -env.GAME_LOSS_PENALTY,
                                                new_state, done))
                elif reward == -env.GAME_DRAW_PENALTY:
                    agent.update_replay_memory((current_state, action, -env.GAME_DRAW_PENALTY,
                                                new_state, done))

                last_reward = reward

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
                        agent.update_replay_memory((current_state, action, -env.ILLEGAL_MOVE_PENALTY,
                                                    current_state, done))
                        counter += 1  # Move to the next index
                    action = q_dict[action[counter]]  # Assigning action its index/col
                else:  # Random move is invalid
                    agent.update_replay_memory((current_state, action, -env.ILLEGAL_MOVE_PENALTY,
                                                current_state, done))
                    action = random.choice(list(BOARD.get_valid_moves()))

                new_state, reward, done, block_loc = env.step(action, agent)

                before_loss[0], before_loss[1], before_loss[2] = current_state, action, new_state
                episode_reward += reward

                agent.update_replay_memory((current_state, action, reward, new_state, done))
                agent.train(done)
                last_reward = reward

            else:  # Bot goes 2nd
                if training_mode == 1 or training_mode > 5:
                    action = bot.action()
                else:
                    action = np.argmax(bot.predict(current_state.reshape((1, 6, 7, 1))))
                    if action not in BOARD.get_valid_moves():
                        action = random.choice(list(BOARD.get_valid_moves()))
                new_state, reward, done, block_loc = env.step(action, bot)
                if reward == -env.GAME_LOSS_PENALTY:
                    agent.update_replay_memory((current_state, action, -env.GAME_LOSS_PENALTY,
                                                new_state, done))
                elif reward == -env.GAME_DRAW_PENALTY:
                    agent.update_replay_memory((current_state, action, -env.GAME_DRAW_PENALTY,
                                                new_state, done))

                last_reward = reward

        turn *= -1
        current_state = new_state
        step += 1
        print(f'Current Move Reward {reward}. Board is \n{BOARD}')
    # Printing episode results
    print(f'Episode #{episode}, Result: {results[reward]}, Reward: {reward}, Epsilon {epsilon}')

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
