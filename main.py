import tensorflow as tf
import numpy as np
from keras import Sequential
from keras.layers import Dense, Dropout, Conv2D, Flatten, InputLayer, MaxPooling2D, Concatenate, Input, TimeDistributed, \
    LSTM
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
from keras.models import Model

from collections import deque
import random
import os
import time

from keras.regularizers import l2
from tqdm import tqdm

import matplotlib.pyplot as plt

tester = False
REPLAY_MEMORY_SIZE = 30_000
MIN_REPLAY_MEMORY_SIZE = 1_000
MODEL_NAME = 'V24-256x2'
MINIBATCH_SIZE = 64
DISCOUNT = 0.85
UPDATE_TARGET_EVERY = 50

EPISODES = 2000

epsilon = 1
EPSILON_DECAY = 0.997
MIN_EPSILON = 0.001

LOWER_DECAY = 50
AGGREGATE_STATS_EVERY = 100


class DQNAgent:
    def __init__(self, weights=None):
        self.model = self.create_model()  # main model

        self.target_model = self.create_model()  # target model
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        self.tensorboard = ModifiedTensorBoard(log_dir=f'logs/{MODEL_NAME}-{int(time.time())}')

        self.target_update_counter = 0

        # print(self.model.summary())

        if weights:
            self.model.set_weights(weights)
            self.target_model.set_weights(weights)

    def create_model(self):
        """Creates both the target model and the main model"""

        model = Sequential()
        model.add(InputLayer(input_shape=(6, 7, 1)))  # Dimension of the 2d list with 1 for greyscale

        # model.add(Conv2D(32, (5, 5), activation='relu', padding='same', kernel_regularizer=l2(0.01)))
        # model.add(Dropout(0.5))

        # Define the single number input layer
        model.add(Flatten())
        model.add(Dense(50, activation='relu', kernel_regularizer=l2(0.01)))
        model.add(Dense(50, activation='relu', kernel_regularizer=l2(0.01)))
        model.add(Dense(50, activation='relu', kernel_regularizer=l2(0.01)))
        model.add(Dense(50, activation='relu', kernel_regularizer=l2(0.01)))
        model.add(Dense(50, activation='relu', kernel_regularizer=l2(0.01)))
        model.add(Dense(50, activation='relu', kernel_regularizer=l2(0.01)))

        # model.add(Dense(256, activation='relu', kernel_regularizer=l2(0.01)))
        # model.add(Dropout(0.5))
        # model.add(Dense(256, activation='relu', kernel_regularizer=l2(0.01)))

        # model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.01)))
        # model.add(Dropout(0.5))
        # model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.01)))
        model.add(Dense(7, activation='softmax'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

        # input = Input((6, 7, 1))
        # hidden_1 = Conv2D(32, (5, 5), activation='relu', padding='same', kernel_regularizer=l2(0.01))(input)
        # hidden_2 = Dropout(0.5)(hidden_1)
        # hidden_3 = Flatten()(hidden_2)
        # hidden_4 = Dense(400, activation='relu', kernel_regularizer=l2(0.01))(hidden_3)
        # hidden_5 = Dropout(0.5)(hidden_4)
        # hidden_6 = Dense(200, activation='relu', kernel_regularizer=l2(0.01))(hidden_5)
        # hidden_7 = Dropout(0.5)(hidden_6)
        # hidden_8 = Dense(100, activation='relu', kernel_regularizer=l2(0.01))(hidden_7)
        # hidden_9 = Dropout(0.5)(hidden_8)
        # hidden_10 = Dense(50, activation='relu', kernel_regularizer=l2(0.01))(hidden_9)
        # output = Dense(7, activation='softmax', kernel_regularizer=l2(0.01))(hidden_10)
        # model = Model(inputs=[input], outputs=[output])

        return model

    def update_replay_memory(self, transition):
        """Updates replay memory"""
        self.replay_memory.append(transition)

    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape))[0]

    def train(self, terminal_state):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:  # Do not train if small sample size
            return

        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        current_states = np.array([transition[0] for transition in minibatch])

        current_qs_list = self.model.predict(current_states)

        new_current_states = np.array([transition[3] for transition in minibatch])
        future_qs_list = self.target_model.predict(new_current_states)

        # for sample in range(len(minibatch)):
        #     print(f'Current Sample is #{sample}\n'
        #           f'Minibatch Sample is:\n{minibatch[sample]}\n'
        #           f'Current State is:\n{current_states[sample]}\n'
        #           f'Current QS List is:\n{current_qs_list[sample]}\n'
        #           f'New_current_states is:\n{new_current_states[sample]}\n'
        #           f'Future QS List is:\n{future_qs_list[sample]}\n')

        X, y = list(), list()

        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):
            if not done:  # episode incomplete
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

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

    def valid_move_check(self, x, y=None):
        """Returns if the current move is possible"""
        if y is None:
            x, y = x[0], x[1]

        if x > 5 or y > 6 or x < 0 or y < 0:
            return False
        if x == 5:  # Bottom piece is always a valid move
            return True
        if self.board[x + 1][y] != 0:  # If there is a piece in the spot below, it is a valid move
            return True
        return False

    def cell_value(self, x, y):
        """Returns the value stored in a cell"""
        if x > 5 or x < 0 or y > 6 or y < 0:  # Invalid location
            return 0
        return self.board[x][y]

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
        except KeyError:
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

        Block_loc refers to the locations the player NOT going must go to prevent connect 4s
        """
        block_loc = set()
        if self.size == 42:  # Board full, draw
            return -2, (-1, -1), block_loc
        if self.col_full(col):  # If full, return -1
            return -1, (-1, -1), block_loc

        safe_row = -1
        for x in range(5, -1, -1):  # Finds the first spot open to drop a piece
            if self.board[x][col] == 0:
                safe_row = x  # Represents the first open spot
                break

        self.board[safe_row][col] = player_id  # Turn that spot for the player

        if safe_row == 0:  # Removes columns as they are filled
            self.valid_cols.remove(col)

        self.size += 1  # Increase the piece count

        # Game is over
        if self.four_in_a_row(safe_row, col):
            return 1, (safe_row, col), block_loc, self.four_in_a_row(safe_row, col)[1]

        elif self.size == 42:  # Board is now filled
            return -2, (-1, -1), block_loc

        # Find if any connect 4s can be made with the newly opened cell
        if safe_row > 0:
            # to_block = self.one_up(safe_row - 1, col)
            to_block_player = self.three_in_a_row(safe_row - 1, col, player_id)  # Where opponent must block
            if to_block_player[0]:  # If there is now locations to block
                to_block_player = to_block_player[1]
            else:  # If there is not
                to_block_player = set()

            to_block_opponent = self.three_in_a_row(safe_row - 1, col, -player_id)  # Where player must block
            if to_block_opponent[0]:  # If there is now locations to block
                to_block_opponent = to_block_opponent[1]
            else:  # If there is not
                to_block_opponent = set()

        else:  # Edge case
            to_block_player = set()
            to_block_opponent = set()

        three_in_a_row = self.three_in_a_row(safe_row, col, player_id)  # Check for 3 in a row

        if three_in_a_row[0]:
            block_loc.update(three_in_a_row[1])

            return 2, (safe_row, col), block_loc, (to_block_player, to_block_opponent)
        else:
            return 3, (safe_row, col), block_loc, (to_block_player, to_block_opponent)

    def four_in_a_row(self, x, y):
        def vertical():
            row = self.board[x]
            for loc in range(3, 7):
                if row[loc - 3] == row[loc - 2] == row[loc - 1] == row[loc] and row[loc] != 0:
                    return True, 'V'
            return False

        def horizontal():
            for loc in range(3, 6):
                if self.board[loc - 3][y] == self.board[loc - 2][y] == self.board[loc - 1][y] == self.board[loc][y] \
                        and self.board[loc][y] != 0:
                    return True, 'H'
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
                return True, 'D'
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
                y_down -= 1

            # 1, 1 = x down y down; 3, 3 = x up y up
            if count >= 4:
                return True, 'D'
            return False

        return vertical() or horizontal() or diagonal_1() or diagonal_2()

    def three_in_a_row(self, x, y, player_id):
        """
        Searches for 3 on the board by the given location
        Returns boolean, set
        """
        block_loc = set()
        has_three = False

        # Checking Vertical
        n = y - 3 if y - 3 > 0 else 0  # Prevents out of bounds on the left

        while n < 4:  # Prevents out of bounds on the right
            id_counter = 0
            zero_counter = 0

            if self.board[x][n] == player_id:
                id_counter += 1
            elif self.board[x][n] == 0:
                zero_counter += 1
                block_position = (x, n)

            if self.board[x][n + 1] == player_id:
                id_counter += 1
            elif self.board[x][n + 1] == 0:
                zero_counter += 1
                block_position = (x, n + 1)

            if self.board[x][n + 2] == player_id:
                id_counter += 1
            elif self.board[x][n + 2] == 0:
                zero_counter += 1
                block_position = (x, n + 2)

            if self.board[x][n + 3] == player_id:
                id_counter += 1
            elif self.board[x][n + 3] == 0:
                zero_counter += 1
                block_position = (x, n + 3)

            if id_counter == 3 and zero_counter == 1:
                if self.valid_move_check(block_position):
                    block_loc.add(block_position)
                    has_three = True

            n += 1

        # Checking Horizontal
        n = x - 3 if x - 3 > 0 else 0

        while n < 3:  # Prevents out of bounds on the right
            id_counter = 0
            zero_counter = 0

            if self.board[n][y] == player_id:
                id_counter += 1
            elif self.board[n][y] == 0:
                zero_counter += 1
                block_position = (n, y)

            if self.board[n + 1][y] == player_id:
                id_counter += 1
            elif self.board[n + 1][y] == 0:
                zero_counter += 1
                block_position = (n + 1, y)

            if self.board[n + 2][y] == player_id:
                id_counter += 1
            elif self.board[n + 2][y] == 0:
                zero_counter += 1
                block_position = (n + 2, y)

            if self.board[n + 3][y] == player_id:
                id_counter += 1
            elif self.board[n + 3][y] == 0:
                zero_counter += 1
                block_position = (n + 3, y)

            if id_counter == 3 and zero_counter == 1:
                if self.valid_move_check(block_position):
                    block_loc.add(block_position)
                    has_three = True

            n += 1

        # Checking Diag 1 (Top Left - Bottom Right)
        while x - n > -1 and y - n > -1:
            n += 1

        if n >= 3:
            n = 3

            while n >= 0 and x - n + 3 <= 5 and y - n + 3 <= 6:
                id_counter = 0
                zero_counter = 0
                if self.board[x - n][y - n] == player_id:
                    id_counter += 1
                elif self.board[x - n][y - n] == 0:
                    zero_counter += 1
                    block_position = (x - n, y - n)

                if self.board[x - n + 1][y - n + 1] == player_id:
                    id_counter += 1
                elif self.board[x - n + 1][y - n + 1] == 0:
                    zero_counter += 1
                    block_position = (x - n + 1, y - n + 1)

                if self.board[x - n + 2][y - n + 2] == player_id:
                    id_counter += 1
                elif self.board[x - n + 2][y - n + 2] == 0:
                    zero_counter += 1
                    block_position = (x - n + 2, y - n + 2)

                if self.board[x - n + 3][y - n + 3] == player_id:
                    id_counter += 1
                elif self.board[x - n + 3][y - n + 3] == 0:
                    zero_counter += 1
                    block_position = (x - n + 3, y - n + 3)

                if id_counter == 3 and zero_counter == 1:
                    if self.valid_move_check(block_position):
                        block_loc.add(block_position)
                        has_three = True

                n -= 1

        # Checking Diag 2 (Top Right - Bottom Left)
        n_x, n_y = x, y
        while n_x > 0 and n_y < 6:
            n_x -= 1
            n_y += 1

        if not (n_x == 0 and (n_y == 0 or n_y == 1 or n_y == 2)) or \
                (n_y == 6 and (n_x == 5 or n_x == 4 or n_x == 3)):
            while n_x + 3 <= 5 and n_y - 3 >= 0:
                id_counter = 0
                zero_counter = 0
                if self.board[n_x][n_y] == player_id:
                    id_counter += 1
                elif self.board[n_x][n_y] == 0:
                    zero_counter += 1
                    block_position = (n_x, n_y)

                if self.board[n_x + 1][n_y - 1] == player_id:
                    id_counter += 1
                elif self.board[n_x + 1][n_y - 1] == 0:
                    zero_counter += 1
                    block_position = (n_x + 1, n_y - 1)

                if self.board[n_x + 2][n_y - 2] == player_id:
                    id_counter += 1
                elif self.board[n_x + 2][n_y - 2] == 0:
                    zero_counter += 1
                    block_position = (n_x + 2, n_y - 2)

                if self.board[n_x + 3][n_y - 3] == player_id:
                    id_counter += 1
                elif self.board[n_x + 3][n_y - 3] == 0:
                    zero_counter += 1
                    block_position = (n_x + 3, n_y - 3)

                if id_counter == 3 and zero_counter == 1:
                    if self.valid_move_check(block_position):
                        block_loc.add(block_position)
                        has_three = True

                n_x += 1
                n_y -= 1
        return has_three, block_loc


class RandomBot:
    def action(self):
        """Makes a completely random move"""
        col = random.choice(list(BOARD.get_valid_moves()))
        return col


class AdvancedBot:
    def __init__(self, winning, blocking):
        self.winning = winning
        self.blocking = blocking

    def action(self, block_loc, win_loc):
        action = 0
        moves = list(BOARD.get_valid_moves())
        if len(block_loc) >= 1 and np.random.random() < self.blocking:  # Has an 80% Chance to block wins
            location = tuple(block_loc)[0]
            if location[1] not in list(BOARD.get_valid_moves()):
                return random.choice(list(BOARD.get_valid_moves()))
            return location[1]
        elif len(win_loc) >= 1 and np.random.random() < self.winning:  # Has a 50% Chance to win games
            location = tuple(win_loc)[0]
            if location[1] not in list(BOARD.get_valid_moves()):
                return random.choice(list(BOARD.get_valid_moves()))
            return location[1]
        else:  # If neither previous option is triggered, do this
            return random.choice(moves)


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

    GAME_WIN_REWARD_H = 20
    GAME_WIN_REWARD_V = 70
    GAME_WIN_REWARD_D = 100

    BLOCK_ENEMY_REWARD = 15
    THREE_IN_A_ROW_REWARD = 5
    # All Below this comment will be turned negative
    MOVE_PENALTY = 0
    THREE_IN_A_ROW_PENALTY = 7
    WIN_MISSED_PENALTY = 10
    GAME_DRAW_PENALTY = 40
    GAME_LOSS_PENALTY = 50
    ILLEGAL_MOVE_PENALTY = 10_000

    OBSERVATION_SPACE_VALUES = (SIZE_COLUMNS, SIZE_ROWS, 3)
    ACTION_SPACE_SIZE = 7
    player_1 = [1, None]
    player_2 = [-1, None]
    episode_step = 0

    def __init__(self):
        pass

    def reset(self):
        """Function to reset the episode to base state"""
        BOARD.clear()
        self.episode_step = 0
        return BOARD.board_state()  # Return the current state (Which is empty)

    def step(self, turn_action, user):
        """The step functionality. Sends the move to drop, then processes the return data and returns the reward"""
        if user == self.player_1[1]:  # Finds if this move was the 1st player or 2nd
            piece = self.player_1[0]
            result = BOARD.drop(turn_action, piece)  # 1 is the piece
            add_spots = player_2_block_loc
            remove_spots = player_1_block_loc

        else:
            piece = self.player_2[0]
            result = BOARD.drop(turn_action, piece)  # -1 is the piece
            add_spots = player_1_block_loc
            remove_spots = player_2_block_loc

        move = result[1]
        move_result = result[0]

        if len(result) > 3 and move != (-1, -1):  # Adding the new possible connect 4 where needed
            if type(result[3]) != str:
                to_block = result[3][1]
                remove_spots.update(to_block)

        if move_result == -2:
            move_reward = -self.GAME_DRAW_PENALTY  # Draw

        # elif move_result == -1:
        #     move_reward = -self.ILLEGAL_MOVE_PENALTY  # Illegal Move || THIS CODE SHOULD NEVER BE REACHED

        elif move_result == 1:  # Win
            if result[3] == 'H':
                move_reward = self.GAME_WIN_REWARD_H
            elif result[3] == 'V':
                move_reward = self.GAME_WIN_REWARD_V
            else:
                move_reward = self.GAME_WIN_REWARD_D

        elif (move_result == 3 or move_result == 2) and move not in add_spots and len(add_spots) >= 1:
            # Failed to get a possible win
            move_reward = -self.WIN_MISSED_PENALTY

        elif (move_result == 3 or move_result == 2) and move not in remove_spots and len(remove_spots) >= 1:
            # Failed to block 3 in a row
            move_reward = -self.THREE_IN_A_ROW_PENALTY

            # Covers edge case where move removes a possible connect 4 while creating a new one
            if move in remove_spots:
                remove_spots.remove(move)

        elif (move_result == 3 or move_result == 2) and move in remove_spots:
            # Blocked 3 in a row
            remove_spots.remove(move)
            move_reward = self.BLOCK_ENEMY_REWARD

        elif move_result == 2:
            # Agent 3 in a row
            move_reward = self.THREE_IN_A_ROW_REWARD

        else:
            # Basic move
            move_reward = -self.MOVE_PENALTY

        add_spots.update(result[2])

        if len(result) > 3 and move != (-1, -1):  # Adding the new possible connect 4 where needed
            if type(result[3]) != str:
                to_block = result[3][1]
                add_spots.update(to_block)

        if move_reward == self.GAME_WIN_REWARD_H or \
                move_reward == self.GAME_WIN_REWARD_D or \
                move_reward == self.GAME_WIN_REWARD_V or \
                move_reward == -self.GAME_LOSS_PENALTY or \
                move_reward == -self.GAME_DRAW_PENALTY:
            complete = True

        else:
            complete = False

        self.episode_step += 1
        return np.array(BOARD.board_state()), move_reward, complete


agent_training = str(
    input("Please give the path to the model you want to train, or write 'NEW' if you wish to train a new model. For "
          "Testing input 'USER':  "))
optimal_percent = int(input('Please give the percent you want the agent to make optimal moves when random is selected '
                            '(.87 = 87%): '))
if agent_training == 'NEW':
    agent = DQNAgent()
elif agent_training == 'USER':
    agent = User()
else:
    agent = DQNAgent(
        tf.keras.models.load_model(agent_training).get_weights())

training_mode = str(input("Please give path to the model you wish to train against. If you wish to train against "
                          "the random bot please enter 'RANDOM'. If you wish to train against the advanced bot please "
                          "enter 'ADVANCED'. If you wish to manually train the bot please enter 'USER':  "))

if training_mode == 'RANDOM':
    bot = RandomBot()
elif training_mode == 'USER':
    bot = User()
elif training_mode == 'ADVANCED':
    winning_percentage = int(input('Please put the percentage of times you want the bot to win (.87 = 87%): '))
    blocking_percentage = int(input('Please put the percentage of times you want the bot to block (.87 = 87%): '))
    bot = AdvancedBot(winning_percentage, blocking_percentage)
else:
    bot = tf.keras.models.load_model(training_mode)

env = Connect4Env()
BOARD = Board()
results = {env.GAME_WIN_REWARD_H: 'Agent Win H',
           env.GAME_WIN_REWARD_D: 'Agent Win D',
           env.GAME_WIN_REWARD_V: 'Agent Win V',
           env.BLOCK_ENEMY_REWARD: 'Block Enemy',
           env.THREE_IN_A_ROW_REWARD: '3 in a row',
           -env.MOVE_PENALTY: 'Basic',
           -env.THREE_IN_A_ROW_PENALTY: 'Unblocked 3 in a row',
           -env.WIN_MISSED_PENALTY: 'Win Was Missed',
           -env.GAME_DRAW_PENALTY: 'Draw',
           -env.GAME_LOSS_PENALTY: 'Bot Win',
           -env.ILLEGAL_MOVE_PENALTY: 'Illegal Move'
           }

ep_rewards_agent = []
ep_rewards_bot = []
specific_rewards = {
    env.GAME_WIN_REWARD_H: [0, 0],
    env.GAME_WIN_REWARD_D: [0, 0],
    env.GAME_WIN_REWARD_V: [0, 0],
    env.BLOCK_ENEMY_REWARD: [0, 0],
    env.THREE_IN_A_ROW_REWARD: [0, 0],
    -env.MOVE_PENALTY: [0, 0],
    -env.THREE_IN_A_ROW_PENALTY: [0, 0],
    -env.WIN_MISSED_PENALTY: [0, 0],
    -env.GAME_DRAW_PENALTY: [0, 0],
    -env.GAME_LOSS_PENALTY: [0, 0],
    -env.ILLEGAL_MOVE_PENALTY: [0, 0]
}

if not os.path.isdir(MODEL_NAME):  # Creates model directory
    os.makedirs(MODEL_NAME)

counting = {
    'AGENT WIN': 0,
    'BOT WIN': 0,
    'DRAW': 0
}

for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episode'):
    if agent_training != 'USER':  # For testing
        agent.tensorboard.step = episode

    current_state = np.array(env.reset())
    last_player_state, winner = None, 'Agent'
    last_move, episode_reward = 0, 0
    step, turn = 1, 1
    done = False
    player_1_block_loc, player_2_block_loc = set(), set()
    reward_sum = 0

    # Player 1 is the locations' player 1 must block, and for 2 2 must block those spots
    before_loss = [current_state, None, None]

    if random.randint(0, 2) == 0:  # Choosing who goes first
        player_1, player_2, env.player_1[1], env.player_2[1] = agent, bot, agent, bot
        players = {player_1: 'Agent', player_2: 'Bot'}
        player_turn = 1

    else:
        player_1, player_2, env.player_1[1], env.player_2[1] = bot, agent, bot, agent
        players = {player_1: 'Bot', player_2: 'Agent'}
        player_turn = -1

    # player_1, player_2, env.player_1[1], env.player_2[1] = agent, bot, agent, bot
    # players = {player_1: 'AGENT', player_2: 'USER'}
    # print(players)
    # print(player_turn)

    while not done:

        if turn == 1:  # Odd Moves
            player = players[player_1]
            q_dict = dict()
            if player_1 == agent:  # Agent goes 1st
                if agent_training == 'USER':  # For testing
                    action = agent.action()

                else:
                    if np.random.random() > epsilon:
                        # print(agent.get_qs(current_state))
                        # action = agent.get_qs(current_state)
                        # q_dict = {action[0]: 0, action[1]: 1, action[2]: 2, action[3]: 3,
                        #           action[4]: 4, action[5]: 5, action[6]: 6}
                        # action[::-1].sort()
                        is_epsi = True
                        action = np.argmax(agent.get_qs(current_state))
                    else:  # Not a epsilon move
                        if np.random.random() < optimal_percent:  # Do a totally random move
                            action = random.choice(list(BOARD.get_valid_moves()))

                        else:
                            # Select a move from the list of better moves. If one is not available do something
                            # totally random
                            optimal_moves = len(player_1_block_loc) + len(player_2_block_loc)

                            if optimal_moves > 0:  # If there is a better move do it
                                if len(player_2_block_loc) > 0:  # Move is getting a win
                                    action = random.choice(list(player_2_block_loc))[1]
                                else:  # Move is blocking a win
                                    action = random.choice(list(player_1_block_loc))[1]

                            else:  # No optimal moves, do a random move
                                action = random.choice(list(BOARD.get_valid_moves()))

                        is_epsi = False

                    if is_epsi:  # epsilon gives a invalid move

                        if action not in BOARD.get_valid_moves():
                            # While we keep doing invalid moves, punish and find new input

                            agent.update_replay_memory(
                                (current_state, action, -env.ILLEGAL_MOVE_PENALTY,
                                 current_state, done))
                            agent.train(done)
                            episode_reward -= env.ILLEGAL_MOVE_PENALTY
                            specific_rewards[-env.ILLEGAL_MOVE_PENALTY][episode] += 1
                            action = random.choice(list(BOARD.get_valid_moves()))  # Assigning action its index/col

                    elif action not in BOARD.get_valid_moves():  # Random move is invalid
                        # print('Random Move')
                        agent.update_replay_memory((current_state, action, -env.ILLEGAL_MOVE_PENALTY,
                                                    current_state, done))
                        agent.train(done)
                        action = random.choice(list(BOARD.get_valid_moves()))
                        episode_reward -= env.ILLEGAL_MOVE_PENALTY
                        specific_rewards[-env.ILLEGAL_MOVE_PENALTY][episode] += 1

                new_state, reward, done = env.step(action, agent)

                before_loss[0], before_loss[1], before_loss[2] = current_state, action, new_state
                episode_reward += reward

                last_move = action

                if agent_training != 'USER':
                    agent.update_replay_memory((current_state, last_move, reward, new_state, done))
                    agent.train(done)

                if reward == env.GAME_WIN_REWARD_H or reward == env.GAME_WIN_REWARD_V or \
                        reward == env.GAME_WIN_REWARD_D:
                    winner = 'AGENT WIN'
                elif reward == -env.GAME_DRAW_PENALTY:
                    winner = 'DRAW'

            else:  # Bot goes 1st
                if training_mode == 'RANDOM' or training_mode == 'USER' or training_mode == 'ADVANCED':
                    if training_mode == 'ADVANCED':
                        action = bot.action(player_1_block_loc, player_2_block_loc)
                    else:
                        action = bot.action()

                else:
                    if np.random.random() > epsilon:
                        action = np.argmax(bot.predict(current_state.reshape((1, 6, 7, 1))))
                        if action not in BOARD.get_valid_moves():
                            action = random.choice(list(BOARD.get_valid_moves()))
                    else:
                        action = random.choice(list(BOARD.get_valid_moves()))

                new_state, reward, done = env.step(action, bot)

                # Adding the bots moves for additional training
                if agent_training != 'USER':
                    if reward == env.GAME_WIN_REWARD_H or reward == env.GAME_WIN_REWARD_V or \
                            reward == env.GAME_WIN_REWARD_D:
                        agent.update_replay_memory(
                            (last_player_state, last_move, -env.GAME_LOSS_PENALTY, new_state, done))
                        agent.update_replay_memory((current_state, action, reward, new_state, done))
                        agent.train(done)
                        episode_reward -= env.GAME_LOSS_PENALTY

                    else:
                        agent.update_replay_memory((current_state, last_move, reward, new_state, done))
                        if reward == -env.GAME_DRAW_PENALTY:
                            episode_reward -= env.GAME_DRAW_PENALTY
                        agent.train(done)

                if reward == env.GAME_WIN_REWARD_H or reward == env.GAME_WIN_REWARD_V or \
                        reward == env.GAME_WIN_REWARD_D:
                    winner = 'BOT WIN'

                elif reward == -env.GAME_DRAW_PENALTY:
                    winner = 'DRAW'

        else:  # Even Moves
            player = players[player_2]
            q_dict = dict()
            if player_2 == agent:  # Agent goes 2nd
                if agent_training == 'USER':
                    action = agent.action()

                else:
                    if np.random.random() > epsilon:
                        # print(agent.get_qs(current_state))

                        # action = agent.get_qs(current_state)
                        # q_dict = {action[0]: 0, action[1]: 1, action[2]: 2, action[3]: 3,
                        #           action[4]: 4, action[5]: 5, action[6]: 6}
                        # action[::-1].sort()
                        is_epsi = True
                        action = np.argmax(agent.get_qs(current_state))
                    else:  # Not a epsilon move
                        if np.random.random() > .5:  # Do a totally random move
                            action = random.choice(list(BOARD.get_valid_moves()))

                        else:
                            # Select a move from the list of better moves. If one is not available do something
                            # totally random
                            optimal_moves = len(player_1_block_loc) + len(player_2_block_loc)

                            if optimal_moves < optimal_percent:  # If there is a better move do it
                                if len(player_2_block_loc) > 0:  # Move is getting a win
                                    action = random.choice(list(player_2_block_loc))[1]
                                else:  # Move is blocking a win
                                    action = random.choice(list(player_1_block_loc))[1]

                            else:  # No optimal moves, do a random move
                                action = random.choice(list(BOARD.get_valid_moves()))

                        is_epsi = False

                    if is_epsi:  # epsilon gives a invalid move

                        if action not in BOARD.get_valid_moves():
                            # While we keep doing invalid moves, punish and find new input
                            # print('Random Move')

                            agent.update_replay_memory(
                                (current_state, action, -env.ILLEGAL_MOVE_PENALTY,
                                 current_state, done))
                            agent.train(done)
                            episode_reward -= env.ILLEGAL_MOVE_PENALTY
                            specific_rewards[-env.ILLEGAL_MOVE_PENALTY][episode] += 1
                            action = random.choice(list(BOARD.get_valid_moves()))

                    elif action not in BOARD.get_valid_moves():  # Random move is invalid

                        if agent_training != 'USER':
                            agent.update_replay_memory((current_state, action, -env.ILLEGAL_MOVE_PENALTY,
                                                        current_state, done))
                            action = random.choice(list(BOARD.get_valid_moves()))
                            agent.train(done)
                            episode_reward -= env.ILLEGAL_MOVE_PENALTY
                            specific_rewards[-env.ILLEGAL_MOVE_PENALTY][episode] += 1

                new_state, reward, done = env.step(action, agent)

                before_loss[0], before_loss[1], before_loss[2] = current_state, action, new_state
                episode_reward += reward

                last_move = action

                if agent_training != 'USER':
                    agent.update_replay_memory((current_state, action, reward, new_state, done))
                    agent.train(done)

                if reward == env.GAME_WIN_REWARD_H or reward == env.GAME_WIN_REWARD_V or \
                        reward == env.GAME_WIN_REWARD_D:
                    winner = 'AGENT WIN'
                elif reward == -env.GAME_DRAW_PENALTY:
                    winner = 'DRAW'

            else:  # Bot goes 2nd
                if training_mode == 'RANDOM' or training_mode == 'USER' or training_mode == 'ADVANCED':
                    if training_mode == 'ADVANCED':
                        action = bot.action(player_2_block_loc, player_1_block_loc)

                    else:
                        action = bot.action()

                else:
                    if np.random.random() > 0.5:
                        action = np.argmax(bot.predict(current_state.reshape((1, 6, 7, 1))))
                        if action not in BOARD.get_valid_moves():
                            action = random.choice(list(BOARD.get_valid_moves()))

                    else:
                        action = random.choice(list(BOARD.get_valid_moves()))

                new_state, reward, done = env.step(action, bot)

                # Adding the bots moves for additional training
                if agent_training != 'USER':
                    if reward == env.GAME_WIN_REWARD_H or reward == env.GAME_WIN_REWARD_V or \
                            reward == env.GAME_WIN_REWARD_D:
                        agent.update_replay_memory(
                            (last_player_state, last_move, -env.GAME_LOSS_PENALTY, new_state, done))
                        agent.update_replay_memory((current_state, action, reward, new_state, done))
                        agent.train(done)
                        episode_reward -= env.GAME_LOSS_PENALTY

                    else:
                        agent.update_replay_memory((current_state, last_move, reward, new_state, done))
                        if reward == -env.GAME_DRAW_PENALTY:
                            episode_reward -= env.GAME_DRAW_PENALTY
                        agent.train(done)

                if reward == env.GAME_WIN_REWARD_H or reward == env.GAME_WIN_REWARD_V or \
                        reward == env.GAME_WIN_REWARD_D:
                    winner = 'BOT WIN'

                elif reward == -env.GAME_DRAW_PENALTY:
                    winner = 'DRAW'

        if player_turn == turn:
            last_player_state = current_state.copy()

        # Error testing code

        # print(f'\n{BOARD}')
        # print(f'Move by {"Agent" if player_turn == turn else "Bot"}')
        # print(f'Action was {action}')
        # print(f'Reward was {reward} which is {results[reward]}')
        # print(f'Block Loc 1 is {player_1_block_loc} ||| Block Loc 2 is {player_2_block_loc}')
        # print(f'Rolling reward count {episode_reward}')
        # print('\n------------------------\n')
        #
        # # To give time to analyze step by step
        # to_continue = input()

        turn *= -1
        current_state = new_state
        step += 1
        reward_sum += episode_reward
        if turn == player_turn:
            specific_rewards[reward][episode] += 1

    # # Printing episode results
    # print(f'Episode ended in a {results[reward]}')
    # print(f'\n{BOARD}')
    for entry in specific_rewards:
        specific_rewards[entry].append(0)
    ep_rewards_agent.append(episode_reward)
    # ep_rewards_bot.append(reward)

    print(f'Episode #{episode}, Result: {results[reward]}\nReward: {episode_reward}, Steps: {step}\nEpsilon {epsilon}')
    counting[winner] += 1
    print(reward_sum)

    # ep_rewards.append(episode_reward)
    if not episode % AGGREGATE_STATS_EVERY or episode == 1:
        average_reward = sum(ep_rewards_agent[-AGGREGATE_STATS_EVERY:]) / len(ep_rewards_agent[-AGGREGATE_STATS_EVERY:])
        #     min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
        #     max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
        #     agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward,
        #                                    reward_max=max_reward, epsilon=epsilon)

        agent.model.save(
            f'{MODEL_NAME}/EPISODE_{episode+6000}___AVG_EPISODE_REWARD_{average_reward:_>7.2f}_TRAINING_1')

    # if not episode % LOWER_DECAY or episode == 1:
    #     if epsilon > MIN_EPSILON:
    epsilon *= EPSILON_DECAY
    epsilon = max(MIN_EPSILON, epsilon)

    if episode % 10_000 == 0:  # Reset epsilon every so often
        epsilon = 1

    if episode == 10_000:  # Changing from the standard random to advanced
        bot = AdvancedBot()

print(ep_rewards_agent)

print(counting)
x_1 = range(1, EPISODES + 1)
x_2 = range(1, EPISODES + 1)

plt.plot(x_1, ep_rewards_agent, label='Episode Rewards')
plt.title('Rewards over Episodes')
plt.xlabel('Episode Count')
plt.ylabel('Reward Gained')
plt.show()

plt.figure(figsize=(12, 8))
for reward_type in specific_rewards:
    if reward_type == -env.GAME_DRAW_PENALTY or reward_type == -env.GAME_LOSS_PENALTY:
        continue
    plt.plot(x_2, specific_rewards[reward_type][1:EPISODES + 1], label={results[reward_type]})
plt.title('Specific Reward Counts Over Episodes')
plt.ylabel('Reward Count')
plt.xlabel('Episode')
plt.legend()
plt.show()

plt.figure(figsize=(12, 8))
for reward_type in specific_rewards:
    if reward_type == -env.GAME_DRAW_PENALTY or reward_type == -env.GAME_LOSS_PENALTY:
        continue
    plt.bar(x_2, specific_rewards[reward_type][1:EPISODES + 1], label={results[reward_type]})
plt.title('Specific Reward Counts Over Episodes')
plt.ylabel('Reward Count')
plt.xlabel('Episode')
plt.legend()
plt.show()

labels = ['Horizontal Win', 'Vertical Win', 'Diagonal Win', 'Bot Win', 'Draw']
values = [specific_rewards[env.GAME_WIN_REWARD_H], specific_rewards[env.GAME_WIN_REWARD_V],
          specific_rewards[env.GAME_WIN_REWARD_D], counting['BOT WIN'], counting['DRAW']]

plt.ylabel('# of Wins')
plt.xlabel('Type of Win')
plt.show()
