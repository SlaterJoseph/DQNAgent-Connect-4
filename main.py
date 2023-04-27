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
MODEL_NAME = '256x2'
MINIBATCH_SIZE = 64
DISCOUNT = 0.99
UPDATE_TARGET_EVERY = 5

MIN_REWARD = -200
MEMORY_FRACTION = 0.20

EPISODES = 20_000

epsilon = 1
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001

AGGREGATE_STATS_EVERY = 50
SHOW_PREVIEW = False


class DQNAgent:
    def __init__(self):
        self.model = self.create_model()  # main model

        self.target_model = self.create_model()  # target model
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        self.tensorboard = ModifiedTensorBoard(log_dir=f'logs/{MODEL_NAME}-{int(time.time())}')

        self.target_update_counter = 0

    def create_model(self):
        """Creates both the target model and the main model"""
        model = Sequential()
        model.add(InputLayer(input_shape=(6, 7, 1)))  # Dimension of the 2d list with 1 for greyscale
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        # model.add(MaxPooling2D(2, 2))
        model.add(Dropout(0.2))
        model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
        # model.add(MaxPooling2D(2, 2))
        model.add(Dropout(0.2))

        model.add(Flatten())
        model.add(Dense(128))
        model.add(Dropout(0.2))
        model.add(Dense(256))
        model.add(Dropout(0.2))
        model.add(Dense(7, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=0.001), metrics=['accuracy'])
        return model

    def update_replay_memory(self, transition):
        """Updates replay memory"""
        self.replay_memory.append(transition)

    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape) / 255)[0]

    def punish(self, penalty, state, action):
        q_values = self.get_qs(state)
        q_values[action] = penalty

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

            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            X.append(current_state)
            y.append(current_qs)

        self.model.fit(np.array(X)/255, np.array(y), batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False,
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
        self._log_write_dir = os.path.join(self.log_dir, MODEL_NAME)

    def set_model(self, model):
        pass

    def on_epoch_end(self, epoch, logs=None):
        self.update_statues(**logs)

    def on_batch_end(self, batch, logs=None):
        pass

    def on_train_end(self, _):
        pass

    def update_stats(self, **stats):
        self._write_logs(stats, self.step)

    def _write_logs(self, logs, index):
        with self.writer.as_default():
            for name, value in logs.items():
                tf.summary.scalar(name, value, step=index)
                self.step += 1
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
        self.valid_cols = [x for x in range(7)]

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

    def __str__(self):
        base = str()
        for row in self.board:
            base += f'{row} \n'
        return base

    def get_valid_moves(self):
        return self.valid_cols

    def board_state(self):
        """Return a copy of the board"""
        return self.board.copy()

    def col_full(self, col):
        """Lets us know if a column is full"""
        if self.board[0][col] != 0:
            print(f'Removing {col} from {self.valid_cols}')
            self.valid_cols.remove(col)
            return True
        return False

    def drop(self, col, player_id):
        """
        Drop the piece into the proper column
        -2 - Draw
        -1 - Illegal move
        1 - Winner Found
        2 - Basic move
        """
        # print(f'In drop: col = {col} : player_id = {player_id}')

        if self.size == 42:  # Board full, draw
            return -2
        if self.col_full(col):  # If full, return -1
            return -1

        safe_row = -1
        for x in range(0, 6):
            # print(f'col - {col} : row - {x}')
            # print(f'Square = {self.board[x][col]}\n')
            if self.board[x][col] != 0:
                safe_row = x - 1
                # print(f'Safe Row = {safe_row}')
                break

        self.board[safe_row][col] = player_id
        # print(self)

        self.size += 1  # Increase the piece count
        return self.found_winner(safe_row, col)  # Return if a winner is found

    def found_winner(self, x, y):
        """
        Checks if the most recent move found a winner
        1 - Winner found
        2 - No winner found
        """

        def across():  # Row check
            row = self.board[x]
            for loc in range(3, 7):
                cell_1, cell_2, cell_3, cell_4 = row[loc - 3], row[loc - 2], row[loc - 1], row[loc]
                if cell_1 == cell_2 == cell_3 == cell_4 and cell_1 != 0:
                    return True
                return False

        def up():  # Col check
            for loc in range(3, 6):
                cell_1, cell_2, cell_3, cell_4 = self.board[loc - 3][y], self.board[loc - 2][y], \
                                                 self.board[loc - 1][y], self.board[loc][y]
                # print(f'{cell_1}, {cell_2}, {cell_3}, {cell_4}')
                if cell_1 == cell_2 == cell_3 == cell_4 and cell_1 != 0:
                    return True
            return False

        def diagonal_1():  # 0,0 to 5,6
            chip = self.board[x][y]
            if chip == 0:  # Edge case
                return False

            x_down, y_down = x - 1, y - 1  # To top left corner
            count = 1
            while x_down > -1 and y_down > -1 and self.board[x_down][y_down] == chip:
                # print(f'{board[x_down][y_down]}: {x_down}, {y_down}')
                count += 1
                x_down -= 1
                y_down -= 1

            x_up, y_up = x + 1, y + 1  # To bottom right corner
            while x_up < 6 and y_up < 7 and self.board[x_up][y_up] == chip:
                # print(f'{board[x_up][y_up]}: {x_up}, {y_down}')
                count += 1
                x_up += 1
                y_up += 1

            return count >= 4

        def diagonal_2():  # 0,6 to 5,0
            chip = self.board[x][y]
            if chip == 0:  # edge case
                return False

            x_down, y_up = x - 1, y + 1  # To top left corner
            count = 1
            while x_down > -1 and y_up < 7 and self.board[x_down][y_up] == chip:
                # print(f'{board[x_down][y_down]}: {x_down}, {y_down}')
                count += 1
                x_down -= 1
                y_up += 1

            x_up, y_down = x + 1, y - 1  # To bottom right corner
            while x_up < 6 and y_down > -1 and self.board[x_up][y_down] == chip:
                # print(f'{board[x_up][y_down]}: {x_up}, {y_down}')
                count += 1
                x_up += 1
                y_down -= 1

                return count >= 4

            if across() or up() or diagonal_1() or diagonal_2():
                return 1
            else:
                return 2


class RandomBot:
    def __init__(self):
        """Create the player move list and the number id associated with it"""
        self.possible_moves = [x for x in range(7)]

    # def __str__(self):
    #     """For Testing"""
    #     return f'Child: id - {self.id}'

    def reset(self):
        self.possible_moves = [x for x in range(7)]

    def action(self):
        """Makes a completely random move"""
        col = random.choice(self.possible_moves)

        if BOARD.col_full(col):
            self.possible_moves.remove(col)
            self.action()

        return col


class Connect4Env:
    SIZE_ROWS = 6
    SIZE_COLUMNS = 7
    RETURN_IMAGES = True
    # First 4 will be negative
    MOVE_PENALTY = 1
    GAME_LOSS_PENALTY = 300
    GAME_DRAW_PENALTY = 100
    ILLEGAL_MOVE_PENALTY = 1_200  # Want to quickly remove illegal moves
    GAME_WIN_REWARD = 100
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
        bot.reset()
        self.episode_step = 0
        return BOARD.board_state()  # Return the current state (Which is empty)

    def step(self, turn_action, user):
        if user == self.player_1[1]: # Finds if this move was the 1st player or 2nd
            result = BOARD.drop(turn_action, self.player_1[0])  # 1 is the piece
        else:
            result = BOARD.drop(turn_action, self.player_2[0])  # -1 is the piece

        if result == -2:  # Draw
            move_reward = -self.GAME_DRAW_PENALTY
        elif result == -1:  # Illegal Move || THIS CODE SHOULD NEVER BE REACHED
            print('ERROR: INVALID MOVE')
            move_reward = -self.ILLEGAL_MOVE_PENALTY
        elif result == 1 and user == bot:  # Loss
            move_reward = -self.GAME_LOSS_PENALTY
        elif result == 1 and user == agent:  # Win
            move_reward = self.GAME_WIN_REWARD
        else:  # Basic Move
            move_reward = -self.MOVE_PENALTY

        if move_reward == self.GAME_WIN_REWARD or move_reward == -self.GAME_LOSS_PENALTY \
                or move_reward == -self.GAME_DRAW_PENALTY:
            complete = True
        else:
            complete = False

        self.episode_step += 1
        return np.array(BOARD.board_state()), move_reward, complete


agent = DQNAgent()
bot = RandomBot()
env = Connect4Env()
BOARD = Board()

ep_rewards = [-200]

if not os.path.isdir('models'):
    os.makedirs('models')

for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episode'):
    agent.tensorboard.step = episode
    current_state = np.array(env.reset())
    episode_reward = 0
    step = 1
    turn = 1
    done = False
    before_loss = [current_state, None]

    if random.randint(0, 1) == 0:  # Choosing who goes first
        player_1, player_2, env.player_1[1], env.player_2[1] = agent, bot, agent, bot
        print('Player 1 - Agent, Player 2 - Bot')
    else:
        player_1, player_2, env.player_1[1], env.player_2[1] = bot, agent, bot, agent
        print('Player 1 - Bot, Player 2 - Agent')

    while not done:
        if turn == 1: # Odd Moves
            if player_1 == agent:  # Agent goes 1st
                if np.random.random() > epsilon:
                    action = np.argmax(agent.get_qs(current_state, step))
                    is_epsi = True
                else:
                    action = np.random.randint(0, 6)
                    is_epsi = False
                print(f'Turn: {turn} \nAction taken was {action}\nBoard is \n{BOARD}')

                while BOARD.col_full(action):  # While we keep doing invalid moves, punish and find new input
                    agent.punish(-env.ILLEGAL_MOVE_PENALTY, current_state, action)
                    if is_epsi:
                        action = np.argmax(agent.get_qs(current_state))
                        is_epsi = True
                    else:
                        action = np.random.randint(0, 6)
                        is_epsi = False

                new_state, reward, done = env.step(action, agent)
                before_loss[0], before_loss[1] = current_state, action
                episode_reward += reward

                agent.update_replay_memory((current_state, action, reward, new_state, done))
                agent.train(done)
            else:  # Bot goes 1st
                action = bot.action()
                print(f'Turn: {turn} \nAction taken was {action}\nBoard is \n{BOARD}')
                new_state, reward, done = env.step(action, bot)
                if reward == env.GAME_WIN_REWARD:
                    agent.punish(-env.GAME_LOSS_PENALTY, before_loss[0], before_loss[1])


        else:  # Even Moves
            if player_2 == agent:  # Agent goes 2nd
                if np.random.random() > epsilon:
                    action = np.argmax(agent.get_qs(current_state))
                    is_epsi = True
                else:
                    action = np.random.randint(0, 6)
                    is_epsi = False
                print(f'Turn: {turn} \nAction taken was {action}\nBoard is \n{BOARD}')

                while BOARD.col_full(action):  # While we keep doing invalid moves, punish and find new input
                    print(f'{action} column filled, punishing')
                    agent.punish(-env.ILLEGAL_MOVE_PENALTY, current_state, action)
                    if is_epsi:
                        action = np.argmax(agent.get_qs(current_state))
                        is_epsi = True
                    else:
                        action = np.random.randint(0, 6)
                        is_epsi = False

                new_state, reward, done = env.step(action, agent)
                before_loss[0], before_loss[1] = current_state, action
                episode_reward += reward

                agent.update_replay_memory((current_state, action, reward, new_state, done))
                agent.train(done)
            else:  # Bot goes 2nd
                print(f'Turn: {turn} \nAction taken was {action}\nBoard is \n{BOARD}')
                action = bot.action()
                new_state, reward, done = env.step(action, bot)
                if reward == env.GAME_WIN_REWARD:
                    agent.punish(-env.GAME_LOSS_PENALTY, before_loss[0], before_loss[1])

        turn *= -1
        current_state = new_state
        step += 1

    ep_rewards.append(episode_reward)
    if not episode % AGGREGATE_STATS_EVERY or episode == 1:
        average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:]) / len(ep_rewards[-AGGREGATE_STATS_EVERY:])
        min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
        max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
        agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward,
                                       reward_max=max_reward, epsilon=epsilon)

        if min_reward >= MIN_REWARD:
            agent.model.save(agent.model.savef('models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}'
                                              f'avg_{min_reward:_>7.2f}min__{int(time.time())}.model'))

        if epsilon > MIN_EPSILON:
            epsilon *= EPSILON_DECAY
            epsilon = max(MIN_EPSILON, epsilon)


