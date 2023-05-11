import tensorflow as tf
import numpy as np
#
position_0 = [
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0]
]

position_1 = [
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [-1, -1, 0, 0, 0, 0, 0],
    [1, -1, 0, 1, 1, 1, 0],
    [1, 1, 1, -1, -1, -1, 1]
]

position_2 = [
    [0, 0, 0, 0, 0, 0, 0],
    [0, -1, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0],
    [-1, -1, 0, 1, 0, 0, 0],
    [1, -1, 1, -1, 1, 1, 0],
    [1, 1, 1, -1, -1, -1, 1]
]

position_3 = [
    [0, 0, -1, 0, 0, 0, 0],
    [0, -1, 1, 0, 0, 0, 0],
    [0, 1, -1, 0, 0, 0, 0],
    [-1, -1, 1, 1, 0, 0, 0],
    [1, -1, 1, -1, 1, 1, 0],
    [1, 1, 1, -1, -1, -1, 1]
]

boards = [
    position_0,
    position_1,
    position_2,
    position_3
]

bot = tf.keras.models.load_model(input('Path to model'))
print(bot.summary)
for position in boards:
    action = bot.predict(np.array(position).reshape((1, 6, 7, 1)))
    print(f'Action taken was {action} on this board\n{position}')

# agent = tf.keras.models.load_model('V12-256x2/EPISODE_400___AVG_EPISODE_REWARD_-53608.18')
# print(agent)


