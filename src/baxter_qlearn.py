#!/usr/bin/env python

import json
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import sgd


class Catch(object):
    def __init__(self):
        self.reset()

    def _update_state(self, action):
        """
        Input: action and self.state
        Ouput: new self.state
        """
        
        # based on the action, command Baxter to move
        # call observe and get a new image
        # set self.state equals to the new state
        pass

    def _is_over(self):
        # decide if action is over or not
        if 1:
            return True
        else:
            return False

    def observe(self):
        # 1) get an image
        # 2) do necessary preprocessing like resizing etc
        # 3) make it a vector, and return canvas.reshape((1, -1))
        pass

    def act(self, action):
        self._update_state(action)
        reward = 0 # get reward
        game_over = self._is_over()
        return self.observe(), reward, game_over

    def reset(self):
        # 1) move robot into initial position
        # 2) spawn object
        # 3) get an image, and give it to self.state
        pass

class ExperienceReplay(object):
    def __init__(self, max_memory=100, discount=.9):
        self.max_memory = max_memory
        self.memory = list()
        self.discount = discount

    def remember(self, states, game_over):
        # memory[i] = [[state_t, action_t, reward_t, state_t+1], game_over?]
        self.memory.append([states, game_over])
        if len(self.memory) > self.max_memory:
            del self.memory[0]

    def get_batch(self, model, batch_size=10):
        len_memory = len(self.memory)
        num_actions = model.output_shape[-1]
        env_dim = self.memory[0][0][0].shape[1] # input dim (1,-1)
        batch_size = min(len_memory, batch_size) # most of time, no effect
        inputs = np.zeros((batch_size, env_dim))
        targets = np.zeros((batch_size, num_actions))
        for i, idx in enumerate(np.random.randint(0, len_memory,
                                                  size=batch_size)):
            state_t, action_t, reward_t, state_tp1 = self.memory[idx][0]
            game_over = self.memory[idx][1]

            inputs[i, :] = state_t
            # There should be no target values for actions not taken.
            # Thou shalt not correct actions not taken #deep
            targets[i] = model.predict(state_t)[0]
            Q_sa = np.max(model.predict(state_tp1)[0])
            if game_over:  # if game_over is True
                targets[i, action_t] = reward_t
            else:
                # reward_t + gamma * max_a' Q(s', a')
                targets[i, action_t] = reward_t + self.discount * Q_sa
        return inputs, targets


if __name__ == "__main__":
    # parameters
    epsilon = .1  # exploration
    num_actions = 3*7  # [up, hold, down] x 7 joints
    epoch = 100
    max_memory = 500
    hidden_size = 100
    batch_size = 50
    img_h = 100
    img_w = 100
    
    model = Sequential()
    model.add(Dense(hidden_size, input_shape=(img_h * img_w), activation='relu'))
    model.add(Dense(hidden_size, activation='relu'))
    model.add(Dense(num_actions))
    model.compile(sgd(lr=.2), "mse")

    # If you want to continue training from a previous model, just uncomment the line bellow
    # model.load_weights("../model/model.h5")

    # Define environment/game
    env = Catch()

    # Initialize experience replay object
    exp_replay = ExperienceReplay(max_memory=max_memory)

    # Train
    win_cnt = 0
    for e in range(epoch):
        loss = 0.
        env.reset()
        game_over = False
        # get initial input
        input_t1 = env.observe()

        while not game_over:
            input_t0 = input_t1
            # get next action
            if np.random.rand() <= epsilon:
                action = np.random.randint(0, num_actions, size=1)
            else:
                q = model.predict(input_t0)
                action = np.argmax(q[0])

            # apply action, get rewards and new state
            input_t1, reward, game_over = env.act(action)
            if reward == 1:
                win_cnt += 1

            # store experience
            exp_replay.remember([input_t0, action, reward, input_t1], game_over)

            # adapt model
            inputs, targets = exp_replay.get_batch(model, batch_size=batch_size)

            # update loss 
            loss += model.train_on_batch(inputs, targets)

        print("Epoch {:03d}/{:03d} | Loss {:.4f} | Win count {:03d}".format(e, epoch, loss, win_cnt))

    # Save trained model weights and architecture, this will be used by the visualization code
    model.save_weights("../model/model.h5", overwrite=True)
    with open("../model/model.json", "w") as outfile:
        json.dump(model.to_json(), outfile)
