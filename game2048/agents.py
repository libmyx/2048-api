import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, BatchNormalization
import numpy as np
import random
from collections import deque
from keras.optimizers import RMSprop, Adam
import time
import os
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten
from keras.optimizers import Adam
from keras import layers
from collections import deque
import numpy as np
import random
from game2048.game import Game
from game2048.displays import Display, IPythonDisplay
from keras.models import load_model
import math
import tensorflow as tf


class Agent:
    '''Agent Base.'''

    def __init__(self, game, display=None):
        self.game = game
        self.display = display

    def play(self, max_iter=np.inf, verbose=False):
        n_iter = 0
        while (n_iter < max_iter) and (not self.game.end):
            direction = self.step()
            self.game.move(direction)
            n_iter += 1
            if verbose:
                print("Iter: {}".format(n_iter))
                print("======Direction: {}======".format(
                    ["left", "down", "right", "up"][direction]))
                if self.display is not None:
                    self.display.display(self.game)

    def step(self):
        direction = int(input("0: left, 1: down, 2: right, 3: up = ")) % 4
        return direction


class RandomAgent(Agent):

    def step(self):
        direction = np.random.randint(0, 4)
        return direction


class ExpectiMaxAgent(Agent):

    def __init__(self, game, display=None):
        if game.size != 4:
            raise ValueError(
                "`%s` can only work with game of `size` 4." % self.__class__.__name__)
        super().__init__(game, display)
        from .expectimax import board_to_move
        self.search_func = board_to_move

    def step(self):
        direction = self.search_func(self.game.board)
        return direction

def get_grids_next_step(grid):
    #Returns the next 4 states s' from the current state s
        
    grids_list = []
        
    for movement in range(4):
        grid_before = grid.copy()
        env1 = Game(4, random=False, enable_rewrite_board=True)
        env1.board = grid_before
        try:
            _ = env1.move(movement) 
        except:
            pass
        grid_after = env1.board
        grids_list.append(grid_after)
        
    return grids_list

#Converting observations in range (0,1) using log(n)/log(max) so that gradients don't vanish
def process_log(observation):
    observation_temp = np.where(observation <= 0, 1, observation) 
    processed_observation = np.log2(observation_temp)/np.log2(65536)
    return processed_observation.reshape(1,4,4)

class DQNAgent(Agent):
    def __init__(self,env,display=None):
        self.game = env
        self.display = display
        #Defining the hyperparameters for the model
        self.env=env.board
        #The replay memory will be stored in a Deque
        self.memory=deque(maxlen=2000)
        self.gamma=0.90
        #self.epsilon=1.0
        self.epsilon_min=0
        self.epsilon_decay=0.995
        self.learning_rate=0.005
        self.epsilon=0
        self.tau=0.125
        #We use 2 models to prevent Bootstrapping
        #self.model=self.create_model()
        #self.target_model=self.create_model()
        self.model = keras.models.load_model('result1/trial num-4767.h5f')
        self.target_model = keras.models.load_model('result1/trial num-4767.h5f')
        
        
    def create_model(self):
        model=Sequential()
        state_shape=4
                
        model.add(Flatten(input_shape=(4,4)))
        model.add(Dense(units=1024,activation="relu"))
        model.add(Dense(units=512,activation="relu"))
        model.add(Dense(units=256,activation="relu"))
        model.add(Dense(units=4))
        model.compile(loss="mean_squared_error",optimizer=Adam(lr=self.learning_rate))
        print(model.summary())
        return model
                  
    def act(self,state):
        #Epsilon value decays as model gains experience
        self.epsilon*=self.epsilon_decay
        self.epsilon=max(self.epsilon_min,self.epsilon)
        if np.random.random()<self.epsilon:
                  return np.random.randint(0,4)
        else:
            #Getting the 4 future states
            allstates=get_grids_next_step(state)
            
            res=[]
            for i in range(len(allstates)):
                if (allstates[i]==state).all():
                    res.append(0)
                else:
                    processed_state=process_log(allstates[i])
                    #max from the 4 future Q_Values is appended in res
                    res.append(np.max(self.model.predict(processed_state)))
            
            a=self.model.predict(process_log(state))
            #Final Q_Values are the sum of Q_Values of current state andfuture states
            final=np.add(a,res)
            
            return np.argmax(final)
    
    def remember(self, state, action, reward, new_state, done):
        #Replay Memory stores tuple(S, A, R, S')
        self.memory.append([state, action, reward, new_state, done])

    def replay(self):
        batch_size=32
        if len(self.memory)<batch_size:
            return
        samples=random.sample(self.memory,batch_size)
        for sample in samples:
            
            state,action,reward,neself.w_state,done=sample
            
            target=self.target_model.predict(process_log(state))
            
            
            if done:
                target[0][action]=reward
            else:
                #Bellman Equation for update
                Q_future=max(self.target_model.predict(process_log(neself.w_state))[0])
                
                #The move which was selected, its Q_Value gets updated
                target[0][action]=reward+Q_future*self.gamma
            self.model.fit((process_log(state)),target,epochs=1,verbose=0)
                  
                  
    def target_train(self):
        weights=self.model.get_weights()
        target_weights=self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i]=weights[i]*self.tau+target_weights[i]*(1-self.tau)
        self.target_model.set_weights(target_weights)
                  
                  
    def save_model(self,fn):
        self.model.save(fn)

    def play(self, max_iter=np.inf, verbose=False):
        n_iter = 0
        while (n_iter < max_iter) and (not self.game.end):
            direction = self.act(self.game.board)
            self.game.move(direction)
            n_iter += 1
            if verbose:
                print("Iter: {}".format(n_iter))
                print("======Direction: {}======".format(
                    ["left", "down", "right", "up"][direction]))
                if self.display is not None:
                    self.display.display(self.game)

def change_values(X):
    power_mat = np.zeros(shape=(4,4,16),dtype=np.float32)
    for i in range(4):
        for j in range(4):
            if(X[i][j]==0):
                power_mat[i][j][0] = 1.0
            else:
                power = int(math.log(X[i][j],2))
                power_mat[i][j][power] = 1.0
    return power_mat

class CNNAgent(Agent):

    def __init__(self, game, display=None):
        self.model = load_model('/home/myx/learning_affairs/EE228/project/2048-api/CNN.h5')
        self.game = game
        self.display = display

    def step(self):
        direction = self.model.predict(change_values(self.game.board).reshape(1,4,4,16))
        direction = np.argmax(direction)
        return direction

class DCNNAgent(Agent):
    
    def Board8(self, image):
        tempB = np.zeros([4, 4])
        for i in range(16):#1024 -> 2^10 -> 10
            t_num = 0
            if image[i//4][i%4] != 0:
                t_num = int(math.log2(image[i//4][i%4]))
            tempB[i//4][i%4] = t_num
        g0 = tempB      
        g1 = g0[::-1,:] 
        g2 = g0[:,::-1] 
        g3 = g2[::-1,:] 
        r0 = g0.swapaxes(0,1) 
        r1 = r0[::-1,:] 
        r2 = r0[:,::-1]
        r3 = r2[::-1,:]

        inputB = np.zeros([8, 4, 4, 16])
        gcount = 0
        for g in [g0,r2,g3,r1,g2,r0,g1,r3]:
            for i in range(16):
                inputB[gcount][i//4][i%4][int(g[i//4][i%4])] = 1
            gcount += 1
        P = np.zeros([4], dtype=np.float32)
        Pcount = np.zeros([4])

        prev = self.sess.run(self.p, feed_dict={self.x_image:inputB})

        self.B0(prev[0][:], P, Pcount)
        self.B1(prev[1][:], P, Pcount)
        self.B2(prev[2][:], P, Pcount)
        self.B3(prev[3][:], P, Pcount)
        self.B4(prev[4][:], P, Pcount)
        self.B5(prev[5][:], P, Pcount)
        self.B6(prev[6][:], P, Pcount)
        self.B7(prev[7][:], P, Pcount)

        return P, Pcount

    def B0(self, pre, P, Pcount):
        for i in range(4):
            P[i] += pre[i]
        Pcount[np.argmax(pre)] += 1

    def B1(self, pre, P, Pcount):
        tempP = np.zeros([4])
        tempP[0] = pre[1]
        tempP[1] = pre[2]
        tempP[2] = pre[3]
        tempP[3] = pre[0]
        Pcount[np.argmax(tempP)] += 1
        for i in range(4):
            P[i] += tempP[i]

    def B2(self,pre, P, Pcount):
        tempP = np.zeros([4])
        tempP[0] = pre[2]
        tempP[1] = pre[3]
        tempP[2] = pre[0]
        tempP[3] = pre[1]
        Pcount[np.argmax(tempP)] += 1
        for i in range(4):
            P[i] += tempP[i]

    def B3(self, pre, P, Pcount):
        tempP = np.zeros([4])
        tempP[0] = pre[3]
        tempP[1] = pre[0]
        tempP[2] = pre[1]
        tempP[3] = pre[2]
        Pcount[np.argmax(tempP)] += 1
        for i in range(4):
            P[i] += tempP[i]

    def B4(self, pre, P, Pcount):
        tempP = np.zeros([4])
        tempP[0] = pre[0]
        tempP[1] = pre[3]
        tempP[2] = pre[2]
        tempP[3] = pre[1]
        Pcount[np.argmax(tempP)] += 1
        for i in range(4):
            P[i] += tempP[i]

    def B5(self, pre, P, Pcount):
        tempP = np.zeros([4])
        tempP[0] = pre[3]
        tempP[1] = pre[2]
        tempP[2] = pre[1]
        tempP[3] = pre[0]
        Pcount[np.argmax(tempP)] += 1
        for i in range(4):
            P[i] += tempP[i]

    def B6(self, pre, P, Pcount):
        tempP = np.zeros([4])
        tempP[0] = pre[2]
        tempP[1] = pre[1]
        tempP[2] = pre[0]
        tempP[3] = pre[3]
        Pcount[np.argmax(tempP)] += 1
        for i in range(4):
            P[i] += tempP[i]

    def B7(self, pre, P, Pcount):
        tempP = np.zeros([4])
        tempP[0] = pre[1]
        tempP[1] = pre[0]
        tempP[2] = pre[3]
        tempP[3] = pre[2]
        Pcount[np.argmax(tempP)] += 1
        for i in range(4):
            P[i] += tempP[i]
    
    def __init__(self, game, display=None):
        self.game = game
        self.display = display
        
        tf.reset_default_graph()

        self.w_1 = 222
        self.w_2 = 222
        self.w_3 = 222
        self.w_4 = 222
        self.w_5 = 222

        self.x_image = tf.placeholder(tf.float32, [None, 4, 4, 16])
        self.W_conv1 = tf.Variable(tf.truncated_normal([2,2,16,self.w_1], stddev=0.1))
        self.h_conv1 = tf.nn.conv2d(self.x_image, self.W_conv1, strides=[1,1,1,1], padding='SAME')
        self.b_conv1 = tf.Variable(tf.constant(0.1, shape=[self.w_1]))
        self.relu1 = tf.nn.relu(self.h_conv1 + self.b_conv1)

        self.W_conv2 = tf.Variable(tf.truncated_normal([2,2,self.w_1,self.w_2], stddev=0.1))
        self.h_conv2 = tf.nn.conv2d(self.relu1, self.W_conv2, strides=[1,1,1,1], padding='SAME')
        self.b_conv2 = tf.Variable(tf.constant(0.1, shape=[self.w_2]))
        self.relu2 = tf.nn.relu(self.h_conv2 + self.b_conv2)

        self.W_conv3 = tf.Variable(tf.truncated_normal([2,2,self.w_2,self.w_3], stddev=0.1))
        self.h_conv3 = tf.nn.conv2d(self.relu2, self.W_conv3, strides=[1,1,1,1], padding='SAME')
        self.b_conv3 = tf.Variable(tf.constant(0.1, shape=[self.w_3]))
        self.relu3 = tf.nn.relu(self.h_conv3 + self.b_conv3)

        self.W_conv4 = tf.Variable(tf.truncated_normal([2,2,self.w_3,self.w_4], stddev=0.1))
        self.h_conv4 = tf.nn.conv2d(self.relu3, self.W_conv4, strides=[1,1,1,1], padding='SAME')
        self.b_conv4 = tf.Variable(tf.constant(0.1, shape=[self.w_4]))
        self.relu4 = tf.nn.relu(self.h_conv4 + self.b_conv4)

        self.W_conv5 = tf.Variable(tf.truncated_normal([2,2,self.w_4,self.w_5], stddev=0.1))
        self.h_conv5 = tf.nn.conv2d(self.relu4, self.W_conv5, strides=[1,1,1,1], padding='SAME')
        self.b_conv5 = tf.Variable(tf.constant(0.1, shape=[self.w_5]))
        self.relu5 = tf.nn.relu(self.h_conv5 + self.b_conv5)

        self.relu5_flat = tf.reshape(self.relu5, [-1, 4*4*self.w_5])

        self.w0 = tf.Variable(tf.zeros([4*4*self.w_5, 4]))
        self.b0 = tf.Variable(tf.zeros([4]))
        self.p = tf.nn.softmax(tf.matmul(self.relu5_flat, self.w0) + self.b0)

        self.t = tf.placeholder(tf.float32, [None, 4])
        self.loss = -tf.reduce_sum(self.t * tf.log(tf.clip_by_value(self.p,1e-10,1.0)))
        self.train_step = tf.train.AdamOptimizer().minimize(self.loss)

        self.correct_prediction = tf.equal(tf.argmax(self.p, 1), tf.argmax(self.t, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep=None)
        
        files = '/home/myx/learning_affairs/EE228/project/2048-api/700000000'
        self.saver.restore(self.sess, files)
    
    def step(self):
        prev_board = self.game.board
        image = [[prev_board[i][j] for j in range(4)] for i in range(4)]
        P, Pcount = self.Board8(image)
        counter = 0
        while True:
            counter += 1
            select = -1
            pmax = -1
            for i in range(4):
                if pmax < Pcount[i]:
                    pmax = Pcount[i]
                    select = i
                elif pmax == Pcount[i] and P[select] < P[i]:
                    pmax = Pcount[i]
                    select = i
            Pcount[select] = -1
            new_game = Game(4, enable_rewrite_board=True)
            new_game.board = prev_board
            new_game.move(3-select)
            new_board = new_game.board
            isMoved = not (prev_board==new_board).all()
            if isMoved: break
            if counter == 4:
                select = np.argmax(Pcount)
                break
        
        return (3-select)
