import numpy as np
import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import keras.backend as K
s = tf.Session()

class env:

    def __init__(self,  distance, neighbour):
        self.demand = [2.0] * 2
        self.td = 2
        self.demand[0] = 0
        self.tp = 0
        self.tl = 1
        self.distance = distance
        self.neighbour = neighbour
        self.state = self.demand + [self.tp] + [self.tl]
        self.dummy = [-1]*4
        self.complete = [0]*4

    def step(self, a):
        action = [int (a/5),(a%5)*0.25]
        if (action[0] == self.tp or action[1] > self.tl or action[1] > self.demand[action[0]]):
            self.state = self.dummy
            return (self.state,-100.0,a)
        else:
            z = self.tp
            self.demand[action[0]] = self.demand[action[0]] - action[1]
            self.td = self.td - action[1]
            self.tl = self.tl - action[1]
            self.tp = action[0]
            if (self.td == 0):
                self.state = self.complete
                return (self.state,100.0, a)
            else:
                if (self.tl == 0.0):
                    self.tl = 1
                    self.tp = 0
            self.state = self.demand + [self.tp] + [self.tl]
            return (self.state ,np.exp(action[1]*5)/(self.distance[z][int (action[0])]),a)

    def reset(self):
        self.demand =  [2.0]*2
        self.td = 2
        self.demand[0] = 0
        self.tp = 0
        self.tl = 1
        self.state =  self.demand + [self.tp] + [self.tl]

class actor_critic:

    def __init__(self):
        self.input = keras.Input(shape=(4,))
        self.l1 = keras.layers.Dense(7, activation='linear')(self.input)
        self.l2 = keras.layers.Dense(7, activation='linear' , use_bias= True)(self.l1)
        self.l3 = keras.layers.Dense(8, activation='linear', use_bias=True)(self.l2)
        self.l4 = keras.layers.Dense(10, activation='linear', use_bias=True)(self.l3)
        '''the actor is here'''
        self.l5 = keras.layers.Dense(10, activation='softmax')(self.l4)
        self.actor = keras.models.Model(inputs=self.input, outputs=self.l5)
        self.z = tf.constant([np.random.uniform(0, 1)])
        RMSprop = keras.optimizers.RMSprop(lr=0.05, rho=0.9, epsilon=None, decay=0.0)

        self.inputq1 = keras.Input(shape=(4,))
        self.l2q = keras.layers.Dense(5, activation='tanh', use_bias=True, bias_initializer=keras.initializers.Constant(0.2))(self.inputq1)
        self.l3q = keras.layers.Dense(7, activation='tanh', use_bias=True,bias_initializer=keras.initializers.Constant(1))(self.l2q)
        self.l4q = keras.layers.Dense(10, activation='tanh', use_bias=True, bias_initializer=keras.initializers.Constant(-1.1223))(self.l3q)
        self.l5q = keras.layers.Dense(6, activation='tanh', use_bias=True, bias_initializer=keras.initializers.Constant(1))(self.l4q)
        self.l6q = keras.layers.Dense(3, activation='tanh', use_bias=True, bias_initializer=keras.initializers.Constant(-1.0))(self.l5q)
        self.l7q = keras.layers.Dense(1, activation='tanh', use_bias=True,bias_initializer=keras.initializers.Constant(2.0))(self.l6q)
        self.inputq = keras.Input(shape=(1,))
        self.l8q = keras.layers.Dense(5, activation = 'tanh' , use_bias= True, bias_initializer=keras.initializers.Constant(-1.5))(self.inputq)
        self.l9q = keras.layers.Dense(7, activation='tanh', use_bias=True, bias_initializer=keras.initializers.Constant(0.7))(self.l8q)
        self.l10q = keras.layers.Dense(4, activation='tanh', use_bias=True, bias_initializer=keras.initializers.Constant(0.3))(self.l9q)
        self.l11q = keras.layers.Dense(1, activation='tanh', use_bias=True, bias_initializer=keras.initializers.Constant(0.6))(self.l10q)
        self.l12q = keras.layers.Concatenate(axis = -1)([self.l7q,self.l11q])
        self.l13q = keras.layers.Dense(4,activation='tanh' , use_bias= True, bias_initializer=keras.initializers.Constant(3))(self.l12q)
        self.l15q = keras.layers.Dense(3, activation='linear', use_bias=True, bias_initializer=keras.initializers.Constant(2))(self.l13q)
        self.l16q = keras.layers.Dense(1,activation= 'exponential' ,use_bias = True,bias_initializer=keras.initializers.Constant(1), dtype= tf.float64)(self.l15q)
        self.criticq = keras.models.Model(inputs=[self.inputq1,self.inputq], outputs=self.l16q)
        self.reward = tf.constant([0.0])
        self.next = tf.constant([0.0])

        self.aloss = tf.multiply(tf.log(tf.reduce_max(self.actor.output)), tf.stop_gradient(self.z))
        self.criticq.compile(optimizer= keras.optimizers.RMSprop(lr = 0.05) , loss= 'mean_squared_error')

    def ao(self):
        op = keras.optimizers.RMSprop(lr=0.001, epsilon=0.1, rho=0.99)
        updates = op.get_updates(self.actor.trainable_weights,[], self.aloss)
        return(K.function([self.input], [], updates = updates))

    def action(self,input):
        return s.run((tf.argmax(self.actor.predict(input), axis=-1))[0])
        #return s.run(tf.random.categorical(tf.log(self.actor.predict(input)),1))[0][0]

class RLagent:

    def __init__(self, distance, neighbour):
        self.e = env(distance, neighbour)
        self.agent = actor_critic()
        self.ao = self.agent.ao()
        '''self.crvo = self.agent.cov()'''

    def episodes(self):
        ans = [(self.e.state.copy(), 0,0)]
        r = 0
        p = 0
        while  not np.array_equal(self.e.state,self.e.dummy) and not np.array_equal(self.e.state, self.e.complete):
             p = self.agent.action(np.array([self.e.state]))
             print(self.e.state, p)
             ans = ans + [self.e.step(p)]
             r = r + ans[-1][1]
        self.e.reset()
        return (ans, r)

    def train1(self, e):
        episode = e[0]
        for i in range(0, len(episode) - 2):
            self.agent.next = self.agent.criticq.predict([np.array([episode[i+1][0]]),np.array([[episode[i+2][2]]])])
            self.agent.q = tf.identity(self.agent.criticq.predict([np.array([episode[i][0]]),np.array([[episode[i+1][2]]])]))
            self.agent.reward = tf.constant([episode[i+1][1]] , dtype= tf.float64)
            self.agent.z = tf.identity(self.agent.q)
            self.ao([np.array([episode[i][0]])])
            #self.crvo([np.array([episode[i][0]])])#
            self.agent.criticq.fit([np.array([episode[i][0]]), np.array([[episode[i + 1][2]]])], s.run(self.agent.next + self.agent.reward), batch_size= None)
        self.agent.next = tf.constant(np.array([-1.0]), dtype= tf.float64)
        self.agent.q = tf.identity(self.agent.criticq.predict([np.array([episode[len(episode) - 2][0]]), np.array([[episode[len(episode) - 1][2]]])]))
        self.agent.reward = tf.constant([episode[len(episode) - 1][1]], dtype= tf.float64)
        self.ao([np.array([episode[len(episode) - 2][0]])])
        self.agent.criticq.fit([np.array([episode[len(episode) - 2][0]]), np.array([[episode[len(episode) - 1][2]]])],s.run(self.agent.next + self.agent.reward), batch_size=None)

    def train(self, i):
        t = ()
        x = [0] * i
        for j in range(0, i):
             t = self.episodes()
             x[j] = t[1]
             self.train1(t)
             print('#'),
        plt.plot(x)
        plt.show()


if __name__ == '__main__':
    distance = np.array([[0,12],[12,0]])
    neighbors = [[0,1],[1,0]]
    a = RLagent(distance,neighbors)
    a.train(1000)



