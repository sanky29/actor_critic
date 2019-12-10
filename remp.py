import numpy as np
import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import keras.backend as K
s = tf.Session()

class env:

    def __init__(self,  distance, neighbour):
        self.demand = np.array([2.0] * 5)
        self.td = 8
        self.demand[0] = 0
        self.tp = 0
        self.tl = 1
        self.distance = distance
        self.neighbour = neighbour
        self.state = np.power(np.sum(np.multiply(self.demand,self.distance[self.tp]))*(2 + self.tp),self.tl)
        self.dummy = -1
        self.complete = 1000

    def step(self, a):
        action = [int (a/5),(a%5)*0.25]
        if (action[0] == self.tp or action[1] > self.tl or action[1] > self.demand[action[0]]):
            self.state = self.dummy
            return ([self.state],-1.0,a)
        else:
            z = self.tp
            self.demand[action[0]] = self.demand[action[0]] - action[1]
            self.td = self.td - action[1]
            self.tl = self.tl - action[1]
            self.tp = action[0]
            if (self.td == 0):
                self.state = self.complete
                return ([self.state],100.0, a)
            else:
                if (self.tl == 0.0):
                    self.tl = 1
                    self.tp = 0
            self.state = np.power(np.sum(np.multiply(self.demand,self.distance[self.tp]))*(2 + self.tp),self.tl)
            return ([self.state] ,np.exp(action[1])*action[0]/(self.distance[z][int (action[0])]),a)

    def reset(self):
        self.demand =  np.array([2.0]*5)
        self.td = 8
        self.demand[0] = 0
        self.tp = 0
        self.tl = 1
        self.state =  np.power(np.sum(np.multiply(self.demand, self.distance[self.tp])) * (2 + self.tp), self.tl)

class actor_critic:

    def __init__(self):
        self.input = keras.Input(shape=(1,))
        self.l1 = keras.layers.Dense(3, activation='linear')(self.input)
        self.l2 = keras.layers.Dense(4, activation='linear' , use_bias= True)(self.l1)
        self.l3 = keras.layers.Dense(7, activation='linear', use_bias=True)(self.l2)
        self.l4 = keras.layers.Dense(10, activation='linear', use_bias=True)(self.l3)
        self.l5 = keras.layers.Dense(13, activation='linear', use_bias=True)(self.l4)
        self.l6 = keras.layers.Dense(16, activation='linear', use_bias= True)(self.l5)
        self.l7 = keras.layers.Dense(19, activation='linear', use_bias=True)(self.l6)
        self.l8= keras.layers.Dense(22, activation='linear', use_bias=True)(self.l7)
        '''the actor is here'''
        self.l9 = keras.layers.Dense(25, activation='softmax')(self.l3)
        self.actor = keras.models.Model(inputs=self.input, outputs=self.l9)
        self.z = tf.constant([np.random.uniform(0, 1)])



        '''     critic1 V(s) is here
        self.inputv = keras.Input(shape=(7,))
        self.l1v = keras.layers.Dense(10, activation='sigmoid')(self.inputv)
        self.l2v = keras.layers.Dense(20, activation='tanh', use_bias=True)(self.l1v)
        self.l3v = keras.layers.Dense(30, activation='sigmoid', use_bias=True)(self.l2v)
        self.l6 = keras.layers.Dense(4, activation='tanh' , use_bias= True)(self.l3v)
        self.l7 = keras.layers.Dense(1, activation='linear' , use_bias= True)(self.l6)
        self.criticv = keras.models.Model(inputs=self.inputv, outputs=self.l7)
        self.q = tf.constant([np.random.uniform(0, 1)])
        self.vloss = tf.negative(tf.square(self.q - self.criticv.output))'''

        '''the critic Q(s,a) is here'''
        self.inputq1 = keras.Input(shape=(1,))
        self.l2q = keras.layers.Dense(5, activation='linear', use_bias=True)(self.inputq1)
        self.l3q = keras.layers.Dense(1, activation='linear', use_bias=True)(self.l2q)
        self.inputq = keras.Input(shape=(1,))
        self.l4q = keras.layers.Dense(1, activation = 'linear' , use_bias= True)(self.inputq)
        self.l44q = keras.layers.Dense(1,activation='linear' , use_bias= True)(self.l4q)
        self.l5q = keras.layers.Concatenate(axis = -1)([self.l3q,self.l44q])
        self.l6q = keras.layers.Dense(2,activation='linear' , use_bias= True)(self.l5q)
        self.l7q = keras.layers.Dense(1,activation= 'linear' ,use_bias = True)(self.l6q)
        self.criticq = keras.models.Model(inputs=[self.inputq1,self.inputq], outputs=self.l7q)
        self.reward = tf.constant([0.0])
        self.next = tf.constant([0.0])
        self.qloss = (tf.square(tf.stop_gradient(self.reward) + tf.stop_gradient(self.next) - self.criticq.output))
        self.aloss = tf.negative(tf.multiply(tf.log(tf.reduce_max(self.actor.output)), tf.stop_gradient(self.z)))

    def ao(self):
        op = keras.optimizers.RMSprop(lr=0.001, epsilon=0.1, rho=0.99)
        updates = op.get_updates(self.actor.trainable_weights,[], self.aloss)
        return(K.function([self.input], [], updates = updates))

    def action(self,input):
        z = random.uniform(0,1)
        if (z <= 0.95):
            return s.run((tf.argmax(self.actor.predict(input), axis=-1))[0])

        elif (z > 0.95):
            return s.run(tf.random.categorical(tf.log(self.actor.predict(input)),1))[0][0]

    '''def cov(self):
        updates = keras.optimizers.sgd(0.005).get_updates(self.criticv.trainable_weights, [], self.vloss)
        return keras.backend.function([self.criticv.input], [], updates=updates)'''

    def qo(self):
        op = keras.optimizers.RMSprop(lr=0.005, epsilon=0.1, rho=0.99)
        updates = op.get_updates(self.criticq.trainable_weights,[], self.qloss)
        return(K.function([self.inputq1, self.inputq], [], updates = updates))


class RLagent:

    def __init__(self, distance, neighbour):
        self.e = env(distance, neighbour)
        self.agent = actor_critic()
        self.ao = self.agent.ao()
        self.qo = self.agent.qo()
        '''self.crvo = self.agent.cov()'''

    def episodes(self):
        ans = [([self.e.state.copy()], 0,0)]
        r = 0
        p = 0
        while  self.e.state != self.e.dummy and self.e.state != self.e.complete:
             p = self.agent.action(np.array([self.e.state]))
             print(self.e.state, p)
             ans = ans + [self.e.step(p)]
             r = r + ans[-1][1]
        self.e.reset()
        return (ans, r)

    def train1(self, e):
        episode = e[0]
        for i in range(0, len(episode) - 2):
            self.agent.next = tf.identity(self.agent.criticq.predict([np.array([episode[i+1][0]]),np.array([[episode[i+2][2]]])]))
            self.agent.q = tf.identity(self.agent.criticq.predict([np.array([episode[i][0]]),np.array([[episode[i+1][2]]])]))
            self.agent.reward = tf.constant([episode[i+1][1]])
            self.agent.z = tf.identity(self.agent.q)
            self.ao([np.array([episode[i][0]])])
            #self.crvo([np.array([episode[i][0]])])#
            self.qo([np.array([episode[i][0]]),np.array([[episode[i+1][2]]])])
        self.agent.next = tf.constant(np.array([-1.0]))
        self.agent.q = tf.identity(self.agent.criticq.predict([np.array([episode[len(episode) - 2][0]]), np.array([[episode[len(episode) - 1][2]]])]))
        self.agent.reward = tf.constant([episode[len(episode) - 1][1]])
        self.ao([np.array([episode[len(episode) - 2][0]])])
        self.qo([np.array([episode[len(episode) - 2][0]]), np.array([[episode[len(episode) - 1][2]]])])

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
    distance = np.array([[0,2,5,6,7],[2,0,3,4,5],[5,3,0,7,4],[6,4,7,0,3],[7,5,4,3,0]])
    neighbors = [[0,1,0,0,0],[1,0,1,1,0],[0,1,0,0,1],[0,1,0,0,1],[0,0,1,1,0]]
    a = RLagent(distance,neighbors)
    print(a.agent.actor.predict(np.array([[80.0]])))
    a.train(1000)
    print (a.agent.actor.predict(np.array([[80.0]])))



