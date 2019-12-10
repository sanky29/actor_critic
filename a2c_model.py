import env.py as en
import actor as ac
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
s = tf.Session()
class RLagent:


    def __init__(self,distance,neighbour):
        self.e = en.env(distance,neighbour)
        self.ac = ac.actor()
        self.cr = ac.critic()
        self.aco = self.ac.optimizer()
        self.cro = self.cr.optimizer()

    def episodes(self):
        a = self.e.state.copy()
        ans = []
        r = 0
        p = 0
        while not np.array_equal(self.e.state, self.e.dummy) and not np.array_equal(self.e.state, self.e.complete):
            p = s.run(self.ac.action(np.array([(self.e.state)]))[0])
            print(self.e.state , p)
            ans = ans + [self.e.step(p)]
            r = r + ans[-1][1]
        ans = [(a,0)] + ans
        self.e.reset()
        return(ans,r)

    def train1(self,e):
        episode = e[0]
        for i in range(0,len(episode) - 1):
            self.ac.z = np.dot(self.cr.model.predict(np.array([episode[i+1][0]])),1000)
            self.aco([np.array([episode[i][0]])])
            self.cr.next = tf.identity(self.cr.model.predict(np.array([episode[i+1][0]])))
            self.cr.reward = tf.constant([episode[i+1][1]])
            self.cro([np.array([episode[i][0]])])

    def train(self,i):
        t = ()
        x = [0]*i
        for j in range(0,i):
            t = self.episodes()
            x[j] = t[1]
            self.train1(t)
            print(int (j*100/i))
        plt.plot(x)
        plt.show()



