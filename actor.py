'''will try to make neursl network as'''

import keras
import tensorflow as tf
import numpy as np
'''
define input line as
2*2 neural network'''
class actor:

    def __init__(self):
        self.input = keras.Input(shape=(7,))
        self.l1 = keras.layers.Dense(10, activation='linear')(self.input)
        self.l2 = keras.layers.Dense(20, activation='linear')(self.l1)
        self.l3 = keras.layers.Dense(30, activation='linear')(self.l2)
        self.l4 = keras.layers.Dense(40, activation='linear')(self.l3)
        self.l5 = keras.layers.Dense(25, activation='softmax')(self.l4)
        self.model = keras.models.Model(inputs=self.input, outputs=self.l5)
        self.z = tf.constant([np.random.uniform(0, 1)])

    def loss(self):
        return (tf.multiply(tf.log(tf.reduce_max(self.model.output)), self.z))

    def optimizer(self):
        updates = keras.optimizers.sgd(0.0001).get_updates(self.model.trainable_weights, [], self.loss())
        return keras.backend.function([self.model.input], [], updates=updates)

    def action(self, input):
        return (tf.argmax(self.model.predict(input) , axis = -1))

class critic:

    def __init__(self):
        self.input = keras.Input(shape=(7,))
        self.l1 = keras.layers.Dense(7, activation='tanh', )(self.input)
        self.l2 = keras.layers.Dense(5, activation='tanh')(self.l1)
        self.l3 = keras.layers.Dense(4, activation='tanh')(self.l2)
        self.l4 = keras.layers.Dense(1, activation='sigmoid')(self.l3)
        self.model = keras.models.Model(inputs=self.input, outputs=self.l4)
        self.reward = tf.constant([0.0])
        self.next = tf.constant([0.0])

    def loss(self):
        return tf.negative(tf.square(self.reward + self.next - self.model.output))

    def optimizer(self):
        updates = keras.optimizers.sgd(0.0001).get_updates(self.model.trainable_weights, [], self.loss())
        return keras.backend.function([self.model.input], [], updates=updates)

ac = actor()
ok = actor.optimizer()
ac.z = tf.constant([-1])
i = tf.constant([[1,2,3,4,5,6,7]])

for j in range(1,100):
    for k in range(1,100):
        ac.action(i)
        ok(i)
    s.run(ac.action(i))
