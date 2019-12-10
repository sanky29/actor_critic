'''
just make discrete action space environment
the model will work for maximum 20 cities
the action will be between current cities
so if i have 20 citites then action space will be 20*10 = 200
right now will do only that
here i am going to do things for only 8 cities including the source city and maximum demant of city is 2
so for 8 states it is
8*10 = 80
'''
import numpy as np



class env:

    def __init__(self,  distance, neighbour, cities):
        self.demand_matrix = np.array([2.0] * cities)
        self.demand_matrix[0] = 0
        self.truck_pos = 0
        self.truck_load = 1
        self.state = np.concatenate((self.demand_matrix , [self.truck_pos] ,  [self.truck_load]) , axis = 0)
        self.distance = distance
        self.complete = [0]*7

    def step(self, action):
        if (action[0] == self.truck_pos or action[1] > self.truck_load or action[1] > self.demand_matrix[action[0]]):
            self.state = self.dummy
            return (self.state,-100)
        else:
            z = self.truck_pos
            self.demand_matrix[action[0]] = float (self.demand_matrix[action[0]] - action[1])
            self.td = self.td - action[1]
            self.truck_load = self.truck_load - action[1]
            self.truck_pos = action[0]
            if (self.td == 0):
                self.state = self.complete
                return (self.state,100)
            else:
                if (self.truck_load == 0):
                    self.truck_load = 1
                    self.truck_pos = 0
            self.state = np.concatenate((self.demand_matrix, [self.truck_pos], [self.truck_load]), axis=0)
            return (self.state ,np.exp(action[1]*20)/(self.distance[z][action[0]]))

    def reset(self):
        self.demand_matrix =  np.array([2.0]*5)
        self.demand_matrix[0] = 0
        self.td = sum(self.demand_matrix)
        self.truck_pos = 0
        self.truck_load = 1
        self.state = np.concatenate((self.demand_matrix, [self.truck_pos], [self.truck_load]), axis=0)



