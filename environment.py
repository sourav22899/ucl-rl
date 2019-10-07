import numpy as np
from config import *

"""
    action: hit == 0, stick == 1
    red == -1, black == +1
"""

def epsilon_greedy(eps,N,p,d):
    x = np.random.random()
    if x < eps:
        return np.random.randint(2)
    else:
        return np.argmax(N,axis=0)[p-1,d-1]

class Environment():
    def __init__(self):
        pass
    
    def draw(self):
        x = np.random.randint(1,11)
        y = 0
        if np.random.random() < 1/3:
            y = -1
        else:
            y = 1
        return y,x

    def step(self,player,dealer,action):
        reward = 0
        end = False
        if action == 1:
            end = True
            while dealer < DEALER_MAX and dealer > 0:
                y,x = self.draw()
                # print('d:',y,x)
                dealer += (y*x)
            if dealer < 1 or dealer > PLAYER_MAX:
                reward = 1
            if dealer >= DEALER_MAX and dealer <= PLAYER_MAX:
                if player < dealer:
                    reward = -1
                elif player > dealer:
                    reward = 1
                elif player == dealer:
                    reward = 0
        elif action == 0:
            y,x = self.draw()
            # print('p:',y,x)
            player += (y*x)
            if player > PLAYER_MAX or player < 1:
                reward = -1
                end = True
        
        return end, reward, player, dealer
        
    def initialize(self):
        _,player = self.draw()
        _,dealer = self.draw()
        return player,dealer