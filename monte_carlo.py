import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm

from environment import Environment,epsilon_greedy
from config import *

class MonteCarlo():
    def __init__(self,epsiodes=EPISODES):
        self.n = np.zeros((N_ACTIONS,N_PLAYER,N_DEALER))
        self.q = np.zeros_like(self.n)
        self.ep = epsiodes
    
    def generate(self,env):
        logs = []
        p,d = env.initialize()
        done = False
        while not done:
            eps = N0/(N0 + np.sum(self.n,axis=0)[p-1,d-1])
            a = epsilon_greedy(eps,self.n,p,d)
            pos = [p,d,a]
            done, r, p, d = env.step(p,d,a)
            pos.append(r)
            logs.append(pos)
        
        # logs.append([player,dealer,action,reward])
        return logs
       
    def mc_control(self,env):
        for _ in tqdm(range(self.ep)):
            logs = self.generate(env)
            r = logs[-1][3]
            for pos in logs:
                p,d,a,_ = pos
                self.n[a,p-1,d-1] += 1
                self.q[a,p-1,d-1] += (1/self.n[a,p-1,d-1])*(r-self.q[a,p-1,d-1])
            # print(np.sum(self.n),np.sum(self.q))
        
        return self.n,self.q

x = int(input('Number of Iterations:'))
mc = MonteCarlo(epsiodes=x)
env = Environment()
N,Q = mc.mc_control(env)

Q_star = np.max(Q,axis=0)
np.save('qstar',Q_star)

fig = plt.figure(figsize=(18,9))
ax = fig.gca(projection='3d')
X = np.arange(N_PLAYER)
Y = np.arange(N_DEALER)
X, Y = np.meshgrid(X, Y)
surf = ax.plot_surface(X,Y,Q_star.T, cmap=cm.jet,linewidth=0, antialiased=False)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()
