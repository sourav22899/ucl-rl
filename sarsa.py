import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm

from environment import Environment,epsilon_greedy
from config import *

class Sarsa():
    def __init__(self,lambda_,episodes=EPISODES):
        self.n = np.zeros((N_ACTIONS,N_PLAYER,N_DEALER))
        self.q = np.random.randn(*np.shape(self.n))
        self.e = np.zeros_like(self.q)
        self.l = lambda_
        self.ep = episodes
    
    def sarsa_control(self,env,qstar):
        mse_error = []
        for i in tqdm(range(self.ep)):
            self.e = np.zeros_like(self.q)
            p,d = env.initialize()
            eps = N0/(N0 + np.sum(self.n,axis=0)[p-1,d-1])
            a = epsilon_greedy(eps,self.n,p,d)
            done = False
            a_ = a
            while not done:
                self.n[a,p-1,d-1] += 1
                done,r,p_,d_ = env.step(p,d,a)
                if not done:
                    eps = N0/(N0 + np.sum(self.n,axis=0)[p_-1,d_-1])
                    a_ = epsilon_greedy(eps,self.n,p_,d_)
                    delta = r + self.q[a_,p_-1,d_-1] - self.q[a,p-1,d-1]
                else:
                    delta = r - self.q[a,p-1,d-1]
                
                self.e[a,p-1,d-1] += 1
                self.q += delta*(1/(self.n[a,p-1,d-1]))*self.e
                self.e *= self.l
                   
                p,d,a = p_,d_,a_

            if i%1000 == 0:
                mse_error.append(np.mean((qstar-np.max(self.q,axis=0))**2))
        
        return mse_error,self.n, self.q        

qstar = np.load('qstar.npy')
x = int(input('Number of Iterations:'))
mse_error_list = []

for l in range(11):
    lambda_ = l/10.0
    sarsa = Sarsa(lambda_,x)
    env = Environment()
    error,N_,Q_ = sarsa.sarsa_control(env,qstar)
    mse_error_list.append(error)

Q_star = np.max(Q_,axis=0)

fig = plt.figure(figsize=(18,9))
ax = fig.gca(projection='3d')
X = np.arange(N_PLAYER)
Y = np.arange(N_DEALER)
X, Y = np.meshgrid(X, Y)
surf = ax.plot_surface(X,Y,Q_star.T, cmap=cm.jet,linewidth=0, antialiased=False)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()

X = 1000*np.arange(len(mse_error_list[0]))
plt.figure(figsize=(18,9))
plt.grid()
plt.plot(X,mse_error_list[0],label='lambda = 0.0')
plt.plot(X,mse_error_list[-1],label='lambda = 1.0')
plt.legend()
plt.show()




