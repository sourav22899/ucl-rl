import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm

from environment import Environment
from config import *

class LFA():
    def __init__(self,lambda_=1.0,alpha=ALPHA,eps=EPSILON,epsiodes=EPISODES):
        self.theta = np.random.randn(N_FEATURES)
        self.e = np.zeros_like(self.theta)
        self.alpha = alpha
        self.eps = eps
        self.ep = epsiodes
        self.l = lambda_

    def coarse_coding(self,p,d):
        """
            dealer(s) = {[1, 4], [4, 7], [7, 10]}
            player(s) = {[1, 6], [4, 9], [7, 12], [10, 15], [13, 18], [16, 21]}
            a = {hit, stick}
            where
            • dealer(s) is the value of the dealer’s first card (1–10)
            • sum(s) is the sum of the player’s cards (1–21)
        """
        d_i,p_i =[],[]

        if d >=1 and d <= 4:
            d_i.append(1)
        if d >= 4 and d <= 7:
            d_i.append(2)
        if d >= 7 and d <= 10:
            d_i.append(3)
        
        if p >= 1 and p <= 6:
            p_i.append(1)
        if p >= 4 and p <= 9:
            p_i.append(2)
        if p >= 7 and p <= 12:
            p_i.append(3)
        if p >= 10 and p <= 15:
            p_i.append(4)
        if p >= 13 and p <= 18:
            p_i.append(5)
        if p >= 16 and p <= 21:
            p_i.append(6)
        
        return p_i,d_i

    def phi(self,p,d,a):
        p_i,d_i = self.coarse_coding(p,d)
        f = np.zeros((2,6,3))
        for i in p_i:
            for j in d_i:
                f[a,i-1,j-1] = 1
        return f.flatten()   

    def q(self,p,d,a):
        return self.phi(p,d,a).T.dot(self.theta)
    
    def best_action(self,p,d):
        return 0 if self.q(p,d,0) > self.q(p,d,1) else 1

    def linear_eps_greedy(self,p,d):
        if np.random.random() < self.eps:
            return np.random.randint(2)
        return self.best_action(p,d)

    def all_q(self,print_=False):
        if print_:
            print(self.theta)
        q = np.zeros((N_ACTIONS,N_PLAYER,N_DEALER))
        for a in range(N_ACTIONS):
            for p in range(N_PLAYER):
                for d in range(N_DEALER):
                    q[a,p-1,d-1] = self.q(p,d,a)
        return q

    def lfa_control(self,env,qstar):
        mse_error = []
        for i in tqdm(range(self.ep)):
            self.e = np.zeros_like(self.theta)
            p,d = env.initialize()
            a = self.linear_eps_greedy(p,d)
            done = False
            a_ = a
            while not done:
                done,r,p_,d_ = env.step(p,d,a)
                phi = self.phi(p,d,a)
                if not done:
                    a_ = self.linear_eps_greedy(p_,d_)
                    delta = r + self.q(p_,d_,a_) - self.q(p,d,a)
                else:
                    delta = r - self.q(p,d,a)
                
                self.e += phi
                self.theta += delta*(self.alpha)*self.e             # Updating the theta.
                self.e *= self.l
                   
                p,d,a = p_,d_,a_

            if i%1000 == 0:
                mse_error.append(np.mean((qstar-np.max(self.all_q(),axis=0))**2))
        
        return mse_error,self.theta       


x = int(input('Number of Iterations:'))
qstar = np.load('qstar.npy')
mse_error_list = []

for l in range(11):
    lambda_ = l/10.0
    lfa = LFA(lambda_=lambda_,epsiodes=x)
    env = Environment()
    error, theta = lfa.lfa_control(env,qstar)
    mse_error_list.append(error)

X = 1000*np.arange(len(mse_error_list[0]))
plt.figure(figsize=(18,9))
plt.grid()

for i in range(11):
    label = 'lambda:' + str(i/10)
    plt.plot(X,mse_error_list[i],label=label)
plt.legend()
plt.show()

Q_star = np.max(lfa.all_q(),axis=0)
fig = plt.figure(figsize=(18,9))
ax = fig.gca(projection='3d')
X = np.arange(N_PLAYER)
Y = np.arange(N_DEALER)
X, Y = np.meshgrid(X, Y)
surf = ax.plot_surface(X,Y,Q_star.T, cmap=cm.jet,linewidth=0, antialiased=False)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()
