'''
author: NUOJIN
'''

from typing import Callable, Tuple
import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm
import math
import matplotlib.pyplot as plt

class LBFIS():
    """
    LBFIS is a class for constructing the L-BF-IS estimator,
    which leverages LF information to improve the performance of MC.
    """
    def __init__(self,
                 f_LF: Callable, # LF evaluation function
                 sample_x_p: Callable, # sampling function for p
                 bound: Tensor # domain of input xi
                 ) -> None:
        super().__init__()
        self.f_LF     = f_LF
        self.sample_p = sample_x_p
        self.bound    = bound
        self.ell      = 1.0

        # initialize the value of ExL
        N        = int(1e6) # number of samples for estimating ExL
        self.ExL = f_LF(sample_x_p(N)).nanmean()

    def potential(self, z): # evaluate the potential function for the given z
        """
        potential function evaluates the potential U for the given z.
        """
        return (self.f_LF(z)-self.ExL)**2/(2*self.l**2)

    def sample_q(self, 
                iter_num: int = int(1e5), # number of iterations as output
                step: float = 1e-2, # step size of Langevin algorithm
                burn_in: int = int(1e5) # number of MC burn-in iterations
                ) -> Tensor:
        """
        This function implements the unadjusted Langevin algorithm
        for sampling biasing distribution q.
        """
        Z0 = self.sample_p(1)
        Zi = Z0
        samples = []
        for i in tqdm(range(iter_num + burn_in)):
            Zi.requires_grad_()
            u = self.potential(Zi).mean()
            grad = torch.autograd.grad(u, Zi)[0]
            rejected = True
            count = 0
            while rejected:
                count += 1
                new_Zi = Zi.detach() - step * self.l**2 * grad 
                + np.sqrt(2 * step * self.l**2) * torch.randn(1, 1000)
                if (new_Zi < self.bound[1]).all() and (new_Zi > self.bound[0]).all():
                    Zi = new_Zi
                    rejected = False
                    count = 0
                elif count > 10:
                    step /= 2
                    count = 0
            samples.append(Zi.detach())
        return torch.cat(samples, 0)[burn_in:]
    
    def weight(self, x: Tensor) -> Tensor: # evaluate the unnormalized weight for the given xi
        """
        This function evaluates the unnormalized weight for the given q samples.
        """
        wt = ((self.f_LF(x) - self.ExL).square()/(2*self.l**2)).exp()
        return wt
    
    def LBFIS_algorithm(self, 
                        x_q_lst: Tensor, # samples from q
                        MC_N_lst: np.ndarray, # list of number of MC samples
                        num_trial: int = 100, # number of trials for data
                        ) -> Tuple[np.ndarray, np.ndarray]:
        """
        This function implements the L-BF-IS algorithm and returns MC and IS results
        """
        MC = []
        IS = []
        iter_num  = len(x_q_lst)
        for j in tqdm(range(num_trial)):
            MC_est_lst = []
            IS_est_lst = []
            for MC_N in MC_N_lst:
                xi_p = self.sample_p(MC_N) # sample MC xi_p
                xi_q = x_q_lst[np.random.choice(iter_num, MC_N)] # sample xi_q
                wt_q = self.weight(xi_q)
                wt_q /= wt_q.sum()
                MC_est_lst.append(x_H(xi_p).mean().item())
                IS_est_lst.append((wt_q*x_H(xi_q)).sum())
            MC.append(MC_est_lst)
            IS.append(IS_est_lst)

        return np.array(MC), np.array(IS)
    
# the element function
def element(xi):
    k   = torch.arange(1,1001)
    vec = torch.sin(k)/k
    return 2 - xi@vec

# the finite sum function
def finite_sum(xi, M):
    res  = torch.zeros(len(xi))
    elem = element(xi)
    for m in range(M+1):
        res += elem**m/math.factorial(m)
    return res
    
# LF evaluation function
def x_L(xi):
    M = 3
    return finite_sum(xi, M)

# HF evaluation function
def x_H(xi):
    elem = element(xi)
    return elem.exp()

# sampling xi by p(xi)
def sample_xi_p(N):
    return torch.rand(N, 1000)

if __name__ == "__main__":
    ## conduct experiments for estimating ExH with regular MC and L-BF-IS
    path = 'data/dim-1000.pt'

    # construct the L-BF-IS estimator
    model = LBFIS(x_L, sample_xi_p, torch.tensor([0.0,1.0]))

    # update model lengthscale
    model.l = 10.0

    try:
        # load q samples
        x_q_lst = torch.load(path)
    except:
        # sample q
        x_q_lst = model.sample_q(iter_num=int(1e4), step=1e-3, burn_in=int(1e4))
        torch.save(x_q_lst, path)

    # collect final results
    MC_N_lst = np.array([int(10**n) for n in np.linspace(0, 4, 50)])
    MC, IS = model.LBFIS_algorithm(x_q_lst, MC_N_lst, 1000)

    MC_mean = np.nanmean(MC, axis=0)
    MC_std  = np.nanstd(MC, axis=0)
    IS_mean = np.nanmean(IS, axis=0)
    IS_std  = np.nanstd(IS, axis=0)

    # plot
    scale = 1.94
    plt.figure(figsize=(8,6))
    plt.fill_between(MC_N_lst, MC_mean-scale*MC_std, MC_mean+scale*MC_std,color='b',alpha=0.2,label='MC')
    plt.fill_between(MC_N_lst, IS_mean-scale*IS_std, IS_mean+scale*IS_std,color='g',alpha=0.2,label='L-BF-IS')
    #plt.axhline(ExH,c='r',label=r'$\mu^{HF}$')
    plt.xscale('log')
    plt.xlabel('HF Sample Size')
    plt.ylabel('Estimator Value')
    plt.legend()
    plt.title(r'$\ell=10.0$')
    plt.show()

# class NN(nn.Module): 
#     def __init__(self, 
#                  input_size: int,
#                  bound: Tensor,
#                  layer_num: int = 2
#                  ) -> None:
#         super().__init__()
#         width   = int(2**np.ceil(np.log2(input_size)))

#         modules = [nn.Sequential(
#             nn.Linear(input_size, width),
#             nn.ReLU())]
#         # Build NN Structure
#         for i in range(layer_num):
#             modules.append(
#                 nn.Sequential(
#                     nn.Linear(width, width),
#                     nn.ReLU())
#             )
#         modules.append(nn.Linear(width, input_size))

#         self.model = nn.Sequential(*modules)
#         self.low_bnd = bound[0]
#         self.upp_bnd = bound[1]
            

#     def forward(self, x: Tensor) -> Tensor:
#         output = self.model(x)
#         return torch.clamp(output, min=self.low_bnd, max=self.upp_bnd)
    
# def train_NN(model: NN,
#         trn_data: Tensor,
#         loss: Any, # this is the input loss function
#         val_data: Tensor = None,
#         batch_size: int = 64,
#         epochs: int = 500
#         ) -> List:
#     trn_dataloader = DataLoader(trn_data, batch_size=batch_size)
#     opt = torch.optim.Adam(model.parameters(), lr=1e-3, betas= (0.9, 0.99))
#     losses, val_losses = [], []
#     with tqdm.tqdm(range(epochs), unit=' Epoch') as tepoch:
#         epoch_loss = 0
#         for epoch in tepoch:
#             for batch_index, data in enumerate(trn_dataloader):
#                 opt.zero_grad()
#                 trn_loss = loss(model, data)
#                 trn_loss.backward()
#                 epoch_loss += trn_loss.item()
#                 opt.step()

#                 epoch_loss += trn_loss
#             epoch_loss /= len(trn_dataloader) * batch_size
#             losses.append(np.copy(epoch_loss.detach().numpy()))
#             tepoch.set_postfix(loss=epoch_loss.detach().numpy())
#             if val_data is not None:
#                 val_losses.append(loss(model,val_data).detach().numpy()/len(val_data))
#     if val_data is not None:
#         return losses, val_losses
#     return losses

# ## The following functions are for sampling q
# class Q():
#     def __init__(self,
#                  x_L: Callable, # LF evaluation function
#                  sample_xi_p: Callable, # sampling function for p
#                  bound: Tensor # domain of input xi
#                  ) -> None:
#         super().__init__()
#         self.x_L      = x_L
#         self.p        = sample_xi_p
#         self.bound    = bound
#         self.grad_avl = False # indicate the availability of x_L gradient
#         self.l        = torch.tensor([1.0])

#         N = int(1e6)
#         self.ExL = x_L(sample_xi_p(N)).nanmean()

#     def update_grad(self,grad_x_L:Callable) -> None:
#         self.grad_avl = True
#         self.grad_x_L = grad_x_L

#     def update_l(self, T:NN) -> None: # function to update the value of lengthscale
#         self.l.requires_grad_(True)
#         opt = torch.optim.SGD([self.l], lr=0.01, momentum=0.9)
#         batch_size = 16

#         for _ in range(5):
#             q_samples = self.q(batch_size, T, iter_num=int(1e3), tau=1e-3)
#             opt.zero_grad()
#             loss = (self.x_L(q_samples) - self.ExL).square().mean()
#             loss.backward()
#             opt.step()

#         self.l.requires_grad_(False)

#     def score(self,
#               xi:Tensor
#               ) -> Tensor: #score function, return grad q(xi)
#         if self.grad_avl:
#             grad_xL = self.grad_x_L(xi)
#         else:
#             new_xi = xi.detach().clone()
#             new_xi.requires_grad_(True)
#             xL_xi = self.x_L(new_xi)
#             xL_xi.backward(torch.ones(len(new_xi)))
#             grad_xL = new_xi.grad.detach()
#         return ((self.ExL - self.x_L(xi))/(self.l**2)).unsqueeze(-1) * grad_xL

#     def langevin(self,
#                  xi_init:Tensor, 
#                  iter_num:int, 
#                  tau:float
#                  ) -> Tensor: # function for Langevin algorithm
#         for _ in range(iter_num):
#             xi_new  = xi_init + tau * self.score(xi_init) + np.sqrt(2*tau)*torch.randn_like(xi_init)
#             xi_init = xi_new.detach().clamp_(min=self.bound[0],max=self.bound[1])
#         return xi_init

#     def q(self,
#           N:int, # number of q samples
#           T:NN,  # an NN for initial state 
#           iter_num:int=int(1e5), # iteration number of Langevin algorithm
#           tau:float=1e-5 # step size of Langevin algorithm
#           ) -> Tensor: # sample xi by q(x)
#         xi_p    = self.p(N)
#         xi_init = T(xi_p).detach() + 1e-2 * torch.randn_like(xi_p)
#         return self.langevin(xi_init,iter_num,tau)