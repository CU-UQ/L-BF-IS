# %%
from typing import Callable, Optional
from tqdm import tqdm
import torch
import numpy as np
from sklearn.mixture import GaussianMixture
# %%
class LBFIS():
    def __init__(self, 
                 model_LF: Callable, 
                 sample_p: Callable,
                 ell: float,
                 num_variable: Optional[int] = None):
        self.model_LF = model_LF
        self.sample_p = sample_p
        self.ell = ell
        self.num_variable = num_variable

    def potential(self, z: torch.Tensor): # evaluate the potential function for the given z
        return self.ell * torch.tanh(self.model_LF(z))

    def unadjusted_langevin_algorithm(self,
                                    iter_num=int(1e3), 
                                    step=1e-2, 
                                    burn_in=int(1e6),
                                    num_thread=1):
        Z0 = self.sample_p(num_thread)
        Zi = Z0
        samples = []
        for i in tqdm(range(iter_num + burn_in)):
            Zi.requires_grad_()
            u = self.potential(Zi).mean()
            grad = torch.autograd.grad(u, Zi)[0]
            rejected = True
            new_step = step
            count = 0
            while rejected:
                new_Zi = Zi.detach() - new_step * grad + torch.sqrt(torch.tensor(2 * new_step)) * torch.randn(self.num_variable)
                if (new_Zi < 1).all() and (new_Zi > -1).all(): # only applicable to the case of [-1, 1]^d
                    Zi = new_Zi
                    rejected = False
                else:
                    count += 1
                if count > 10:
                    new_step /= 2
                    count = 0
            samples.append(Zi.detach())
        return torch.cat(samples, 0)[burn_in*num_thread:]

    def weight(self, x:torch.Tensor): # evaluate the unnormalized weight for the given xi
        wt = (self.ell * torch.tanh(self.model_LF(x))).exp()
        return wt
    
def gaussian_mix(est: LBFIS, 
                 h_HF: Callable, 
                 N: int,
                 num_samps: int = int(1e6),
                 num_component: int = 10) -> float:
    fail_samps = np.empty((0, est.num_variable))
    while len(fail_samps) < 10:
        samps = est.sample_p(num_samps)
        fail_samps = samps[est.model_LF(samps) < 0].detach().numpy()
    gm = GaussianMixture(n_components=num_component, random_state=0).fit(fail_samps)
    gm_samps = gm.sample(N)[0]
    wt = 0.5 * gm.predict_proba(gm_samps) ** (-1) # 0.5 is from p, which is uniform between -1 and 1
    return np.sum(wt[h_HF(torch.from_numpy(gm_samps).type(torch.FloatTensor)).detach().numpy() < 0]) / N
    
def control_variate(est: LBFIS, f_HF: Callable, x: torch.Tensor) -> float:
    ExL = est.ExL
    f_LF = est.model_LF
    fHx = f_HF(x)
    fLx = f_LF(x)
    cov = ((fHx - fHx.mean()) * (fLx - fLx.mean())).mean()
    var = (fLx - fLx.mean()).square().mean()
    if var.item() == 0.0:
        mu = 0.0
    else:
        mu  = cov.item()/var.item()

    return (fHx.mean() - mu * (fLx - ExL).mean()).item()

# # %%
# # import data
# from BidataGen import sample_p
# from BidataGen import cav_HF as HF
# from BidataGen import cav_LF as LF
# ell = 1.0
# num_variable = 52
# sample = lambda n: sample_p(n, num_variable)
# h_LF = lambda x: -0.2320 - LF(x)
# h_HF = lambda x: -0.2085 - HF(x)
# est = LBFIS(h_LF, sample, ell, num_variable)

# ExH = np.mean((h_HF(sample(int(1e6))) < 0).detach().numpy())
# # %%
# q_lst = est.unadjusted_langevin_algorithm(iter_num=int(1e4), step=1e-2, burn_in=int(1e4), num_thread=1)
# # %%
# MC_N_lst = (10**np.linspace(1, 4, 10)).astype('int')
# MC = []
# IS = []
# #CV = []
# num_trial = 1000
# for j in range(num_trial):
#     MC_est_lst = []
#     IS_est_lst = []
# #    CV_est_lst = []
#     for MC_N in MC_N_lst:
#         xi_p = sample(MC_N) # sample MC xi_p
#         xi_q = q_lst[np.random.choice(len(q_lst),MC_N)] # sample xi_q
#         wt_q = est.weight(xi_q)
#         wt_q /= wt_q.sum() 
#         MC_est_lst.append(np.mean((h_HF(xi_p) < 0).detach().numpy()))
#         IS_est_lst.append(wt_q[h_HF(xi_q) < 0].sum().item())
# #        CV_est_lst.append(control_variate(est, HF, xi_p))
#     MC.append(MC_est_lst)
#     IS.append(IS_est_lst)
# #    CV.append(CV_est_lst)

# MC      = np.array(MC)
# IS      = np.array(IS)
# #CV      = np.array(CV)
# MC_mean = np.nanmean(MC, axis=0)
# MC_std  = np.nanstd(MC, axis=0)
# IS_mean = np.nanmean(IS, axis=0)
# IS_std  = np.nanstd(IS, axis=0)
# MC_err  = ((MC - ExH)**2)
# IS_err  = ((IS - ExH)**2)
# MC_rmse = np.sqrt(np.nanmean(MC_err, axis=0))
# IS_rmse = np.sqrt(np.nanmean(IS_err, axis=0))

# # %%
# import matplotlib.pyplot as plt
# fig, axs = plt.subplots(1, 2, figsize=(12, 4))
# fig.suptitle('Flow Over Cylindar (Max)', fontsize=20)
# scale = 1.96

# axs[0].fill_between(MC_N_lst, MC_mean-scale*MC_std, MC_mean+scale*MC_std,color='b',alpha=0.2,label='MC')
# axs[0].fill_between(MC_N_lst, IS_mean-scale*IS_std, IS_mean+scale*IS_std,color='g',alpha=0.2,label='L-BF-IS')
# # axs[0].fill_between(MC_N_lst, CV_mean-scale*CV_std, CV_mean+scale*CV_std,color='y',alpha=0.2,label='CV')
# axs[0].axhline(ExH,c='r',label=r'$P^{HF}$')
# axs[0].set_xscale('log')
# axs[0].set_xlabel('HF Sample Size')
# axs[0].set_ylabel('Estimator Value')
# axs[0].legend()
# axs[0].set_title(f'$\ell={est.ell:.4f}$')

# axs[1].plot(MC_N_lst, MC_rmse,color='b',alpha=1.0,label='MC')
# axs[1].plot(MC_N_lst, IS_rmse,color='g',alpha=1.0,label='L-BF-IS')
# # axs[1].plot(MC_N_lst, CV_rmse,color='y',alpha=1.0,label='CV')
# axs[1].set_xscale('log')
# axs[1].set_yscale('log')
# axs[1].set_xlabel('HF Sample Size')
# axs[1].set_ylabel('RMSE')
# axs[1].legend()
# axs[1].set_title(f'$\ell={est.ell:.4f}$')

# fig.tight_layout()
# # fig.savefig('../figures/focM.pdf', dpi=100)
# # %%
