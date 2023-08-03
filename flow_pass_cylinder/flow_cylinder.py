# %%
import torch
import numpy as np
import scipy.io as sio
from scipy.special import legendre
from itertools import product
from numpy.polynomial.legendre import Legendre

# %%
def generate_total_degree_basis(order, num_variables):
    """
    Generate all possible combinations of exponents in the total degree space.
    """
    # total_degree_basis = []
    # whole_basis = list(product(range(order + 1), repeat=num_variables))
    # for i, tpl in enumerate(whole_basis):
    #     if sum(tpl) <= order:
    #         total_degree_basis.append(tpl)
    # return total_degree_basis
    if order == 3 and num_variables == 4:
        file = sio.loadmat('v4o3.mat')
        total_degree_basis = file['base']
    elif order == 4 and num_variables == 4:
        file = sio.loadmat('v4o4.mat')
        total_degree_basis = file['base']
    else:
        raise ValueError("Incorrect order or num_variable.")
    return torch.from_numpy(total_degree_basis).type(torch.FloatTensor)

def polynomial_chaos_expansion(inputs, coefficients, total_degree_basis):
    """
    Compute the Polynomial Chaos Expansion given the inputs, coefficients, total degree basis,
    and Legendre polynomials.
    """
    result = torch.zeros(inputs.shape[0])
    for i in range(len(coefficients)):
        exponents = total_degree_basis[i]
        product_term = torch.ones(inputs.shape[0])
        for j, exp in enumerate(exponents):
            product_term *= Legendre.basis(exp)(inputs[:, j])
        result += coefficients[i] * product_term
    return result

# %%
# load max qoi functions 
def sample_p(num_samples:int) -> torch.Tensor:
    '''
    Sampling function for sampling density p(x)
    Input: 
        num_samples: number of samples
    Output:
        p_samples: num_samples x 4 uniform random matrix between -1 and 1
    '''
    return torch.rand(num_samples, 4) * 2 - 1

def build_pce(p_samples: torch.Tensor,
              coefficients: torch.Tensor,
              order: int) -> torch.Tensor:
    '''
    This function evaluate PCE results given inputs and order
    Input:
        p_samples: samples drawing from p
        coefficients: coefficents for PCE design matrix
        order: the order of PCE
    Output: 
        pce_results: results of PCE
    '''
    _, num_variables = p_samples.shape

    # Generate the total degree basis.
    total_degree_basis = generate_total_degree_basis(order, num_variables)

    # Compute the Polynomial Chaos Expansion using Legendre polynomials in the total degree space.
    pce_results = polynomial_chaos_expansion(p_samples, coefficients, 
                                             total_degree_basis)

    return pce_results

def f_LF_mean(p_samples):
    '''
    Low-fidelity PCE function for mean QoI
    Input dimension is 4 and order is 3
    '''
    # load coefficient
    file = sio.loadmat('c_c_mean.mat')
    coef = torch.from_numpy(file['c_l1']).type(torch.FloatTensor).squeeze()

    return build_pce(p_samples, coef, order=3)

def f_HF_mean(p_samples):
    '''
    Low-fidelity PCE function for mean QoI
    Input dimension is 4 and order is 3
    '''
    # load coefficient
    file = sio.loadmat('c_f_mean.mat')
    coef = torch.from_numpy(file['c_l1']).type(torch.FloatTensor).squeeze()

    return build_pce(p_samples, coef, order=3)

def f_LF_max(p_samples):
    '''
    Low-fidelity PCE function for max QoI
    Input dimension is 4 and order is 3
    '''
    # load coefficient
    file = sio.loadmat('c_c_max.mat')
    coef = torch.from_numpy(file['c_l1']).type(torch.FloatTensor).squeeze()

    return build_pce(p_samples, coef, order=3)

def f_HF_max(p_samples):
    '''
    Low-fidelity PCE function for max QoI
    Input dimension is 4 and order is 3
    '''
    # load coefficient
    file = sio.loadmat('c_f_max.mat')
    coef = torch.from_numpy(file['c_l1']).type(torch.FloatTensor).squeeze()

    return build_pce(p_samples, coef, order=4)

# %%
x = sample_p(100)
fL_mean = f_LF_mean(x)
fH_mean = f_HF_mean(x)
fL_max  = f_LF_max(x)
fH_max  = f_HF_max(x)
# %%
import matplotlib.pyplot as plt
# evaluate ExL and ExH
x   = sample_p(int(1e6))
xL  = f_LF_mean(x)
xH  = f_HF_mean(x)
ExL = xL.mean().item()
ExH = xH.mean().item()
SxL = xL.std().item()
SxH = xH.std().item()

bound = torch.tensor([0, 1]) # range of inputs

# plot histograms of xL and xH
plt.figure(figsize=(8,4))
plt.hist([xL,xH],label=[r'$f^{LF}(X)$',r'$f^{HF}(X)$'],color=['tab:blue','tab:red'])
plt.axvline(x = ExL, color = 'tab:blue', label = r'$\mu^{LF}$')
plt.axvline(x = ExH, color = 'tab:red', label = r'$\mu^{HF}$')
plt.legend()
# %%
# plot histograms of normalized xL and xH
n_xL = (xL - ExL)/SxL
n_xH = (xH - ExH)/SxH

plt.figure(figsize=(8,4))
plt.hist([n_xL, n_xH],label=[r'$\bar{f}^{LF}(X)$',r'$\bar{f}^{HF}(X)$'],color=['tab:blue','tab:red'])
plt.legend()
# %%
from tqdm import tqdm

l = 0.001

def potential(z,l): # evaluate the potential function for the given z
    return (f_LF_mean(z)-ExL)**2/(2*l**2)

def unadjusted_langevin_algorithm(potential, iter_num=int(1e4), step=1e-2, burn_in=int(1e6)):
    Z0 = sample_p(1)
    Zi = Z0
    samples = []
    for i in tqdm(range(iter_num + burn_in)):
        Zi.requires_grad_()
        u = potential(Zi, l).mean()
        grad = torch.autograd.grad(u, Zi)[0]
        rejected = True
        new_step = np.copy(step)
        count = 0
        while rejected:
            new_Zi = Zi.detach() - new_step * l**2 * grad + np.sqrt(2 * new_step * l**2) * torch.randn(1)
            if (new_Zi < 1).all() and (new_Zi > -1).all():
                Zi = new_Zi
                rejected = False
                count += 1
            if count > 10:
                new_step /= 2
        samples.append(Zi.detach())
    return torch.cat(samples, 0)[burn_in:]

def weight(x): # evaluate the unnormalized weight for the given xi
    wt = ((f_LF_mean(x) - ExL).square()/(2*l**2)).exp()
    return wt
# %%
q_lst = unadjusted_langevin_algorithm(potential, iter_num=int(1e5), step=1e-3, burn_in=int(1e5))
# %%
# define control variate function
def control_variate(x):
    fHx = f_HF_mean(x)
    fLx = f_LF_mean(x)
    cov = ((fHx - fHx.mean()) * (fLx - fLx.mean())).mean()
    var = (fLx - fLx.mean()).square().mean()
    if var.item() == 0.0:
        mu = 0.0
    else:
        mu  = cov.item()/var.item()

    return (fHx.mean() - mu * (fLx - ExL).mean()).item()

# %%
# Inversion Sampling for Monte Carlo
MC_N_lst = (10**np.linspace(0, 4, 10)).astype('int')
MC = []
IS = []
CV = []
num_trial = 1000
for j in range(num_trial):
    MC_est_lst = []
    IS_est_lst = []
    CV_est_lst = []
    for MC_N in MC_N_lst:
        xi_p = sample_p(MC_N) # sample MC xi_p
        xi_q = q_lst[np.random.choice(len(q_lst),MC_N)] # sample xi_q
        wt_q = weight(xi_q)
        wt_q /= wt_q.sum()
        MC_est_lst.append(f_HF_mean(xi_p).mean().item())
        IS_est_lst.append((wt_q * f_HF_mean(xi_q)).sum().item())
        CV_est_lst.append(control_variate(xi_p))
    MC.append(MC_est_lst)
    IS.append(IS_est_lst)
    CV.append(CV_est_lst)

MC      = np.array(MC)
IS      = np.array(IS)
CV      = np.array(CV)
MC_mean = np.nanmean(MC, axis=0)
MC_std  = np.nanstd(MC, axis=0)
IS_mean = np.nanmean(IS, axis=0)
IS_std  = np.nanstd(IS, axis=0)
CV_mean = np.nanmean(CV, axis=0)
CV_std  = np.nanstd(CV, axis=0)
MC_err  = ((MC - ExH)**2)
IS_err  = ((IS - ExH)**2)
CV_err  = ((CV - ExH)**2)
MC_rmse = np.sqrt(np.nanmean(MC_err, axis=0))
IS_rmse = np.sqrt(np.nanmean(IS_err, axis=0))
CV_rmse = np.sqrt(np.nanmean(CV_err, axis=0))

# %%
fig, axs = plt.subplots(1, 2, figsize=(12, 4))

scale = 1.96

axs[0].fill_between(MC_N_lst, MC_mean-scale*MC_std, MC_mean+scale*MC_std,color='b',alpha=0.2,label='MC')
axs[0].fill_between(MC_N_lst, IS_mean-scale*IS_std, IS_mean+scale*IS_std,color='g',alpha=0.2,label='L-BF-IS')
axs[0].fill_between(MC_N_lst, CV_mean-scale*CV_std, CV_mean+scale*CV_std,color='y',alpha=0.2,label='CV')
axs[0].axhline(ExH,c='r',label=r'$\mu^{HF}$')
axs[0].set_xscale('log')
axs[0].set_xlabel('HF Sample Size')
axs[0].set_ylabel('Estimator Value')
axs[0].legend()
axs[0].set_title(r'$\ell=$'+str(l))

axs[1].plot(MC_N_lst, MC_rmse,color='b',alpha=1.0,label='MC')
axs[1].plot(MC_N_lst, IS_rmse,color='g',alpha=1.0,label='L-BF-IS')
axs[1].plot(MC_N_lst, CV_rmse,color='y',alpha=1.0,label='CV')
axs[1].set_xscale('log')
axs[1].set_yscale('log')
axs[1].set_xlabel('HF Sample Size')
axs[1].set_ylabel('RMSE')
axs[1].legend()
axs[1].set_title(r'$\ell=$'+str(l))

fig.tight_layout()
# %%
