#%%
import torch
import numpy as np
from numpy.polynomial.legendre import Legendre
import scipy.io as sio
from itertools import product
import math
import matplotlib.pyplot as plt

#####################################################
#
# build flow over cylinder bi-fidelity data using PCE
#
#####################################################
def generate_total_degree_basis_foc(order, num_variables):
    """
    Generate all possible combinations of exponents in the total degree space.
    """
    if order == 3 and num_variables == 4:
        file = sio.loadmat('../data/flow_pass_cylinder/v4o3.mat')
        total_degree_basis = file['base']
    elif order == 4 and num_variables == 4:
        file = sio.loadmat('../data/flow_pass_cylinder/v4o4.mat')
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

def sample_p(num_samples:int, num_random:int) -> torch.Tensor:
    '''
    Sampling function for sampling density p(x)
    Input: 
        num_samples: number of samples
    Output:
        p_samples: num_samples x 4 uniform random matrix between -1 and 1
    '''
    return torch.rand(num_samples, num_random) * 2 - 1

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
    total_degree_basis = generate_total_degree_basis_foc(order, num_variables)

    # Compute the Polynomial Chaos Expansion using Legendre polynomials in the total degree space.
    pce_results = polynomial_chaos_expansion(p_samples, coefficients, 
                                             total_degree_basis)

    return pce_results

def foc_LF_mean(p_samples):
    '''
    Low-fidelity PCE function for mean QoI
    Input dimension is 4 and order is 3
    '''
    # load coefficient
    file = sio.loadmat('../data/flow_pass_cylinder/c_c_mean.mat')
    coef = torch.from_numpy(file['c_l1']).type(torch.FloatTensor).squeeze()

    return build_pce(p_samples, coef, order=3)

def foc_HF_mean(p_samples):
    '''
    Low-fidelity PCE function for mean QoI
    Input dimension is 4 and order is 3
    '''
    # load coefficient
    file = sio.loadmat('../data/flow_pass_cylinder/c_f_mean.mat')
    coef = torch.from_numpy(file['c_l1']).type(torch.FloatTensor).squeeze()

    return build_pce(p_samples, coef, order=3)

def foc_LF_max(p_samples):
    '''
    Low-fidelity PCE function for max QoI
    Input dimension is 4 and order is 3
    '''
    # load coefficient
    file = sio.loadmat('../data/flow_pass_cylinder/c_c_max.mat')
    coef = torch.from_numpy(file['c_l1']).type(torch.FloatTensor).squeeze()

    return build_pce(p_samples, coef, order=3)

def foc_HF_max(p_samples):
    '''
    Low-fidelity PCE function for max QoI
    Input dimension is 4 and order is 3
    '''
    # load coefficient
    file = sio.loadmat('../data/flow_pass_cylinder/c_f_max.mat')
    coef = torch.from_numpy(file['c_l1']).type(torch.FloatTensor).squeeze()

    return build_pce(p_samples, coef, order=4)

#####################################################
#
# build cavity flow bi-fidelity data using PCE
#
#####################################################

def generate_total_degree_basis(order, num_variables):
    """
    Generate all possible combinations of exponents in the total degree space.
    """
    total_degree_basis = []
    whole_basis = list(product(range(order + 1), repeat=num_variables))
    for i, tpl in enumerate(whole_basis):
        if sum(tpl) <= order:
            total_degree_basis.append(tpl)
    return total_degree_basis

def cav_LF(p_samples):
    '''
    Low-fidelity PCE function for mean QoI
    Input dimension is 4 and order is 3
    '''
    # load coefficient
    file = torch.load('../data/cavity_flow/coefficients.pt')
    coef = file['cL'].type(torch.FloatTensor)
    A = torch.cat((torch.ones(len(p_samples), 1), Legendre.basis(1)(p_samples)), 1)

    return A@coef

def cav_HF(p_samples):
    '''
    Low-fidelity PCE function for mean QoI
    Input dimension is 4 and order is 3
    '''
    # load coefficient
    file = torch.load('../data/cavity_flow/coefficients.pt')
    coef = file['cH'].type(torch.FloatTensor)
    A = torch.cat((torch.ones(len(p_samples), 1), Legendre.basis(1)(p_samples)), 1)

    return A@coef

#####################################################
#
# a one-thousand-dim simulation (otds)
# source: https://www.sciencedirect.com/science/...
# article/pii/S0021999118301955 section 3.3
#
#####################################################

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
def otds_LF(xi):
    M = 2
    return finite_sum(xi, M)

# HF evaluation function
def otds_HF(xi):
    elem = element(xi)
    return elem.exp()

#####################################################
#
# Borehole function bi-fidelity data (separate sample p function)
# source: https://www.sfu.ca/~ssurjano/borehole.html
#
#####################################################

def transform_p_bh(x): # transform xi into correct range
    if x.ndim != 2 or len(x[0]) != 8:
        raise ValueError('Incorrect xi structure.')
    new_x = torch.zeros_like(x)
    new_x[:,0] = 0.10 + 0.0161812 * x[:,0]
    new_x[:,1] = (7.71 + 1.0056 * x[:,1]).exp()
    new_x[:,2] = (115600 - 63070) * x[:,2] + 63070
    new_x[:,3] = (1110 - 990) * x[:,3] + 990
    new_x[:,4] = (116 - 63.1) * x[:,4] + 63.1
    new_x[:,5] = (820 - 700) * x[:,5] + 700
    new_x[:,6] = (1680 - 1120) * x[:,6] + 1120
    new_x[:,7] = (12045 - 9855) * x[:,7] + 9855
    return new_x

def bh_LF(x):
    x = transform_p_bh(x)

    lnrrw  = torch.log(x[:,1]/x[:,0])
    top    = 2 * np.pi * x[:,2] * (x[:,3] - x[:, 5])
    bottom = lnrrw * (1 + 2 * x[:,6] * x[:,2]/(lnrrw * x[:,0] * x[:,7]) + x[:,2]/x[:,4])
    return top/bottom

def bh_HF(x):
    x = transform_p_bh(x)

    lnrrw  = torch.log(x[:,1]/x[:,0])
    top    = 5 * x[:,2] * (x[:,3] - x[:, 5])
    bottom = lnrrw * (1.5 + 2 * x[:,6] * x[:,2]/(lnrrw * x[:,0] * x[:,7]) + x[:,2]/x[:,4])
    return top/bottom

def sample_p_bh(N):
    return torch.hstack((torch.randn(N, 2),torch.rand(N,6)))

#%%
if __name__ == "__main__":

    # global variables
    N = int(1e3)
    titles = ['FlowOverCylinderMean', 'FlowOverCylinderMax', 'CavityFlow', 'Borehole']

    # generate foc mean data
    num_random = 4
    p_samples = sample_p(N, num_random)
    focm_LF = foc_LF_mean(p_samples)
    focm_HF = foc_HF_mean(p_samples)
    focm_ExL = focm_LF.mean().item()
    focm_ExH = focm_HF.mean().item()

    # generate foc max data
    num_random = 4
    p_samples = sample_p(N, num_random)
    focM_LF = foc_LF_max(p_samples)
    focM_HF = foc_HF_max(p_samples)
    focM_ExL = focM_LF.mean().item()
    focM_ExH = focM_HF.mean().item()

    # generate cavity flow data
    num_random = 52
    p_samples = sample_p(N, num_random)
    cav_LF_data = cav_LF(p_samples)
    cav_HF_data = cav_HF(p_samples)
    cav_ExL = cav_LF_data.mean().item()
    cav_ExH = cav_HF_data.mean().item()

    # generate borehole data
    p_samples = sample_p_bh(N)
    bh_LF_data = bh_LF(p_samples)
    bh_HF_data = bh_HF(p_samples)
    bh_ExL = bh_LF_data.mean().item()
    bh_ExH = bh_HF_data.mean().item()

    fig, axs = plt.subplots(2, 2, figsize=(10, 7))
    axs[0, 0].hist([focm_LF,focm_HF],label=[r'$f^{LF}(X)$',r'$f^{HF}(X)$'],color=['tab:blue','tab:red'])
    axs[0, 0].axvline(x = focm_ExL, color = 'tab:blue', label = r'$\mu^{LF}$')
    axs[0, 0].axvline(x = focm_ExH, color = 'tab:red', label = r'$\mu^{HF}$')
    axs[0, 0].set_title(titles[0])
    axs[0, 0].legend()
    axs[0, 1].hist([focM_LF,focM_HF],label=[r'$f^{LF}(X)$',r'$f^{HF}(X)$'],color=['tab:blue','tab:red'])
    axs[0, 1].axvline(x = focM_ExL, color = 'tab:blue', label = r'$\mu^{LF}$')
    axs[0, 1].axvline(x = focM_ExH, color = 'tab:red', label = r'$\mu^{HF}$')
    axs[0, 1].set_title(titles[1])
    axs[0, 1].legend()
    axs[1, 0].hist([cav_LF_data,cav_HF_data],label=[r'$f^{LF}(X)$',r'$f^{HF}(X)$'],color=['tab:blue','tab:red'])
    axs[1, 0].axvline(x = cav_ExL, color = 'tab:blue', label = r'$\mu^{LF}$')
    axs[1, 0].axvline(x = cav_ExH, color = 'tab:red', label = r'$\mu^{HF}$')
    axs[1, 0].set_title(titles[2])
    axs[1, 0].legend()
    axs[1, 1].hist([bh_LF_data,bh_HF_data],label=[r'$f^{LF}(X)$',r'$f^{HF}(X)$'],color=['tab:blue','tab:red'])
    axs[1, 1].axvline(x = bh_ExL, color = 'tab:blue', label = r'$\mu^{LF}$')
    axs[1, 1].axvline(x = bh_ExH, color = 'tab:red', label = r'$\mu^{HF}$')
    axs[1, 1].set_title(titles[3])
    axs[1, 1].legend()
    fig.suptitle('Centered Histograms of LF and HF Data', fontsize=20)
    plt.tight_layout()
    # plt.savefig('../figures/centered_histograms.png', dpi=100)


# fig, axs = plt.subplots(2, 2, figsize=(10, 7))
# focm_corr = np.corrcoef(focm_LF_centered, focm_HF_centered)[0, 1]
# axs[0, 0].scatter(focm_LF_centered, focm_HF_centered, label=f'Correlation = {focm_corr:.2f}', alpha=0.4)
# axs[0, 0].set_title(titles[0])
# axs[0, 0].legend()
# focM_corr = np.corrcoef(focM_LF_centered, focM_HF_centered)[0, 1]
# axs[0, 1].scatter(focM_LF_centered, focM_HF_centered, label=f'Correlation = {focM_corr:.2f}', alpha=0.4)
# axs[0, 1].set_title(titles[1])
# axs[0, 1].legend()
# cav_corr = np.corrcoef(cav_LF_centered, cav_HF_centered)[0, 1]
# axs[1, 0].scatter(cav_LF_centered, cav_HF_centered, label=f'Correlation = {cav_corr:.2f}', alpha=0.4)
# axs[1, 0].set_title(titles[2])
# axs[1, 0].legend()
# bh_corr = np.corrcoef(bh_LF_centered, bh_HF_centered)[0, 1]
# axs[1, 1].scatter(bh_LF_centered, bh_HF_centered, label=f'Correlation = {bh_corr:.2f}', alpha=0.4)
# axs[1, 1].set_title(titles[3])
# axs[1, 1].legend()
# fig.suptitle('Correlation of LF and HF Data', fontsize=20)
# plt.tight_layout()
# plt.savefig('../figures/correlation.png', dpi=100)
# # %%
# fig, axs = plt.subplots(2, 2, figsize=(10, 7))
# focm_log_ratio = torch.abs(focm_HF_centered).log() - torch.abs(focm_LF_centered).log()
# axs[0, 0].hist([focm_log_ratio])
# axs[0, 0].set_title(titles[0])
# focM_log_ratio = torch.abs(focM_HF_centered).log() - torch.abs(focM_LF_centered).log()
# axs[0, 1].hist([focM_log_ratio])
# axs[0, 1].set_title(titles[1])
# axs[0, 1].legend()
# cav_log_ratio = torch.abs(cav_HF_centered).log() - torch.abs(cav_LF_centered).log()
# axs[1, 0].hist([cav_log_ratio])
# axs[1, 0].set_title(titles[2])
# axs[1, 0].legend()
# bh_log_ratio = torch.abs(bh_HF_centered).log() - torch.abs(bh_LF_centered).log()
# axs[1, 1].hist([bh_log_ratio])
# axs[1, 1].set_title(titles[3])
# axs[1, 1].legend()
# fig.suptitle('Histograms of Log Ratio', fontsize=20)
# plt.tight_layout()
# plt.savefig('../figures/log_ratio.png', dpi=100)
# # %%
# fig, axs = plt.subplots(2, 2, figsize=(10, 7))
# focm_corr = np.corrcoef(focm_LF_centered, focm_HF_centered)[0, 1]
# focm_ind = torch.abs(focm_LF_centered) < 1e-1
# axs[0, 0].scatter(focm_LF_centered[~focm_ind], focm_HF_centered[~focm_ind], color='tab:blue', label=f'Correlation = {focm_corr:.2f}', alpha=0.4)
# axs[0, 0].scatter(focm_LF_centered[focm_ind], focm_HF_centered[focm_ind], color='tab:red', alpha=0.4)
# axs[0, 0].set_title(titles[0])
# axs[0, 0].legend()
# focM_corr = np.corrcoef(focM_LF_centered, focM_HF_centered)[0, 1]
# focM_ind = torch.abs(focM_LF_centered) < 5e-1
# axs[0, 1].scatter(focM_LF_centered[~focM_ind], focM_HF_centered[~focM_ind], color='tab:blue', label=f'Correlation = {focM_corr:.2f}', alpha=0.4)
# axs[0, 1].scatter(focM_LF_centered[focM_ind], focM_HF_centered[focM_ind], color='tab:red', alpha=0.4)
# axs[0, 1].set_title(titles[1])
# axs[0, 1].legend()
# cav_corr = np.corrcoef(cav_LF_centered, cav_HF_centered)[0, 1]
# cav_ind = torch.abs(cav_LF_centered) < 5e-2
# axs[1, 0].scatter(cav_LF_centered[~cav_ind], cav_HF_centered[~cav_ind], color='tab:blue', label=f'Correlation = {cav_corr:.2f}', alpha=0.4)
# axs[1, 0].scatter(cav_LF_centered[cav_ind], cav_HF_centered[cav_ind], color='tab:red', alpha=0.4)
# axs[1, 0].set_title(titles[2])
# axs[1, 0].legend()
# bh_corr = np.corrcoef(bh_LF_centered, bh_HF_centered)[0, 1]
# bh_ind = torch.abs(bh_LF_centered) < 10
# axs[1, 1].scatter(bh_LF_centered[~bh_ind], bh_HF_centered[~bh_ind], color='tab:blue', label=f'Correlation = {bh_corr:.2f}', alpha=0.4)
# axs[1, 1].scatter(bh_LF_centered[bh_ind], bh_HF_centered[bh_ind], color='tab:red', alpha=0.4)
# axs[1, 1].set_title(titles[3])
# axs[1, 1].legend()
# fig.suptitle('Correlation of LF and HF Data', fontsize=20)
# plt.tight_layout()
# plt.savefig('../figures/correlation_highlight.png', dpi=100)
# # %%
# # extreme case examples
# fig, axs = plt.subplots(2, 2, figsize=(10, 7))

# x = torch.rand(1000)*2 - 1
# y = torch.rand(1000)*20 - 30
# x_centered = x - x.mean().item()
# y_centered = y - y.mean().item()
# corr = np.corrcoef(x_centered, y_centered)[0, 1]
# eps = 1e-1
# ind = torch.abs(x) < eps
# C = torch.abs(y_centered[ind]).max().item()/eps
# axs[0, 0].scatter(x_centered[~ind], y_centered[~ind], color='tab:blue', label=f'Correlation = {corr:.2f}', alpha=0.4)
# axs[0, 0].scatter(x_centered[ind], y_centered[ind], label=f'C = {C:.2f}', color='tab:red', alpha=0.4)
# axs[0, 0].set_title('Large C',fontsize=14)
# axs[0, 0].set_ylabel('Low correlation',fontsize=14)
# axs[0, 0].legend()

# focM_corr = np.corrcoef(focM_LF_centered, focM_HF_centered)[0, 1]
# eps = 2.0
# focM_ind = torch.abs(focM_LF_centered) < 1.0
# C = torch.abs(focM_HF_centered[focM_ind]).max().item()/eps
# axs[0, 1].scatter(focM_LF_centered[~focM_ind], focM_HF_centered[~focM_ind], color='tab:blue', label=f'Correlation = {focM_corr:.2f}', alpha=0.4)
# axs[0, 1].scatter(focM_LF_centered[focM_ind], focM_HF_centered[focM_ind], label=f'C = {C:.2f}', color='tab:red', alpha=0.4)
# axs[0, 1].set_title('Small C',fontsize=14)
# axs[0, 1].legend()

# x = torch.cat((torch.randn(1000), 0.2*torch.rand(20)-0.1))
# y = torch.cat((30 * x[:1000] - 100 - 10*torch.rand(1000), 50*torch.randn(20)))
# x_centered = x - x.mean().item()
# y_centered = y - y.mean().item()
# eps = 0.1
# ind = torch.abs(x_centered) < eps
# C = torch.abs(y_centered[ind]).max().item()/eps
# corr = np.corrcoef(x_centered, y_centered)[0, 1]
# axs[1, 0].scatter(x_centered[~ind], y_centered[~ind], label=f'Correlation = {corr:.2f}', alpha=0.4, color='tab:blue')
# axs[1, 0].scatter(x_centered[ind], y_centered[ind], alpha=0.4, label=f'C = {C:.2f}', color="tab:red")
# axs[1, 0].set_ylabel('High correlation',fontsize=14)
# axs[1, 0].legend()

# bh_corr = np.corrcoef(bh_LF_centered, bh_HF_centered)[0, 1]
# eps = 10
# bh_ind = torch.abs(bh_LF_centered) < eps
# C = torch.abs(bh_HF_centered[bh_ind]).max().item()/eps
# axs[1, 1].scatter(bh_LF_centered[~bh_ind], bh_HF_centered[~bh_ind], color='tab:blue', label=f'Correlation = {bh_corr:.2f}', alpha=0.4)
# axs[1, 1].scatter(bh_LF_centered[bh_ind], bh_HF_centered[bh_ind], label=f'C = {C:.2f}', color='tab:red', alpha=0.4)
# axs[1, 1].legend()
# fig.suptitle('Correlation vs. Value of C', fontsize=20)
# fig.savefig('../figures/counterexamples.png', dpi=100)

# %%
