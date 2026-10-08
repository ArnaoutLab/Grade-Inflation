from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from scipy.spatial.distance import cdist
import ot
from scipy.stats import pearsonr
import sys
import math
from datetime import datetime
import pyemd
import csv
import os
import scipy.stats as stats
from ucimlrepo import fetch_ucirepo 
from scipy.stats import gaussian_kde
from tqdm import trange
from Eden import FastEdenScorer


real_color, synthetic_color = 'blue','orange'

### loading uciml data ####

def factorize_col(data):
    for col in data.columns:
    
        if data.loc[:,col].dtype == 'object':
            data[col] = pd.factorize(data[col])[0].astype(float)
            print(col, data.loc[:,col].dtype)
    
    return data

def load_uci_data(id, integer_enconding = False, verbose = False):
    """
    Loading data from uci machine learning repo
    """
    dataset = fetch_ucirepo(id = id) 
  
    # data (as pandas dataframes) 
    X = dataset.data.features 
    y = dataset.data.targets
    
    if not y.empty:
        data = pd.concat([X,y], axis = 1)
    else:
        data = X
    
    if verbose:
        # metadata 
        print(f'Dataset loaded: {dataset.metadata.name}') 
        
        # variable information 
        print(dataset.variables) 

    if integer_enconding:
        return factorize_col(data)
    else:
        return data

#### scoring method ####

def pick_score_method(score):
    if score == 'corr':
        get_score = get_corr_similarity
    elif score == 'ems':
        get_score = get_earth_movers_score
    elif score == 'jaccard':
        get_score = get_Jaccard_score
    elif score == 'Eden':
        get_score = get_Eden_score
    elif score == 'KL':
        get_score = get_Renyi_score_in_2d
    elif score == 'CS':
        get_score = get_cs_divergence
    elif score == 'Shannon_Jensson':
        get_score = get_Jensen_Shannon
    else:
        raise KeyError
    return get_score

def get_Eden_score(real_columns, synthetic_columns, n_levels = 5, size = 50000, seed = 42):
    scorer = FastEdenScorer(
        n_of_levels=n_levels,
        thresh=0.05,
        size=size,
        seed=seed,
    )
    result = scorer.score(real_columns, synthetic_columns, return_more=True)
    return result.EdenScore

def get_corr_similarity(real_columns, synthetic_columns):
    try:
        pearsonr_real = pearsonr(real_columns.iloc[:,0], real_columns.iloc[:,1]).statistic
        pearsonr_synthetic = pearsonr(synthetic_columns.iloc[:,0], synthetic_columns.iloc[:,1]).statistic
        correlation_similarity = 1 - (1/2)*abs(pearsonr_real - pearsonr_synthetic)
    except:
        correlation_similarity = np.nan
    
    return correlation_similarity

def get_Jaccard_score(real_columns, synthetic_columns, threshold=0.1):
    """real_columns_without_nans and synthetic_columns_without_nans are supposed to be dataframes with 2 columns each."""
    # would only work when na is dropped, synthetic data doesn't have NAs
    real_columns_without_nans = real_columns.dropna()

    try:
        kernel_real = gaussian_kde(real_columns_without_nans.T)
        kernel_synthetic = gaussian_kde(synthetic_columns.T)

        real_data_evaluated_with_real_kde = kernel_real.evaluate(real_columns_without_nans.T)
        synthetic_data_evaluated_with_synthetic_kde = kernel_synthetic.evaluate(synthetic_columns.T)
        real_data_evaluated_with_synthetic_kde = kernel_synthetic.evaluate(real_columns_without_nans.T)
        synthetic_data_evaluated_with_real_kde = kernel_real.evaluate(synthetic_columns.T)

        real_data_max_likelihood = np.max(real_data_evaluated_with_real_kde)
        synthetic_data_max_likelihood = np.max(synthetic_data_evaluated_with_synthetic_kde)

        real_data_overlap_flag = real_data_evaluated_with_synthetic_kde >= threshold*synthetic_data_max_likelihood
        synthetic_data_overlap_flag = synthetic_data_evaluated_with_real_kde >= threshold*real_data_max_likelihood
        Jaccard_score = (np.sum(real_data_overlap_flag) + np.sum(synthetic_data_overlap_flag))/(len(real_data_overlap_flag) + len(synthetic_data_overlap_flag))
    except:
        Jaccard_score = np.nan
        
    return Jaccard_score

def get_Renyi_score_in_2d(real_columns, synthetic_columns, viewpoint_parameter=1, size=5000000, seed=None):
    """Note: real_columns and synthetic_columns must each consist of 2 columns only."""
    kernel_real = gaussian_kde(real_columns.dropna().T)
    kernel_syn = gaussian_kde(synthetic_columns.dropna().T)

    sample_from_kernel_real = kernel_real.resample(size=size, seed=seed)
    real_likelihoods = kernel_real.evaluate(sample_from_kernel_real)
    syn_likelihoods = kernel_syn.evaluate(sample_from_kernel_real)

    likelihood_ratios = syn_likelihoods/real_likelihoods
    likelihood_ratios = likelihood_ratios[likelihood_ratios > 1e-100] #in case there are zeros 

    if viewpoint_parameter != 1:
        weighted_average_of_likelihood_ratios = np.mean(likelihood_ratios**(1-viewpoint_parameter))
        Renyi_divergence = (1/(viewpoint_parameter-1))*np.log(weighted_average_of_likelihood_ratios)
    else: #in this case the Renyi divergence is the Kullback-Leibler divergence
        Renyi_divergence = np.mean(np.log(1/likelihood_ratios))

    Renyi_score = np.exp(-Renyi_divergence)
    
    return Renyi_score

def get_Jensen_Shannon(real_columns, synthetic_columns, n_samples=500000, seed=None):
    """
    Monte Carlo estimation of Jensen-Shannon divergence between 
    two continuous distributions using KDE density estimates.
    
    Returns: float (JSD in bits when using log2)
    """
    rng = np.random.default_rng(seed)
    
    # Prepare data (ensure 2D arrays)
    X_real = real_columns.dropna().to_numpy()
    X_synth = synthetic_columns.dropna().to_numpy()
    
    if X_real.ndim == 1:
        X_real = X_real[:, None]
    if X_synth.ndim == 1:
        X_synth = X_synth[:, None]
    
    # Fit KDEs
    kde_real = gaussian_kde(X_real.T)
    kde_synth = gaussian_kde(X_synth.T)
    
    # Sample n/2 from each distribution
    half_n = n_samples // 2
    
    samples_real = kde_real.resample(size=half_n, seed=rng.integers(0, 2**32))
    samples_synth = kde_synth.resample(size=half_n, seed=rng.integers(0, 2**32))
    
    def kl_mc(samples, log_p_fn, log_q_fn):
        """Estimate KL(P||Q) = E_P[log p(x) - log q(x)] where samples ~ P"""
        log_ratios = log_p_fn(samples) - log_q_fn(samples)
        return np.mean(log_ratios)
    
    # Log-density functions for both distributions
    log_p_real = lambda x: np.log(kde_real(x) + 1e-300)  # Add tiny epsilon
    log_p_synth = lambda x: np.log(kde_synth(x) + 1e-300)
    
    # Mixture density: M = 0.5*P + 0.5*Q
    log_mixture = lambda x: np.log(0.5 * kde_real(x) + 0.5 * kde_synth(x) + 1e-300)
    
    # Estimate KL(P || M) from samples of P
    kl_pm = kl_mc(samples_real, log_p_real, log_mixture)
    
    # Estimate KL(Q || M) from samples of Q
    kl_qm = kl_mc(samples_synth, log_p_synth, log_mixture)
    
    # JSD = average of the two KL divergences
    jsd = 0.5 * kl_pm + 0.5 * kl_qm
    
    return np.exp(-jsd)

def get_cs_divergence(real_columns, synthetic_columns, bw_method="scott"):
    real = real_columns.dropna().to_numpy().T
    synthetic = synthetic_columns.dropna().to_numpy().T

    kde_real = gaussian_kde(real, bw_method=bw_method)
    kde_syn = gaussian_kde(synthetic, bw_method=bw_method)

    overlap = kde_real.integrate_kde(kde_syn)      # ∫ p_real(x) p_syn(x) dx
    real_norm = kde_real.integrate_kde(kde_real)   # ∫ p_real(x)^2 dx
    syn_norm = kde_syn.integrate_kde(kde_syn)      # ∫ p_syn(x)^2 dx
    cs = -np.log(overlap) + 0.5 * np.log(real_norm) + 0.5 * np.log(syn_norm)

    return np.exp(-cs)

def get_earth_movers_score(data, syn, standardize=True):
    """Score in (0, 1]; closer to 1 = synthetic more similar to real data.

    Computes the exact 1-Wasserstein distance between the empirical joint
    distributions of two 2D arrays (each row = one atom of mass).
    """
    real = np.asarray(data, dtype=np.float64)
    synth = np.asarray(syn, dtype=np.float64)

    # Drop rows containing NaNs (EMD can't handle them)
    real = real[~np.isnan(real).any(axis=1)]
    synth = synth[~np.isnan(synth).any(axis=1)]

    if standardize:
        mu = real.mean(axis=0)
        sd = real.std(axis=0)
        sd[sd == 0] = 1.0
        real = (real - mu) / sd
        synth = (synth - mu) / sd

    n, m = len(real), len(synth)
    dist_matrix = cdist(real, synth)  # n x m, Euclidean ground metric

    emd_value = ot.emd2(
        np.full(n, 1.0 / n),
        np.full(m, 1.0 / m),
        dist_matrix,
    )

    return float(np.exp(-emd_value))

### blind test ####

def plot_kde(data1, syn_gan, inds, method, score = 'corr', seed = None):
    if seed:
        np.random.seed(seed) 

    get_score = pick_score_method(score)

    sns.kdeplot(data=data1.iloc[:,inds], x=data1.iloc[:,inds[0]], y=data1.iloc[:,inds[1]], color=real_color, \
                                levels=5, bw_adjust=1, alpha=0.5,legend=False)
    sns.kdeplot(data=syn_gan.iloc[:,inds], x=syn_gan.iloc[:,inds[0]], y=syn_gan.iloc[:,inds[1]], color=synthetic_color, \
                                levels=5, bw_adjust=1, alpha=0.5,legend=False)
    cols_score = get_score(data1.iloc[:,inds], syn_gan.iloc[:,inds])
    plt.xlabel(f'{method} \n {score} score: {cols_score:.2f}', fontdict={'size':10})
    plt.show()



def blind_test(real, syn1, syn2, inds, pair_n, t_score, flip_color=False, linewidth=2.5, rotate=False, plot = False, path = 'score.csv'):
    if flip_color:
        flip_sign = '-flip-color'
        real_color, synthetic_color = 'tab:red', 'blue'
    else:
        flip_sign = ''
        real_color, synthetic_color = 'blue', 'tab:red'
    
    if rotate:
        inds = [inds[1], inds[0]]
        rotate_sign = '-rotated'
    else:
        rotate_sign = ''
    
    
    if plot:
        x_min = min(real.iloc[:,inds[0]].min(), syn1.iloc[:,inds[0]].min(), syn2.iloc[:,inds[0]].min())
        x_max = max(real.iloc[:,inds[0]].max(), syn1.iloc[:,inds[0]].max(), syn2.iloc[:,inds[0]].max())
        y_min = min(real.iloc[:,inds[1]].min(), syn1.iloc[:,inds[1]].min(), syn2.iloc[:,inds[1]].min())
        y_max = max(real.iloc[:,inds[1]].max(), syn1.iloc[:,inds[1]].max(), syn2.iloc[:,inds[1]].max())
        x_range = x_max - x_min
        y_range = y_max - y_min

        # First subplot

        fig1, ax1 = plt.subplots(figsize=(10, 10))
        fig2, ax2 = plt.subplots(figsize=(10, 10))
        sns.kdeplot(data=real.iloc[:,inds], x=real.iloc[:,inds[0]], y=real.iloc[:,inds[1]], ax=ax1, color=real_color,
                    levels=5, bw_adjust=1, alpha=0.5, legend=False, linewidths=linewidth)
        sns.kdeplot(data=syn1.iloc[:,inds], x=syn1.iloc[:,inds[0]], y=syn1.iloc[:,inds[1]], ax=ax1, color=synthetic_color,
                    levels=5, bw_adjust=1, alpha=0.5, legend=False, linewidths=linewidth)
        
        ax1.set_xlim(x_min - 0.25 * x_range, x_max + 0.25 * x_range)
        ax1.set_ylim(y_min - 0.25 * y_range, y_max + 0.25 * y_range)

        ax1.get_yaxis().set_visible(False)
        ax1.get_xaxis().set_visible(False)
        fig1.savefig(f'/Users/miaoli/Desktop/maxent/blind_test/{pair_n}-a{flip_sign}{rotate_sign}', pad_inches=0, bbox_inches='tight')
        plt.close(fig1)

        # Second subplot

        sns.kdeplot(data=real.iloc[:,inds], x=real.iloc[:,inds[0]], y=real.iloc[:,inds[1]], ax=ax2, color=real_color,
                    levels=5, bw_adjust=1, alpha=0.5, legend=False, linewidths=linewidth)
        sns.kdeplot(data=syn2.iloc[:,inds], x=syn2.iloc[:,inds[0]], y=syn2.iloc[:,inds[1]], ax=ax2, color=synthetic_color,
                    levels=5, bw_adjust=1, alpha=0.5, legend=False, linewidths=linewidth)
        
        ax2.set_xlim(x_min - 0.25 * x_range, x_max + 0.25 * x_range)
        ax2.set_ylim(y_min - 0.25 * y_range, y_max + 0.25 * y_range)

        ax2.get_yaxis().set_visible(False)
        ax2.get_xaxis().set_visible(False)
        fig2.savefig(f'/Users/miaoli/Desktop/maxent/blind_test/{pair_n}-b{flip_sign}{rotate_sign}', pad_inches=0, bbox_inches='tight')
        plt.close(fig2)

    compute_lst = ['corr', 'ems', 'jaccard', 'Eden', 'KL','CS','JSD']

    result1 = []
    for score in compute_lst:
        get_score = pick_score_method(score)
        result1.append(f'{score}: {get_score(real.iloc[:,inds], syn1.iloc[:,inds]):.3f}')

    result2 = []
    for score in compute_lst:
        get_score = pick_score_method(score)
        result2.append(f'{score}: {get_score(real.iloc[:,inds], syn2.iloc[:,inds]):.3f}')
    
    # CSV writing code remains the same
    existing_rows = []
    if os.path.exists(path):
        with open(path, 'r') as f:
            reader = csv.reader(f)
            existing_rows = list(reader)

    new_rows = [
        [f'{pair_n}-a{flip_sign}{rotate_sign}', result1, t_score],
        [f'{pair_n}-b{flip_sign}{rotate_sign}', result2, t_score]
    ]

    all_rows = new_rows + existing_rows

    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(all_rows)


def t_test(real, syn1, syn2, inds, score, round = 50):
    '''
    run t-test to make sure two synthetic data are statistically significantly different
    '''
    result1 = []
    for _ in trange(round):
        get_score = pick_score_method(score)
        result1.append(np.round(get_score(real.iloc[:,inds], syn1.iloc[:,inds]),3))
    
    result2 = []
    for _ in trange(round):
        get_score = pick_score_method(score)
        result2.append(np.round(get_score(real.iloc[:,inds], syn2.iloc[:,inds]),3))
    
    t_stat, p_value = stats.ttest_ind(result1, result2)
    return t_stat, p_value

def run_score_for_pair(real, pair_n, ind, file = 'score.csv', check_p = False):
    syn1 = pd.read_csv(f'csv/{pair_n}_syn1.csv').drop(columns = 'Unnamed: 0')
    syn2 = pd.read_csv(f'csv/{pair_n}_syn2.csv').drop(columns = 'Unnamed: 0')
    if check_p:
        print(t_test(real, syn1, syn2, ind, 'J_q', round=50))
    blind_test(real, syn1, syn2, ind, pair_n, (0), path = file)
    blind_test(real, syn1, syn2, ind, pair_n, (0), rotate=True, path = file)
    blind_test(real, syn1, syn2, ind, pair_n, (0), flip_color=True, path = file)

if  __name__ == '__main__':
    filename = 'blind_test_scores.csv'
    data = load_uci_data(242, integer_enconding=False)
    real = data[['X1','X2', 'X3']]
    run_score_for_pair(real, 1, [1,2], file = filename)
    run_score_for_pair(real, 2, [1,2], file = filename)

    data = load_uci_data(183, integer_enconding=False)
    real = data[['racepctblack','racePctWhite']]
    run_score_for_pair(real, 3, [0,1], file = filename)
    run_score_for_pair(real, 4, [0,1], file = filename)

    real = pd.read_csv('data/made_up_dart_in_2d.csv').iloc[:,1:]
    run_score_for_pair(real, 5, [0,1], file = filename)
    run_score_for_pair(real, 6, [0,1], file = filename)
    run_score_for_pair(real, 7, [0,1], file = filename)

    data = load_uci_data(545, integer_enconding=False)
    real = data[['Eccentricity', 'Extent']]
    run_score_for_pair(real, 8, [0,1], file = filename)
    
    data = load_uci_data(31, integer_enconding=False).sample(5000)
    real = data.loc[:,['Aspect', 'Hillshade_9am', 'Hillshade_3pm']]
    run_score_for_pair(real, 9, [0,1], file = filename)

    real = pd.read_csv('data/trimodal_target.csv')
    run_score_for_pair(real, 10, [0,1], file = filename)

    data = load_uci_data(183, integer_enconding=False)
    real = data.loc[:,['agePct12t21','agePct12t29','agePct16t24', 'agePct65up']]
    run_score_for_pair(real, 11, [1, 3], file = filename)
    run_score_for_pair(real, 12, [1, 3], file = filename)
    run_score_for_pair(real, 13, [0, 1], file = filename)



