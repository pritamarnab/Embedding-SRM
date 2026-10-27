import os
import glob
import pickle
import numpy as np
import pandas as pd  
import csv
from scipy.io import savemat
from scipy.io import loadmat
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from functools import reduce 
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression, RidgeCV
import random
import operator
from itertools import combinations
from itertools import permutations
from numpy.linalg import inv
from datetime import datetime
from nilearn.plotting import plot_markers

from brainiak.funcalign import srm

import brainiak
from scipy.stats import sem
plt.rcParams["svg.fonttype"] = "none"


def load_pickle(file_path):
    
    pickle_file = open(file_path, "rb")
    objects = []

    while True:

        try:

            objects.append(pickle.load(pickle_file))

        except EOFError:

            break

    pickle_file.close()

    a=objects[0]
    
    return a

def pca_reduction(emb_train_org,emb_test_org,sub_index,fold,word_num_train,word_num_test,subjects,pca_length=50):
 
    # emb_train_2=emb_train.copy()
    # emb_test_2=emb_test.copy()

    # del emb_train
    # del emb_test

    pca = PCA(n_components=pca_length)

    emb_train=np.zeros((len(subjects),np.max(word_num_train),pca_length))                  
    emb_test=np.zeros((len(subjects), np.max(word_num_test),pca_length))    
    d=[]
    for sub in range(len(subjects)):

        d11=emb_train_org[sub, fold,:word_num_train[sub][fold],:]- np.mean(emb_train_org[sub,fold,:word_num_train[sub][fold],:], axis=0)
        d.append(d11)

    d= np.vstack(d)
    d1=pca.fit(d)

    for i in range(len(subjects)):

        # print(i)

        d11=emb_train_org[i,fold,:word_num_train[i][fold],:]- np.mean(emb_train_org[i,fold,:word_num_train[i][fold],:], axis=0)
        
        d1=pca.transform(d11)
        d1 = normalize(d1, axis=1, norm='l2')

        emb_train[i,:word_num_train[i][fold],:]=d1
        
        d22=emb_test_org[i,fold,:word_num_test[i][fold],:]- np.mean(emb_train_org[i,fold,:word_num_train[i][fold],:], axis=0)
        d2=pca.transform(d22)
        d2 = normalize(d2, axis=1, norm='l2')
        emb_test[i,:word_num_test[i][fold],:]=d2

    return emb_train,emb_test

# ============================================================
# SRM + Teacher-based Synthetic Augmentation for Short Subject
# ============================================================
# What it does
# ------------
# 1) Fit SRM on well-sampled subjects (excluding the short subject) using encoding weights theta.
# 2) Train a teacher encoder B to predict shared responses Z from embeddings (X -> Z).
# 3) From a fixed contiguous 500-block of the short subject, estimate W0 with a low-variance method.
# 4) Synthesize extra (X, Y) for the short subject using the teacher's Z and W0^T, inject realistic noise.
# 5) Refit theta_i on (real 500) + (synthetic) with down-weighting (OLS), recompute W0 by Procrustes.
# 6) Evaluate your three metrics: ww1 (subject-only), ww2 (subject-only in SRM space), ww3 (pooled).
#
# Notes
# -----
# - Final encoders remain OLS (LinearRegression with fit_intercept=False).
# - Only alignment-side steps use mild regularization internally (ridge for teacher; optional RRR).
# - The script *expects* you already define `pca_reduction(...)` elsewhere (your original function).
#
# Required inputs (NumPy arrays you already have in memory)
# ---------------------------------------------------------
# emb_train_org, emb_test_org, Y_train, Y_test
# word_num_train, word_num_test: lists/arrays of per-subject time counts per fold
# elec_num: number of electrodes per subject
# subjects: list of subject IDs or indices
#
# Output
# ------
# Prints ww1, ww2, ww3 and returns a results dict with W0/S, etc.
#
# ============================================================

import numpy as np

from brainiak.funcalign.srm import SRM

# --------------------------
# Utility functions
# --------------------------

def row_demean_cols(theta):
    """Row-wise demean across columns (treating columns as 'time')."""
    return theta - theta.mean(axis=1, keepdims=True)

def procrustes_W(theta_prep, S):
    """
    Orthogonal Procrustes to get W: argmin ||theta - W S||_F s.t. W^T W=I.
    theta_prep: (E, F), S: (k, F) with row-orthonormal rows ideally.
    """
    A = theta_prep @ S.T  # (E, k)
    U, _, Vt = np.linalg.svd(A, full_matrices=False)
    return U @ Vt  # (E, k)

def polar_orth(Q):
    """Nearest orthogonal via polar decomposition."""
    U, _, Vt = np.linalg.svd(Q, full_matrices=False)
    return U @ Vt

def damp_W(W_old, W_new, eta=0.2):
    """Damped update on Stiefel manifold."""
    return polar_orth((1 - eta) * W_old + eta * W_new)

def mean_corr(Yhat, Ytrue):
    """Mean Pearson correlation across electrodes."""
    E = Yhat.shape[1]
    vals = []
    for e in range(E):
        c = np.corrcoef(Yhat[:, e], Ytrue[:, e])[0, 1]
        vals.append(c)
    return float(np.nanmean(vals))

def normalize_S_shape(S, k_expected):
    """Ensure S is (k, F). BrainIAK typically returns (k, F); else transpose."""
    if S.shape[0] == k_expected:
        return S
    elif S.shape[1] == k_expected:
        return S.T
    else:
        raise ValueError(f"Unexpected S shape {S.shape} for k={k_expected}.")

# --------------------------
# SRM + Teacher
# --------------------------

def fit_srm_and_teacher(emb_train, Y_train, word_num_train, elec_num, subjects,
                        short_subj, fold, lag_index, k=10, rand_seed=0):
    """
    Fit SRM on well-sampled subjects (exclude short subject), then train teacher B: X->Z.
    Z_k = Y_k W_k where W_k is Procrustes(theta_k, S).
    """
    good = [s for s in range(len(subjects)) if s != short_subj]
    thetas = []
    for s in good:
        T = word_num_train[s][fold]
        Xs = emb_train[s, :T, :]
        Ys = Y_train[s, fold, lag_index, :T, :elec_num[s]]
        Ys_c = Ys - Ys.mean(axis=0, keepdims=True)
        beta = LinearRegression(fit_intercept=False).fit(Xs, Ys_c).coef_  # (E,F)
        thetas.append(row_demean_cols(beta))

    srm = SRM(n_iter=100, features=k, rand_seed=rand_seed)
    srm.fit(thetas)
    S = normalize_S_shape(srm.s_, k)  # (k, F)

    # Build pooled (X, Z) for teacher
    X_list, Z_list = [], []
    for idx, s in enumerate(good):
        T = word_num_train[s][fold]
        Xs = emb_train[s, :T, :]
        Ys = Y_train[s, fold, lag_index, :T, :elec_num[s]]
        Ys_c = Ys - Ys.mean(axis=0, keepdims=True)
        Wk = procrustes_W(thetas[idx], S)
        Zk = Ys_c @ Wk  # (T, k)
        X_list.append(Xs); Z_list.append(Zk)

    X_pool = np.vstack(X_list)
    Z_pool = np.vstack(Z_list)

    teacher = RidgeCV(alphas=(1e-3, 1e-2, 1e-1, 1, 10, 100), fit_intercept=True, cv=5)
    teacher.fit(X_pool, Z_pool)

    return S, teacher

# --------------------------
# 500-only W0 (low-variance)
# --------------------------

def estimate_W0_from_500(X500, Y500, S, k, lam=0.1):
    """
    Compute W0 using only the 500 contiguous samples:
      Z = X S^T (T x k)
      W_ls = Y^T Z (Z^T Z + lam I)^-1   (E x k)
      W0 = polar_orth(W_ls)
    """
    Yc = Y500 - Y500.mean(axis=0, keepdims=True)
    Z = X500 @ S.T  # (T, k)
    ZtZ = Z.T @ Z + lam * np.eye(k)
    W_ls = (Yc.T @ Z) @ np.linalg.inv(ZtZ)  # (E, k)
    return polar_orth(W_ls)

# --------------------------
# Synthetic data generation
# --------------------------

def estimate_noise(Y500, X500, W0, S, teacher):
    """
    Estimate residual covariance for subject i from the 500 block:
    Res = (Y - mean) - (teacher(X) W0^T - mean).
    Return a shrinked covariance Sigma.
    """
    Zhat = teacher.predict(X500)            # (T,k)
    Yhat = Zhat @ W0.T                      # (T,E)
    Res = (Y500 - Y500.mean(0, keepdims=True)) - (Yhat - Yhat.mean(0, keepdims=True))
    E = Res.shape[1]
    Sigma = np.cov(Res, rowvar=False)
    lam = 0.05 * np.trace(Sigma) / E
    return Sigma + lam * np.eye(E)

def synthesize_pairs(X_sources, S, teacher, W0, Sigma,
                     n_per_source=None, jitter_std=0.02, conf_tau=0.1, seed=None):
    """
    Generate synthetic (X, Y) for the short subject:
    - Predict Z = teacher(X)
    - Map Y_mu = Z W0^T
    - Add noise ~ N(0, Sigma)
    - Confidence filter via consistency under small X jitter (keep low-variance predictions)
    """
    if seed is not None:
        np.random.seed(seed)

    X_syn_list, Y_syn_list = [], []
    E = Sigma.shape[0]

    for Xs in X_sources:
        if n_per_source is not None and Xs.shape[0] > n_per_source:
            idx = np.random.choice(Xs.shape[0], n_per_source, replace=False)
            Xs = Xs[idx]

        # consistency under jitter
        noise = np.random.normal(0, jitter_std, size=Xs.shape)
        Z1 = teacher.predict(Xs)
        Z2 = teacher.predict(Xs + noise)
        conf = np.mean((Z1 - Z2)**2, axis=1)
        # keep "confident" samples
        thresh = conf_tau * np.median(conf) + 1e-12
        keep = conf <= thresh
        if not np.any(keep):
            continue
        X_keep = Xs[keep]
        Z_keep = teacher.predict(X_keep)
        Y_mu = Z_keep @ W0.T

        # add noise with covariance Sigma
        L = np.linalg.cholesky(Sigma)
        eps = np.random.normal(size=(Y_mu.shape[0], E)) @ L.T
        Y_keep = Y_mu + eps

        X_syn_list.append(X_keep); Y_syn_list.append(Y_keep)

    if not X_syn_list:
        return None, None
    return np.vstack(X_syn_list), np.vstack(Y_syn_list)

# --------------------------
# Theta with synthetic (weighted OLS) and RRR option
# --------------------------

def fit_theta_with_synth(X500, Y500, X_syn, Y_syn,
                         synth_weight=0.2, use_rrr=False, rrr_rank=8):
    """
    Fit theta_i on real 500 + synthetic (down-weighted).
    Final encoders remain OLS; optional RRR further stabilizes theta_i.
    """
    if X_syn is None:
        reg = LinearRegression(fit_intercept=False)
        reg.fit(X500, Y500 - Y500.mean(0, keepdims=True))
        return reg.coef_

    X = np.vstack([X500, X_syn])
    Y = np.vstack([Y500 - Y500.mean(0, keepdims=True),
                   Y_syn - Y_syn.mean(0, keepdims=True)])
    w = np.hstack([np.ones(X500.shape[0]),
                   synth_weight * np.ones(X_syn.shape[0])])

    reg = LinearRegression(fit_intercept=False)
    reg.fit(X, Y, sample_weight=w)
    Theta_ols = reg.coef_  # (E, F)

    if not use_rrr:
        return Theta_ols

    # Reduced-Rank on top
    U, s, Vt = np.linalg.svd(Theta_ols, full_matrices=False)
    r = min(rrr_rank, U.shape[1])
    Theta_rrr = (U[:, :r] @ np.diag(s[:r]) @ Vt[:r, :])
    return Theta_rrr

def drop_non_unique_words(train, test):

    
    # Convert b to a set for fast lookup (O(1))
    b_set = set(test)
    
    # Filter words + track indices
    surviving_words = []
    surviving_indices = []
    
    for idx, word in enumerate(train):
        if word not in b_set:
            surviving_words.append(word)
            surviving_indices.append(idx)

    return surviving_words, surviving_indices

# --------------------------
# Main runner
# --------------------------

def run_srm_synthetic_augmentation(
    emb_train_org, emb_test_org,
    Y_train, Y_test,
    word_num_train, word_num_test, elec_num,
    subjects,
    # You must have your own pca_reduction function defined in your environment:
    # def pca_reduction(emb_train_org, emb_test_org, sub_index, fold, word_num_train, word_num_test):
    pca_reduction_fn, df_word,
    short_subj_index=0, fold=0, lag_index=0,
    k=10, pca_length=50, anchor_len=500,
    lam_z=0.1,              # ridge in Z-space for W0 estimation
    synth_weight=0.2,       # weight for synthetic samples in theta fit
    use_rrr=False, rrr_rank=8,
    n_per_source=2000, jitter_std=0.02, conf_tau=0.1,
    rand_seed=0, verbose=True, zero_shot=False
):
    np.random.seed(rand_seed)
    i = short_subj_index
    ii = lag_index

    # -- PCA reduction using your function (kept as-is)
    emb_train, emb_test = pca_reduction_fn(emb_train_org, emb_test_org, i, fold, word_num_train, word_num_test,subjects, pca_length=pca_length)

    # -- SRM on good subjects + Teacher (X->Z)
    S, teacher = fit_srm_and_teacher(
        emb_train, Y_train, word_num_train, elec_num, subjects,
        short_subj=i, fold=fold, lag_index=ii, k=k, rand_seed=rand_seed
    )

    # -- Fixed contiguous 500 block for short subject
    T_i = word_num_train[i][fold]
    if T_i < anchor_len:
        print(f"Subject {subjects[i]} fold {fold} has only {T_i} training samples, less than anchor_len={anchor_len}. Skipping synthetic augmentation.")
        raise ValueError(f"anchor_len={anchor_len} exceeds available training samples ({T_i}) for subject {i}.")
    elif anchor_len == word_num_train[i][fold]:
        idx500 = np.arange(T_i)  # use all samples if anchor_len equals total

    else:
        start = np.random.randint(0, T_i - anchor_len + 1)
        idx500 = np.arange(start, start + anchor_len)

    if zero_shot:
        train_words = df_word[f'words_{subjects[i]}_fold_{fold}_train'].dropna()
        test_words = df_word[f'words_{subjects[i]}_fold_{fold}_test'].dropna()
        surviving_words, surviving_indices = drop_non_unique_words(train_words, test_words)

        T_i= len(surviving_indices)

        if T_i > anchor_len:

            start = np.random.randint(0, T_i - anchor_len + 1)
            # idx500 = np.arange(start, start + anchor_len)
            idx500=surviving_indices[start: start + anchor_len]      
        else:
            idx500 = surviving_indices.copy()  # use all if not enough unique words


    X500 = emb_train[i, idx500, :]
    Y500 = Y_train[i, fold, ii, idx500, :elec_num[i]]

    # -- Initial W0 from 500 (low-variance Z-space method)
    W0 = estimate_W0_from_500(X500, Y500, S, k, lam=lam_z)

    ## Wk0=W0.copy()  # keep a copy of the initial W0 for the control check in evaluation

    # -- Build X_sources from other subjects' train
    # X_sources = []
    # for s in range(len(subjects)):
    #     if s == i: 
    #         continue
    #     Ts = word_num_train[s][fold]
    #     X_sources.append(emb_train[s, :Ts, :])

    # # -- Estimate subject noise and synthesize extra (X, Y)
    # Sigma = estimate_noise(Y500, X500, W0, S, teacher)
    # X_syn, Y_syn = synthesize_pairs(
    #     X_sources, S, teacher, W0, Sigma,
    #     n_per_source=n_per_source, jitter_std=jitter_std, conf_tau=conf_tau, seed=rand_seed
    # )

    # # -- Theta_i on real+synthetic (weighted), update W0 by Procrustes
    # theta_aug = fit_theta_with_synth(X500, Y500, X_syn, Y_syn,
    #                                  synth_weight=synth_weight, use_rrr=use_rrr, rrr_rank=rrr_rank)
    # W0_new = procrustes_W(row_demean_cols(theta_aug), S)
    # W0 = damp_W(W0, W0_new, eta=0.2)  # eta = 0.2 is a reasonable default for damping

    # ------------- Evaluate your three models -------------
    # Subject-only (OLS)
    Y500_c = Y500 - Y500.mean(0, keepdims=True)
    X_test = emb_test[i, :word_num_test[i][fold], :]
    Y_test_sub = Y_test[i, fold, ii, :word_num_test[i][fold], :elec_num[i]]
    Y_test_sub = Y_test_sub - Y500.mean(0, keepdims=True)

    reg1 = LinearRegression(fit_intercept=False).fit(X500, Y500_c)
    Y_pred_org = reg1.predict(X_test)

    # Subject-only in W0-projected space
    
    Y500_srm = (Y500_c @ W0) @ W0.T
    Y500_srm = Y500_srm - Y500_srm.mean(0, keepdims=True)

    ## if we dont update W0 by Procrustes with the new theta, and just use the initial W0 from the 500-block, how does that do?
    Wk0 = procrustes_W(row_demean_cols(reg1.coef_), S)
    Y500_srm_2 = (Y500_c @ Wk0) @ Wk0.T
    Y500_srm_2 = Y500_srm_2 - Y500_srm_2.mean(0, keepdims=True)

    reg2 = LinearRegression(fit_intercept=False).fit(X500, Y500_srm_2)
    Y_pred_srm_single = reg2.predict(X_test)

    # Augment with other subjects mapped into i's electrode space (your ww3 path)
    X_train_aug = X500.copy()
    Y_train_aug_srm = Y500_srm.copy()
    Y_train_aug_srm_without_new_method = Y500_srm_2.copy()  # for the control check without the new W0 update

    # print("Norm of difference between W0 and Wk0:", np.linalg.norm(W0 - Wk0, 'fro'))

    for s in range(len(subjects)):
        if s == i: 
            continue
        # print(s)
        Ts = np.arange(0, word_num_train[s][fold])

        if zero_shot:
            train_words = df_word[f'words_{subjects[s]}_fold_{fold}_train'].dropna()
            surviving_words, surviving_indices = drop_non_unique_words(train_words, test_words)
            Ts= surviving_indices.copy()


        Xs = emb_train[s, Ts, :]
        Ys = Y_train[s, fold, ii, Ts, :elec_num[s]]
        Ys_c = Ys - Ys.mean(0, keepdims=True)
        # W_k via Procrustes(theta_k, S)
        beta_s = LinearRegression(fit_intercept=False).fit(Xs, Ys_c).coef_
        Wk = procrustes_W(row_demean_cols(beta_s), S)
        X_train_aug = np.vstack([X_train_aug, Xs])

        ## with W0 updated by synthetic data augmentation
        Ys_to_i = (Ys_c @ Wk) @ W0.T
        Ys_to_i = Ys_to_i - Ys_to_i.mean(0, keepdims=True)
        Y_train_aug_srm = np.vstack([Y_train_aug_srm, Ys_to_i])

        # control check: if we use to augmentation via synthetic data and use Wk0 (OLS-based) instead of W0, do we see improvement over the single-subject model?
        Ys_to_i_ols = (Ys_c @ Wk) @ Wk0.T
        Ys_to_i_ols = Ys_to_i_ols - Ys_to_i_ols.mean(0, keepdims=True)
        Y_train_aug_srm_without_new_method = np.vstack([Y_train_aug_srm_without_new_method, Ys_to_i_ols])

        
        
        
    # Final pooled centering with syhtnetic data augmentation
    Y_train_aug_srm_c = Y_train_aug_srm - Y_train_aug_srm.mean(0, keepdims=True)
    Y_test_srm2 = (Y_test_sub @ W0) @ W0.T
    Y_test_srm2 = Y_test_srm2 - Y_train_aug_srm_c.mean(0, keepdims=True)

    reg3 = LinearRegression(fit_intercept=False).fit(X_train_aug, Y_train_aug_srm_c)
    Y_pred_aug = reg3.predict(X_test)

    

    #control check for the augmentation with Wk0 instead of W0
    Y_train_aug_srm_without_new_method_c = Y_train_aug_srm_without_new_method - Y_train_aug_srm_without_new_method.mean(0, keepdims=True)
    Y_test_srm_control = (Y_test_sub @ Wk0) @ Wk0.T
    Y_test_srm_control = Y_test_srm_control - Y_train_aug_srm_without_new_method.mean(0, keepdims=True)
    
    reg3_control = LinearRegression(fit_intercept=False).fit(X_train_aug, Y_train_aug_srm_without_new_method_c)
    
    
    
    Y_pred_aug_control = reg3_control.predict(X_test)

    ww1 = mean_corr(Y_pred_org, Y_test_sub)
    ww2 = mean_corr(Y_pred_srm_single, (Y_test_sub @ Wk0) @ Wk0.T - Y500_srm_2.mean(0, keepdims=True))
    ww3 = mean_corr(Y_pred_aug, Y_test_srm2)
    ww3_control = mean_corr(Y_pred_aug_control, Y_test_srm_control)

    if verbose:
        print(f"[Synthetic augmentation] k={k}, anchor_len={anchor_len}, synth_w={synth_weight}, RRR={use_rrr}")
        print(f"ww1={ww1:.4f}, ww2={ww2:.4f}, ww3_control={ww3_control:.4f}")

    return {
        
        "ww1": ww1, "ww2": ww2,  "ww3": ww3, "ww3_control": ww3_control,
        "W0": W0, "S": S,
        # "X_syn_shape": None if X_syn is None else X_syn.shape
    }


