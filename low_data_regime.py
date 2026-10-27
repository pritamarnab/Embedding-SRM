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
from sklearn.linear_model import LinearRegression
import random
import operator
from itertools import combinations
from itertools import permutations
from numpy.linalg import inv
from datetime import datetime
from nilearn.plotting import plot_markers
# from brainiak.utils import utils
from brainiak.funcalign import srm
# from brainiak.funcalign import rsrm
import brainiak
from scipy.stats import sem
# from utils import *
from utils import load_pickle, pca_reduction, run_srm_synthetic_augmentation
# ...existing code...
import argparse
plt.rcParams["svg.fonttype"] = "none"

def parse_arguments():
    """Read commandline arguments

    Returns:
        args (Namespace): input as well as default arguments
    """

    def _parse_size_token(t):
        if isinstance(t, str) and t.lower() == 'all':
            return 'all'
        try:
            return int(t)
        except Exception:
            parser.error(f"Invalid --data_size token: {t}")


    parser = argparse.ArgumentParser()
    
    # main args
    parser.add_argument("--production", type=int, required=True)
    parser.add_argument("--subject",  nargs="+", type=int, required=True) 
    parser.add_argument("--lags",  nargs="+", type=int, required=True)
    parser.add_argument("--folds", type=int, required=True) 
    # parser.add_argument("--elec_list", type=str, required=True)
    # parser.add_argument("--random_seed", type=int, required=False, default=0)
    parser.add_argument("--srm_k", type=int, required=True)

    parser.add_argument("--zero_shot", action="store_true")
    # parser.add_argument("--mat_file_saving", action="store_true")
    parser.add_argument("--num_random_run", type=int, required=True)
    # parser.add_argument("--data_size", nargs="+", type=int, required=True)

    parser.add_argument("--data_size", nargs="+", type=str, required=True,
                         help="list of sizes e.g. 500 1000 all")
    
    
    # parser.add_argument("--proportion_analysis", type=str, required=False, default='percentage')
    # parser.add_argument("--window_length", type=int, required=True)
    # parser.add_argument("--desired_fold", type=int, required=True)
    # parser.add_argument("--data-dir", type=str, required=True)
    # parser.add_argument("--saving-dir", type=str, required=True)
    # parser.add_argument("--freeze-decoder", action="store_true")
    # parser.add_argument("--training-step", type=int, default="3000")
    # parser.add_argument("--z_score", action="store_true")
    # parser.add_argument("--make_position_embedding_zero", action="store_true")

    args = parser.parse_args()
    args.data_size = [_parse_size_token(t) for t in args.data_size]

    return args


args = parse_arguments()

print("Running SRM-augmentation for low data regime with arguments:")
for arg, value in vars(args).items():
    print(f"  {arg}: {value}")

if args.production==0:
    status='comp'
else:
    status='prod'

path='/scratch/gpfs/HASSON/arnab/Embedding-SRM/results/pickle_files/'


file_path=path+'data_zscore_717_out_with_words_new_elec_id_gpt2-xl_lag_'+str(args.lags)+'_'+status+'.pkl'

db=load_pickle(file_path)
Y_train=db['elec_train'] 
Y_test=db['elec_test'] 
emb_train_org= db['emb_train']
emb_test_org= db['emb_test']
word_num_train=db['word_num_train']
word_num_test =db['word_num_test']
elec_num=db['elec_num']
df_word=db['df_words']




# ---- CONFIG: tune if needed ----
K_SHARED     = args.srm_k     # k for SRM (you can try 8 or 10)
ANCHOR_LEN   = args.data_size[0]    # contiguous samples for short subject
LAM_Z        = 0.2     # ridge in Z-space for W0 estimation (stabilizes 500-only)
SYNTH_WEIGHT = 0.05     # weight for synthetic samples in theta fit
USE_RRR      = False    # use reduced-rank for theta_i stabilization (alignment step only)
RRR_RANK     = 5
N_PER_SOURCE = None    # None -> use all source samples
JITTER_STD   = 0.005   # teacher consistency jitter
CONF_TAU     = 0.05     # looser confidence gate (increase if still strict)

N_FOLDS      = args.folds
pca_length   = 50   # make sure your pca_reduction uses this global or its own internal setting
subjects     =args.subject

if ANCHOR_LEN == 'all':
    args.num_random_run = 1  # only need to run once if using all data, since no randomness in synthetic generation


# proportions=[500]  #1000,3000, 5000, 7000, 10000,'all']

# ---- Safety check: bail out early if prerequisites are missing ----
required_symbols = [
    'emb_train_org', 'emb_test_org', 'Y_train', 'Y_test',
    'word_num_train', 'word_num_test', 'elec_num', 'subjects',
    'pca_reduction', 'run_srm_synthetic_augmentation'
]
missing = [s for s in required_symbols if s not in globals()]
if missing:
    raise NameError(f"Missing in environment: {missing}. "
                    "Load your data, define pca_reduction and run_srm_synthetic_augmentation first.")

# -------------------
# Run across subjects × folds
# -------------------
records = []
for kk in range(args.num_random_run):

    RAND_SEED    = np.random.randint(0, 100000)  # for reproducibility of synthetic data generation


    for si, subj in enumerate(subjects):
        for fold in range(N_FOLDS):

            if args.data_size[0] == 'all':
                ANCHOR_LEN=word_num_train[si][fold]

            # If this subject-fold lacks enough training samples for a 500-block, record NaNs
            if word_num_test[si][fold] < 1000:
                records.append({'subject': subj, 'fold': fold, 'ww1': np.nan, 'ww2': np.nan, 'ww3': np.nan, 'ww3_control': np.nan})
                continue
    
            res = run_srm_synthetic_augmentation(
                emb_train_org, emb_test_org,
                Y_train, Y_test,
                word_num_train, word_num_test, elec_num,
                subjects,
                pca_reduction_fn=pca_reduction,
                df_word=df_word,
                short_subj_index=si, fold=fold, lag_index=0,
                k=K_SHARED, pca_length=50, anchor_len=ANCHOR_LEN,
                lam_z=LAM_Z, synth_weight=SYNTH_WEIGHT,
                use_rrr=USE_RRR, rrr_rank=RRR_RANK,
                n_per_source=N_PER_SOURCE, jitter_std=JITTER_STD, conf_tau=CONF_TAU,
                rand_seed=RAND_SEED, verbose=False, zero_shot=args.zero_shot
            )
            records.append({
                'subject': subj, 'fold': fold,
                'ww1': res['ww1'], 'ww2': res['ww2'],  'ww3': res['ww3'], 'ww3_control': res['ww3_control'],
                'rand_seed': RAND_SEED, 'fold': fold
            })

# -------------------
# Aggregate by subject (mean over folds), then plot
# -------------------
results_df = pd.DataFrame(records)
path2=os.getcwd()+'/results/csv_files_247/pooling/'+status

if not os.path.exists(path2):
    os.makedirs(path2)

if args.zero_shot:
    results_df.to_csv(path2+f'/srm_augmentation_zero_shot_subjects_k_{K_SHARED}_words_{args.data_size[0]}_random_runs_{args.num_random_run}.csv', index=False)
else:
    results_df.to_csv(path2+f'/srm_augmentation_subjects_k_{K_SHARED}_words_{args.data_size[0]}_random_runs_{args.num_random_run}.csv', index=False)


    

# subject_summary = results_df.groupby('subject')[['ww1','ww2','ww3','ww3_control','rand_seed']].mean().reset_index()

# # Plot grouped bars: 3 bars per subject
# subjects_order = subject_summary['subject'].tolist()
# indices = np.arange(len(subjects_order))
# bar_width = 0.25

# fig, ax = plt.subplots(figsize=(max(8, 1.2*len(subjects_order)), 5))
# ax.bar(indices - bar_width, subject_summary['ww1'], width=bar_width, label='original (subject-only)')
# ax.bar(indices,             subject_summary['ww2'], width=bar_width, label='srm-denoised (subject-only)')
# ax.bar(indices + bar_width, subject_summary['ww3'], width=bar_width, label='all sub pooled + SRM')

# ax.set_xticks(indices)
# ax.set_xticklabels([str(s) for s in subjects_order], rotation=45, ha='right')
# ax.set_ylabel('Mean correlation (across folds)')
# ax.set_title(f'Per-subject performance {ANCHOR_LEN} words ')
# ax.legend(frameon=False)
# ax.grid(axis='y', linestyle='--', alpha=0.4)
# fig.tight_layout()
# path='/scratch/gpfs/HASSON/arnab/model_based_srm/result/247/figures/'
# # fig.savefig(path+f'encoding_all_sub_new_method_k_{K_SHARED}_words_{ANCHOR_LEN}.png',dpi=600)
# # Optional: print the table for quick inspection
# print(subject_summary)