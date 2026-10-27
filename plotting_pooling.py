import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from matplotlib.ticker import FormatStrFormatter

# data_size=[100, 500, 1000, 2500, 5000, 10000,'all']

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
    
    parser.add_argument("--srm_k", type=int, required=True)

    parser.add_argument("--zero_shot", action="store_true")
    
    parser.add_argument("--num_random_run", type=int, required=True)


    parser.add_argument("--data_size", nargs="+", type=str, required=True,
                         help="list of sizes e.g. 500 1000 all")

    args = parser.parse_args()
    args.data_size = [_parse_size_token(t) for t in args.data_size]

    return args


args = parse_arguments()

if args.production==0:
    status='comp'
else:
    status='prod'

path=os.getcwd()+f'/results/csv_files_247/pooling/{status}/'

org=[]
srm=[]
pooled=[]
error_org=[]
error_srm=[]
error_pooled=[]
for size in args.data_size:

    if args.zero_shot:
    
        if size=='all':
            filename=path+'srm_augmentation_zero_shot_subjects_k_'+str(args.srm_k)+'_words_'+str(size)+'_random_runs_1.csv'
        else:
            filename=path+'srm_augmentation_zero_shot_subjects_k_'+str(args.srm_k)+'_words_'+str(size)+'_random_runs_'+str(args.num_random_run)+'.csv'

    else:
        if size=='all':
            filename=path+'srm_augmentation_subjects_k_'+str(args.srm_k)+'_words_'+str(size)+'_random_runs_1.csv'
        else:
            filename=path+'srm_augmentation_subjects_k_'+str(args.srm_k)+'_words_'+str(size)+'_random_runs_'+str(args.num_random_run)+'.csv'
    
    df=pd.read_csv(filename)
    cols = ['ww1', 'ww2', 'ww3', 'ww3_control']

    df = df.dropna(subset=cols, how='all')

    subject_summary = df.groupby('subject')[cols].mean().reset_index()
    org.append(df.ww1.mean())
    srm.append(df.ww2.mean())
    pooled.append(df.ww3.mean())

    error_org.append(df.ww1.sem())
    error_srm.append(df.ww2.sem())
    error_pooled.append(df.ww3.sem())

# plt.figure(figsize=(8,5))
indices = np.arange(len(args.data_size))
# Plot grouped bars: 3 bars per subject


bar_width = 0.25

colors=["#fee8c8","#fdbb84","#e34a33"]

fig, ax = plt.subplots(figsize=(12, 7))
ax.bar(indices - bar_width, org, width=bar_width, label='Original (subject-only)', yerr=error_org, color=colors[0])
ax.bar(indices,             srm, width=bar_width, label='eSRM-denoised (subject-only)', yerr=error_srm, color=colors[1])
ax.bar(indices + bar_width, pooled, width=bar_width, label='eSRM-pooling', yerr=error_pooled, color=colors[2])

ax.set_xticks(indices)
ax.set_xticklabels([str(s) for s in args.data_size], rotation=45, ha='right', fontsize=12)
# ax.set_yticklabels([f'{y:.2f}' for y in ax.get_yticks()], fontsize=12)
ax.set_ylabel('Encoding Performance (r)', fontsize=15, fontweight='bold')
ax.set_xlabel('Data Size', fontsize=15, fontweight='bold')
# ax.set_title(f'Per-subject performance {ANCHOR_LEN} words ')
ax.legend(frameon=False, fontsize=12)
ax.grid(axis='y', linestyle='--', alpha=0.4)
ax.set_ylim(0, 0.25)
ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax.tick_params(axis='y', labelsize=12)
fig.tight_layout()
path=os.getcwd()+f'/results/figures_247/'
fig.savefig(path+f'eSRM_pooling_zero_shot_{args.zero_shot}_k_{args.srm_k}_{status}.svg',dpi=600)
    
