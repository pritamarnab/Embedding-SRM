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
from sklearn.preprocessing import StandardScaler
import random
from nilearn.plotting import plot_markers
from itertools import combinations
from itertools import permutations
from numpy.linalg import inv
from brainiak.utils import utils
from brainiak.funcalign import srm
from brainiak.funcalign import rsrm
import brainiak
import argparse

# folds=2

def parse_arguments():
    """Read commandline arguments

    Returns:
        args (Namespace): input as well as default arguments
    """

    parser = argparse.ArgumentParser()
    
    # main args
    parser.add_argument("--production", type=int, required=True)
    parser.add_argument("--subject",  nargs="+", type=int, required=True) 
    parser.add_argument("--total_conv_numbers",  nargs="+", type=int, required=True)
    parser.add_argument("--lags",  nargs="+", type=int, required=True)
    parser.add_argument("--original_srm_encoding", action="store_true")
    parser.add_argument("--across_subject_encoding", action="store_true")
    parser.add_argument("--across_subject_shared_space", action="store_true")
    parser.add_argument("--folds", type=int, required=True) 
    parser.add_argument("--selected_elec_id", action="store_true") 
    parser.add_argument("--across_subject_with_replacing_srm", action="store_true")
    parser.add_argument("--across_subject_with_various_train_data_proportion", action="store_true")
    parser.add_argument("--encoding_with_all_subjects_data", action="store_true")
    parser.add_argument("--elec_list", type=str, required=True)
    parser.add_argument("--random_seed", type=int, required=False, default=0)
    parser.add_argument("--srm_k", type=int, required=True)
    parser.add_argument("--plotting", action="store_true")
    parser.add_argument("--mat_file_saving", action="store_true")
    parser.add_argument("--proportion_analysis", type=str, required=False, default='percentage')
    parser.add_argument("--window_length", type=int, required=True)
    # parser.add_argument("--desired_fold", type=int, required=True)
    # parser.add_argument("--data-dir", type=str, required=True)
    # parser.add_argument("--saving-dir", type=str, required=True)
    # parser.add_argument("--freeze-decoder", action="store_true")
    # parser.add_argument("--training-step", type=int, default="3000")
    # parser.add_argument("--z_score", action="store_true")
    # parser.add_argument("--make_position_embedding_zero", action="store_true")

    args = parser.parse_args()

    return args


args = parse_arguments()

subjects=args.subject    #[625, 676, 7170, 798]
model_name='gpt2-xl'
lags=args.lags
min_num_word=5
srm_features=args.srm_k


pca_length=50
production=args.production

if production==1:
    status='prod'
else:
    status='comp'

# Set the parameter
plt.rcParams["svg.fonttype"] = "none"

folds=args.folds
across_subject_encoding=args.across_subject_encoding
original_srm_encoding=args.original_srm_encoding
across_subject_shared_space=args.across_subject_shared_space
selected_elec_id=args.selected_elec_id
across_subject_with_replacing_srm=args.across_subject_with_replacing_srm
across_subject_with_various_train_data_proportion=args.across_subject_with_various_train_data_proportion
encoding_with_all_subjects_data=args.encoding_with_all_subjects_data

if original_srm_encoding:

    filename_mat='csrm_zscore2_data_org_srm_new_elec_id_'+status+'_fold_'+str(folds)+'_lag_'+str(lags)+'_srm_k_'+str(srm_features)+'.mat'

if across_subject_encoding:

    filename_mat='csrm_zscore2_data_across_sub_erp_new_elec_id_'+status+'_fold_'+str(folds)+'_lag_'+str(lags)+'_srm_k_'+str(srm_features)+'.mat'
    
if across_subject_shared_space:

    filename_mat='csrm_ecog_across_sub_shared_space_'+status+'_fold_'+str(folds)+'_lag_'+str(lags)+'_srm_k_'+str(srm_features)+'.mat'
if across_subject_with_replacing_srm:

    filename_mat='csrm_zscore2_data_across_sub_with_replace_new_elec_id_'+status+'_fold_'+str(folds)+'_lag_'+str(lags)+'_srm_k_'+str(srm_features)+'.mat'

if across_subject_with_various_train_data_proportion:
    path='/scratch/gpfs/HASSON/arnab/model_based_srm/result/247/train_data_diff_proportion/'
    filename_mat=path+'csrm_zscore2_data_across_sub_with_various_train_data_proportion_new_elec_id_'+status+'_fold_'+str(folds)+'_lag_'+str(lags)+'_random_run_'+str(random_seed)+'.mat'

if encoding_with_all_subjects_data:
    path='/scratch/gpfs/HASSON/arnab/model_based_srm/result/247/encoding_all_subject/'
    filename_mat=path+'csrm_zscore2_data_encoding_with_717_out_data_new_elec_id_'+status+'_fold_'+str(folds)+'_lag_'+str(lags)+'_srm_k_'+str(srm_features)+'_proportion_method_'+str(args.proportion_analysis)+'_random_run_'+str(random_seed)+'.mat'
    

print('Status:',status)
print(args)

# original=[]
# srm_denoise=[]
# across_sub_srm_no_isc=[]
# across_sub_srm_isc=[]
# across_sub_no_srm=[]

a=loadmat('/scratch/gpfs/HASSON/arnab/model_based_srm/elec_id.mat')
index=a['index']


def load_pickle(file_path):
    
    pickle_file = open(file_path, "rb")
    objects = []

    i=0

    while True:
        # print('i',i)
        try:

            objects.append(pickle.load(pickle_file))

        except EOFError:

            break

    pickle_file.close()

    a=objects[0]
    
    return a

def load_label(filepath):

    with open(filepath, "rb") as f:
        full_labels = pickle.load(f)
        labels_df = pd.DataFrame(full_labels)

    return labels_df

# getting base df
def base_df_load(subject,model_name,min_num_word):

    path_base_df='/scratch/gpfs/HASSON/kw1166/247/247-pickling/results/tfs/'  #625
    path_base_df=path_base_df+str(subject)+'/pickles/embeddings/'+model_name+'/full/base_df.pkl'
    
    pickle_file = open(path_base_df, "rb")
    objects = []
    
    i=0
    
    while True:
        # print('i',i)
        try:
    
            objects.append(pickle.load(pickle_file))
    
        except EOFError:
    
            break
    
    pickle_file.close()
    
    a=objects[0]
    df=pd.DataFrame(a)
    df=df.loc[(~df.adjusted_onset.isnull()) & (df.token_idx==0) & (df.is_nonword==False)]

    df=df[df.num_words>min_num_word]
    
    # inserting last word_index
    df['last_word_index']=0
    sentence_id=(df.sentence_idx.values)
    a1=list(sentence_id)
    a2=list(sentence_id[1:])
    a2.append(1)
    a3=np.asarray(a2)-np.asarray(a1)
    id=[]
    for x in a3:
        if x==0:
            id.append(0)
        else:
            id.append(1)
    
    id=np.asarray(id)
    
    df['last_word_index']=id  
   
    return df
def get_elec_id(production,subject,args):

    df=pd.DataFrame()
      

    if args.elec_list=='old_path':
        #old path
        path='/projects/HASSON/247/plotting/sig-elecs/20230510-tfs-sig-file/'    
        sig_file=path+'tfs-sig-file-glove-'+ str(subject)+'-'+status+'.csv'

    elif args.elec_list=='new_path':
        #new path
        path='/home/kw1166/scratch/247/247-plotting/data/plotting/sig-elecs/20230723-tfs-sig-file/'    
        sig_file=path+'tfs-sig-file-glove-'+ str(subject)+'-'+status+'.csv'
        # sig_file=path+'tfs-sig-file-glove-'+ str(subject)+'-'+'comp'+'.csv'

    elif args.elec_list=='erp-corr':
        sig_file='/scratch/gpfs/HASSON/arnab/247_data_read/result/significant_elec_list_erp_corr/'+'erp_corr_elec_list_'+str(subject)+'_'+status+'_threshold_50.csv'

        df_sig=pd.read_csv(sig_file)
        elec_id=list(df_sig.electrode_mat_file_num.values)

        return elec_id
        
    df_sig=pd.read_csv(sig_file)
    elecs=df_sig.electrode.values
    
    path='/scratch/gpfs/HASSON/kw1166/247/247-pickling/results/tfs/'+str(subject)+'/pickles/'+str(subject)+'_electrode_names.pkl'
    
    pickle_file = open(path, "rb")
    objects = []
    
    i=0
    
    while True:
        # print('i',i)
        try:
    
            objects.append(pickle.load(pickle_file))
    
        except EOFError:
    
            break
    
    pickle_file.close()
    
    a=objects[0]
    
    df_name_match=pd.DataFrame(a)
    
    elec_id=[]
    for elec in elecs:
        elec_id.extend(df_name_match[df_name_match.electrode_name==elec].electrode_id.values)

    df['electrode_name']=elecs
    df['electrode_id']=elec_id
    df['subject']=subject

    if subject==7170:
        subject=717

    coordinate_csv_path='/projects/HASSON/247/plotting/brainplot/'+str(subject)+'/'+str(subject)+'-electrode-coordinates.csv'
    df_coord=pd.read_csv(coordinate_csv_path)

    # Perform the merge and handle unmatched rows gracefully
    columns_to_keep = ['MNI_X', 'MNI_Y', 'MNI_Z', 'type', 'reg1']
    df = pd.merge(
        df,
        df_coord[columns_to_keep + ['name']],
        how='left',
        left_on='electrode_name',
        right_on='name',
        suffixes=('', '_coord')
    )

    # parsopercularis=IFG
    # temporalpole=TP 
    # supramarginal=AG
    # cSTG=STG 
    # rSTG=STG 
    # mSTG=STG 
    # parstriangularis=IFG
    # rostralmiddlefrontal= MFG
    # inferiorparietal=AG
    # postcentral=SM 
    # caudalmiddlefrontal=MFG
    # # bankssts=STG

    # ...existing code...
    # map region names to short area codes (case-insensitive)
    mapping = {
        'parsopercularis': 'IFG',
        'parstriangularis': 'IFG',
        'temporalpole': 'TP',
        'supramarginal': 'AG',
        'inferiorparietal': 'AG',
        'cSTG': 'STG',
        'rSTG': 'STG',
        'mSTG': 'STG',
        'rostralmiddlefrontal': 'MFG',
        'caudalmiddlefrontal': 'MFG',
        'postcentral': 'SM',
        'precentral': 'M',
    }

    # apply mapping in a case-insensitive way, keep original value if no match
    df['reg1'] = df['reg1'].astype(str).str.strip()
    df['reg1'] = df['reg1'].apply(lambda x: mapping.get(x, x))



    # Check for rows with missing values after the merge
    if df[columns_to_keep].isnull().any().any():
        print("Warning: Some rows in df have missing values after the merge.")

    # Drop the 'name' column from df
    df = df.drop(columns=['name'])
                      
    return elec_id, df



path='/scratch/gpfs/HASSON/arnab/Embedding-SRM/results/pickle_files/'

file_path=path+'data_zscore_717_out_with_words_new_elec_id_gpt2-xl_lag_'+str(lags)+'_'+status+'.pkl'


db=load_pickle(file_path)
Y_train=db['elec_train'] 
Y_test=db['elec_test'] 
emb_train_org= db['emb_train']
emb_test_org= db['emb_test']
word_num_train=db['word_num_train']
word_num_test =db['word_num_test']
elec_num=db['elec_num']


# breakpoint()

def pca_reduction(emb_train_org,emb_test_org,sub_index,fold,word_num_train,word_num_test):
 
    # emb_train_2=emb_train.copy()
    # emb_test_2=emb_test.copy()

    # del emb_train
    # del emb_test

    pca = PCA(n_components=pca_length)

    emb_train=np.zeros((len(subjects),np.max(word_num_train),pca_length))                  
    emb_test=np.zeros((len(subjects), np.max(word_num_test),pca_length))    


    d11=emb_train_org[sub_index, fold,:word_num_train[sub_index][fold],:]- np.mean(emb_train_org[sub_index,fold,:word_num_train[sub_index][fold],:], axis=0)
    d1=pca.fit(d11)

    for i in range(len(subjects)):

        # print(i)

        # d11=emb_train_org[i,fold,:word_num_train[i][fold],:]- np.mean(emb_train_org[i,fold,:word_num_train[i][fold],:], axis=0)
        d11=emb_train_org[i,fold,:word_num_train[i][fold],:]- np.mean(emb_train_org[sub_index,fold,:word_num_train[sub_index][fold],:], axis=0)
        
        d1=pca.transform(d11)
        d1 = normalize(d1, axis=1, norm='l2')

        emb_train[i,:word_num_train[i][fold],:]=d1
        
        # d22=emb_test_org[i,fold,:word_num_test[i][fold],:]- np.mean(emb_train_org[i,fold,:word_num_train[i][fold],:], axis=0)
        d22=emb_test_org[i,fold,:word_num_test[i][fold],:]- np.mean(emb_train_org[sub_index,fold,:word_num_train[sub_index][fold],:], axis=0)
        d2=pca.transform(d22)
        d2 = normalize(d2, axis=1, norm='l2')
        emb_test[i,:word_num_test[i][fold],:]=d2

    return emb_train,emb_test

def pca_and_srm(emb_train_org,emb_test_org,sub_index,fold,word_num_train,word_num_test,lag_index, Y_train,Y_test,srm_features):

    print('pca and srm')
 
    # emb_train_2=emb_train.copy()
    # emb_test_2=emb_test.copy()

    # del emb_train
    # del emb_test
  

    pca = PCA(n_components=pca_length)

    emb_train=np.zeros((len(subjects),np.max(word_num_train),pca_length))                  
    emb_test=np.zeros((len(subjects), np.max(word_num_test),pca_length))    

    p=0

   
    for i in range(len(subjects)): 

        if i!=sub_index:

            if p==0:

                emb=(emb_train_org[i, fold,:word_num_train[i][fold],:])
                p=p+1
            else:
                emb=np.concatenate((emb,emb_train_org[i, fold,:word_num_train[i][fold],:]),axis=0)

    
    emb=emb-np.mean(emb,axis=0)
    emb=pca.fit_transform(emb)
    
    for i in range(len(subjects)):

        if i!=sub_index:
            emb_train[i,:word_num_train[i][fold],:]=pca.transform(emb_train_org[i, fold,:word_num_train[i][fold],:])
            emb_test[i,:word_num_test[i][fold],:]=pca.transform(emb_test_org[i, fold,:word_num_test[i][fold],:])
        else:
            emb_train[i,:word_num_train[i][fold],:]=pca.transform(emb_train_org[i, fold,:word_num_train[i][fold],:])
            emb_test[i,:word_num_test[i][fold],:]=pca.transform(emb_test_org[i, fold,:word_num_test[i][fold],:])

    print('done pca')

    train_data=[]
    # emb_train=np.zeros((len(subjects),folds, np.max(word_num_train),emb_length)) 
    # Y_train=np.zeros((len(subjects), folds, len(lags),np.max(word_num_train),max(elec_num)))

    for i in range(len(subjects)):
        
        if i!=sub_index:
            Y1=Y_train[i,fold, lag_index,:word_num_train[i][fold],:elec_num[i]]-np.mean(Y_train[i,fold, lag_index,:word_num_train[i][fold],:elec_num[i]],axis=0)
            reg1 = LinearRegression().fit(emb_train[i,:word_num_train[i][fold],:],Y1)
            theta1=(reg1.coef_)
            train_data.append(theta1)

        else:
            Y1=Y_train[i,fold, lag_index,:word_num_train[i][fold],:elec_num[i]]-np.mean(Y_train[i,fold, lag_index,:word_num_train[i][fold],:elec_num[i]],axis=0)
            reg1 = LinearRegression().fit(emb_train[i,:word_num_train[i][fold],:],Y1)
            theta1=(reg1.coef_)
            test_data=theta1

    srm = brainiak.funcalign.srm.SRM(n_iter=100, features=srm_features)
    srm.fit(train_data)

    w_new_subject=srm.transform_subject(test_data) 

    print('done srm')    

    p=0
    q=0


    for i in range(len(subjects)):
        if i!=sub_index:
            Y1=np.matmul(Y_train[i,fold, lag_index,:word_num_train[i][fold],:elec_num[i]],srm.w_[q])
            if p==0:
                Y11=Y1
                p=p+1
            else:
                Y11=np.concatenate((Y11,Y1),axis=0)    
            q=q+1
    
    Y11=np.matmul(Y11,w_new_subject.T)

    # Y2=np.matmul(np.matmul(Y_train[sub_index,fold, lag_index,:word_num_train[sub_index][fold],:elec_num[sub_index]],w_new_subject),w_new_subject.T)

    # Y11=np.concatenate((Y11,Y2),axis=0)   
    # emb=np.concatenate((emb,emb_train[sub_index,:word_num_train[sub_index][fold],:] ),axis=0) 
    
    Y11_new=Y11-np.mean(Y11,axis=0)

    scaler = StandardScaler()
    emb=scaler.fit_transform(emb)

    reg = LinearRegression().fit(emb,Y11_new)

    # Y_pred=np.matmul(reg.predict(emb_test[sub_index,:word_num_test[sub_index][fold],:]),w_new_subject.T)
    # Y_actual=Y_test[sub_index,fold, lag_index,:word_num_test[sub_index][fold],:elec_num[sub_index]]-np.mean(Y11,axis=0)

    Y_actual=np.matmul(np.matmul(Y_test[sub_index,fold, lag_index,:word_num_test[sub_index][fold],:elec_num[sub_index]],w_new_subject),w_new_subject.T)
    Y_actual=Y_actual-np.mean(Y11,axis=0)
    emb2=emb_test[sub_index,:word_num_test[sub_index][fold],:]
    # emb2=scaler.transform(emb2)
    Y_pred=reg.predict(emb2)
    # Y_actual=np.matmul(Y1,w_new_subject.T)



    a3=[]
    for elec in range(np.shape(Y_pred)[1]):
        
        a3.append(np.corrcoef(Y_pred[:,elec],Y_actual[:,elec])[0,1])

    

    return np.mean(a3)



def doing_srm(lag_index,emb_train,Y_train,fold):

    train_data=[]
    # emb_train=np.zeros((len(subjects),folds, np.max(word_num_train),emb_length)) 
    # Y_train=np.zeros((len(subjects), folds, len(lags),np.max(word_num_train),max(elec_num)))

    for i in range(len(subjects)):
        reg1 = LinearRegression().fit(emb_train[i,:word_num_train[i][fold],:],Y_train[i,fold, lag_index,:word_num_train[i][fold],:elec_num[i]])
        theta1=(reg1.coef_)
        train_data.append(theta1)

    srm = brainiak.funcalign.srm.SRM(n_iter=100, features=srm_features)
    srm.fit(train_data)

    return srm

def doing_srm_with_replace(lag_index,emb_train,Y_train,fold,sub_index):

    train_data=[]
    # emb_train=np.zeros((len(subjects),folds, np.max(word_num_train),emb_length)) 
    # Y_train=np.zeros((len(subjects), folds, len(lags),np.max(word_num_train),max(elec_num)))

    for i in range(len(subjects)):

        if i!=sub_index:
            reg1 = LinearRegression().fit(emb_train[i,:word_num_train[i][fold],:],Y_train[i,fold, lag_index,:word_num_train[i][fold],:elec_num[i]])
            theta1=(reg1.coef_)
            train_data.append(theta1)
        else:
            reg1 = LinearRegression().fit(emb_train[i,:word_num_train[i][fold],:],Y_train[i,fold, lag_index,:word_num_train[i][fold],:elec_num[i]])
            theta1=(reg1.coef_)
            test_data=theta1

    srm = brainiak.funcalign.srm.SRM(n_iter=100, features=srm_features)
    srm.fit(train_data)

    w_new_subject=srm.transform_subject(test_data) 

    return srm, w_new_subject

def doing_srm_with_replace_with_proportion(lag_index,emb_train,Y_train,fold,sub_index,proportion):

    proportion=proportion/100
    train_data=[]
    # emb_train=np.zeros((len(subjects),folds, np.max(word_num_train),emb_length)) 
    # Y_train=np.zeros((len(subjects), folds, len(lags),np.max(word_num_train),max(elec_num)))

    for i in range(len(subjects)):

        if i!=sub_index:
            reg1 = LinearRegression().fit(emb_train[i,:word_num_train[i][fold],:],Y_train[i,fold, lag_index,:word_num_train[i][fold],:elec_num[i]])
            theta1=(reg1.coef_)
            train_data.append(theta1)
        else:
            # Randomly select indices based on the proportion
            num_samples = int(proportion * word_num_train[i][fold])
            random_indices = np.random.choice(word_num_train[i][fold], num_samples, replace=False)

            # Use the randomly selected indices to extract samples
            # emb_train[i, random_indices, :]

            # reg1 = LinearRegression().fit(emb_train[i,:int(proportion*word_num_train[i][fold]),:],Y_train[i,fold, lag_index,:int(proportion*word_num_train[i][fold]),:elec_num[i]])
            reg1 = LinearRegression().fit(emb_train[i, random_indices, :], Y_train[i, fold, lag_index, random_indices, :elec_num[i]])

            
            theta1=(reg1.coef_)
            test_data=theta1

    srm = brainiak.funcalign.srm.SRM(n_iter=100, features=srm_features)
    srm.fit(train_data)

    w_new_subject=srm.transform_subject(test_data) 

    return srm, w_new_subject


def doing_srm_without_replace_with_proportion(lag_index,emb_train,Y_train,fold,sub_index,proportion,args):

    if args.proportion_analysis=='percentage':
        proportion=proportion/100

    train_data=[]
    # emb_train=np.zeros((len(subjects),folds, np.max(word_num_train),emb_length)) 
    # Y_train=np.zeros((len(subjects), folds, len(lags),np.max(word_num_train),max(elec_num)))

    for i in range(len(subjects)):

        if i!=sub_index:
            reg1 = LinearRegression().fit(emb_train[i,:word_num_train[i][fold],:],Y_train[i,fold, lag_index,:word_num_train[i][fold],:elec_num[i]])
            theta1=(reg1.coef_)
            train_data.append(theta1)
        else:
            if args.proportion_analysis=='percentage':
                num_samples = int(proportion * word_num_train[i][fold])
            else:
                if proportion=='all':
                    num_samples=word_num_train[i][fold]
                else:
                    num_samples=proportion

            # Ensure the contiguous block fits within the range
            max_start_index = word_num_train[i][fold] - num_samples
            start_index = np.random.randint(0, max_start_index + 1)  # Random starting index
            random_indices = np.arange(start_index, start_index + num_samples)  # Contiguous indices
            # random_indices = np.random.choice(word_num_train[i][fold], num_samples, replace=False)    
            

            reg1 = LinearRegression().fit(emb_train[i, random_indices, :],Y_train[i,fold, lag_index,random_indices,:elec_num[i]])
            # reg1 = LinearRegression().fit(emb_train[i,:int(proportion*word_num_train[i][fold]),:],Y_train[i,fold, lag_index,:int(proportion*word_num_train[i][fold]),:elec_num[i]])
            theta1=(reg1.coef_)
            train_data.append(theta1)

    srm = brainiak.funcalign.srm.SRM(n_iter=100, features=srm_features)
    srm.fit(train_data)

    return srm, random_indices


original_encoding_all=np.zeros((len(lags), len(subjects),folds))
srm_denoising_all=np.zeros((len(lags),len(subjects),folds))


if original_srm_encoding:

    df = pd.DataFrame()

    for sub in subjects:

        #get elec id
        [_, df1]=get_elec_id(production,sub,args)
        df = pd.concat([df, df1], ignore_index=True)

    new_columns = [f'col_{i}' for i in range(1, folds + 1)]

    # Add the new columns to the existing DataFrame
    df = pd.concat([df, pd.DataFrame(columns=new_columns)], axis=1)

    df_orginal = df.copy()
    df_srm = df.copy()


    print('original and srm encoding')

    for ii in range(len(lags)):
        
        for i in range(len(subjects)):

            original=[]
            srm_denoise=[]

            for fold in range(folds):

                

                # if fold==desired_fold:

                   
                [emb_train,emb_test]=pca_reduction(emb_train_org,emb_test_org,i,fold,word_num_train,word_num_test)

                
                srm=doing_srm(ii,emb_train,Y_train,fold)
                w0 = srm.w_[i] 

                

                # emb_train=np.zeros((len(subjects),folds, np.max(word_num_train),emb_length)) 
                # Y_train=np.zeros((len(subjects), folds, len(lags),np.max(word_num_train),max(elec_num)))


                Y_train_sub=Y_train[i,fold,ii,:word_num_train[i][fold],:elec_num[i]]
                Y_train_sub=Y_train_sub-np.mean(Y_train_sub,axis=0)
                Y_test_sub=Y_test[i,fold,ii,:word_num_test[i][fold],:elec_num[i]]
                Y_test_sub=Y_test_sub-np.mean(Y_train_sub,axis=0)

                Y_train_srm=np.matmul(np.matmul(Y_train_sub,w0),w0.T)
                Y_train_srm=Y_train_srm- np.mean(Y_train_srm,axis=0)
                Y_test_srm=np.matmul(np.matmul(Y_test_sub,w0),w0.T)
                Y_test_srm=Y_test_srm- np.mean(Y_train_srm,axis=0)

                X_train=emb_train[i,:word_num_train[i][fold],:]
                X_test=emb_test[i,:word_num_test[i][fold],:]

                reg3 = LinearRegression().fit(X_train,Y_train_sub)
                reg4 = LinearRegression().fit(X_train,Y_train_srm)

                Y_pred=reg3.predict(X_test)
                Y_pred_srm=reg4.predict(X_test)

                a1=[]
                a2=[]
                for elec in range(np.shape(Y_pred)[1]):
                    
                    a1.append(np.corrcoef(Y_test_sub[:,elec],Y_pred[:,elec])[0,1])
                    a2.append(np.corrcoef(Y_test_srm[:,elec],Y_pred_srm[:,elec])[0,1])

                if word_num_test[i][fold] >400:    
                    df_orginal.loc[(df_orginal['subject'] == subjects[i]), f'col_{fold+1}'] = a1
                    df_srm.loc[(df_srm['subject'] == subjects[i]) , f'col_{fold+1}'] = a2
                    

                # print('Subject:',subjects[i])
                original.append(np.mean(a1))
                srm_denoise.append(np.mean(a2))  #srm_denoised



            original_encoding_all[ii,i,:]=np.asarray(original)
            srm_denoising_all[ii,i,:]=np.asarray(srm_denoise)


    df_orginal.to_csv('/scratch/gpfs/HASSON/arnab/Embedding-SRM/results/csv_files_247/original_encoding_elec_id_'+args.elec_list+'_'+'srm_k_'+str(args.srm_k)+'_lag_'+str(lags)+'_'+status+'.csv', index=False)
    df_srm.to_csv('/scratch/gpfs/HASSON/arnab/Embedding-SRM/results/csv_files_247/srm_denoising_encoding_elec_id_'+args.elec_list+'_'+'srm_k_'+str(args.srm_k)+'_lag_'+str(lags)+'_'+status+'.csv', index=False)



## across subject


across_subject=np.zeros((len(lags),len(subjects),folds,len(subjects)))



if across_subject_encoding:

    print('across_subject_encoding')

    for ii in range(len(lags)):
        
        for i in range(len(subjects)):

            for fold in range(folds):

                # if fold==desired_fold:

               
                [emb_train,emb_test]=pca_reduction(emb_train_org,emb_test_org,i,fold,word_num_train,word_num_test)
                # corr=pca_and_srm(emb_train_org,emb_test_org,i,fold,word_num_train,word_num_test,lag_index, Y_train,Y_test,srm_features)
                
                srm=doing_srm(ii,emb_train,Y_train,fold)
                w0 = srm.w_[i] 

                # emb_train=np.zeros((len(subjects),folds, np.max(word_num_train),emb_length)) 
                # Y_train=np.zeros((len(subjects), folds, len(lags),np.max(word_num_train),max(elec_num)))


                Y_train_sub=Y_train[i,fold,ii,:word_num_train[i][fold],:elec_num[i]]
                Y_train_sub=Y_train_sub-np.mean(Y_train_sub,axis=0)
                Y_test_sub=Y_test[i,fold,ii,:word_num_test[i][fold],:elec_num[i]]
                Y_test_sub=Y_test_sub-np.mean(Y_train_sub,axis=0)

                Y_train_srm=np.matmul(np.matmul(Y_train_sub,w0),w0.T)
                Y_train_srm=Y_train_srm- np.mean(Y_train_srm,axis=0)
                Y_test_srm=np.matmul(np.matmul(Y_test_sub,w0),w0.T)
                Y_test_srm=Y_test_srm- np.mean(Y_train_srm,axis=0)

                X_train=emb_train[i,:word_num_train[i][fold],:]
                X_test=emb_test[i,:word_num_test[i][fold],:]

                reg4 = LinearRegression().fit(X_train,Y_train_srm)

                ww=[]
                for k in range(len(subjects)):

                    w_k= srm.w_[k] 

                    if k !=i:

                        emb=emb_test[k,:word_num_test[k][fold],:]
                        Y=Y_test[k,fold,ii,:word_num_test[k][fold],:elec_num[k]] -np.mean(Y_train[k,fold,ii,:word_num_train[k][fold],:elec_num[k]],axis=0)
                        # Y=Y_test[k,ii,:word_num_test[k][fold],:elec_num[k]]
                        # Y=Y_test[:word_num_test[k],:elec_num[k]]

                        Y_pred_srm_2=reg4.predict(emb)
                        
                        Y_test_srm_from_other_sub=np.matmul(np.matmul(Y,w_k),w0.T)

                        

                        a3=[]
                        for elec in range(np.shape(Y_pred_srm_2)[1]):
                            
                            a3.append(np.corrcoef(Y_test_srm_from_other_sub[:,elec],Y_pred_srm_2[:,elec])[0,1])

                        ww.append(np.mean(a3))  #across subject_no_isc
                        # isc=get_isc(Y_test_srm_from_other_sub, Y_pred_srm_2)
                        # ww_isc.append(isc)

                        # Y_pred=reg3.predict(emb)
                        # b1=get_isc(Y, Y_pred)
                        # ww_isc_no_srm.append(b1)
                    else:
                        ww.append(0)

                across_subject[ii,i,fold,:]=(np.asarray(ww))


## across subject shared space

across_sub_share_space=np.zeros((len(lags),len(subjects),folds))


if across_subject_shared_space:

    print('across_subject_shared_space')

    for ii in range(len(lags)):
        
        for i in range(len(subjects)):

            for fold in range(folds):

                corr=pca_and_srm(emb_train_org,emb_test_org,i,fold,word_num_train,word_num_test,ii, Y_train,Y_test,srm_features)

                print(corr)
                

                across_sub_share_space[ii,i,fold]=corr

def add_data_to_df(df_all_data,p,sub1,a1):
    column_to_add_data=f'col_{p}'
    df_all_data.loc[df_all_data.subject == subjects[sub1], column_to_add_data] = a1
    return df_all_data


across_subject_with_replace=np.zeros((len(lags),len(subjects),folds,len(subjects)))

if across_subject_with_replacing_srm:

    df = pd.DataFrame()

    print('across_subject_encoding')

    for sub in subjects:

        #get elec id
        [_, df1]=get_elec_id(production,sub,args)
        df = pd.concat([df, df1], ignore_index=True)

    new_columns = [f'col_{i}' for i in range(0, 100)]

    # Add the new columns to the existing DataFrame
    df = pd.concat([df, pd.DataFrame(columns=new_columns)], axis=1)

    for ii in range(len(lags)):

        col_index=0
        
        for i in range(len(subjects)):           

            for fold in range(folds):
               
                [emb_train,emb_test]=pca_reduction(emb_train_org,emb_test_org,i,fold,word_num_train,word_num_test)
                # corr=pca_and_srm(emb_train_org,emb_test_org,i,fold,word_num_train,word_num_test,lag_index, Y_train,Y_test,srm_features)

                ww=[]
                
                for k in range(len(subjects)):

                    if k!=i:
                
                        [srm, w_new_subject]=doing_srm_with_replace(ii,emb_train,Y_train,fold,k)

                        if k>i:
                            w0 = srm.w_[i] 
                        else:
                            w0 = srm.w_[i-1] 

                        Y_train_sub=Y_train[i,fold,ii,:word_num_train[i][fold],:elec_num[i]]
                        Y_train_sub=Y_train_sub-np.mean(Y_train_sub,axis=0)
                        Y_test_sub=Y_test[i,fold,ii,:word_num_test[i][fold],:elec_num[i]]
                        Y_test_sub=Y_test_sub-np.mean(Y_train_sub,axis=0)

                        Y_train_srm=np.matmul(np.matmul(Y_train_sub,w0),w0.T)
                        Y_train_srm=Y_train_srm- np.mean(Y_train_srm,axis=0)
                        Y_test_srm=np.matmul(np.matmul(Y_test_sub,w0),w0.T)
                        Y_test_srm=Y_test_srm- np.mean(Y_train_srm,axis=0)

                        X_train=emb_train[i,:word_num_train[i][fold],:]
                        X_test=emb_test[i,:word_num_test[i][fold],:]

                        reg4 = LinearRegression().fit(X_train,Y_train_srm)

              

                        emb=emb_test[k,:word_num_test[k][fold],:]
                        Y=Y_test[k,fold,ii,:word_num_test[k][fold],:elec_num[k]] -np.mean(Y_train[k,fold,ii,:word_num_train[k][fold],:elec_num[k]],axis=0)
                        # Y=Y_test[k,ii,:word_num_test[k][fold],:elec_num[k]]
                        # Y=Y_test[:word_num_test[k],:elec_num[k]]

                        Y_pred_srm_2=reg4.predict(emb)
                        
                        # Y_test_srm_from_other_sub=np.matmul(np.matmul(Y,w_new_subject),w0.T)
                        Y_pred_srm_2=np.matmul(np.matmul(Y_pred_srm_2,w0),w_new_subject.T)
                        Y_test_srm_from_other_sub=np.matmul(np.matmul(Y,w_new_subject),w_new_subject.T)

                        

                        a3=[]
                        for elec in range(np.shape(Y_pred_srm_2)[1]):
                            
                            a3.append(np.corrcoef(Y_test_srm_from_other_sub[:,elec],Y_pred_srm_2[:,elec])[0,1])

                        if word_num_test[k][fold] >400:
                            df = add_data_to_df(df, col_index, k, a3)

                        ww.append(np.mean(a3))  

                    else:
                        ww.append(0)

                col_index=col_index+1

                across_subject_with_replace[ii,i,fold,:]=(np.asarray(ww))
                

    df.to_csv('/scratch/gpfs/HASSON/arnab/Embedding-SRM/results/csv_files_247/across_subject_with_replacing_srm_elec_id_'+args.elec_list+'_'+'srm_k_'+str(args.srm_k)+'_lag_'+str(lags)+'_'+status+'.csv', index=False) 


proportions=np.arange(10,110,10)

across_subject_with_proportion=np.zeros((len(proportions),len(lags),len(subjects),folds,len(subjects)))


if across_subject_with_various_train_data_proportion:

    print('across_subject_with_various_train_data_proportion')

    
    for kk in range(len(proportions)):
    
        for ii in range(len(lags)):
        
            for i in range(len(subjects)):

                for fold in range(folds):
               
                    [emb_train,emb_test]=pca_reduction(emb_train_org,emb_test_org,i,fold,word_num_train,word_num_test)
                    # corr=pca_and_srm(emb_train_org,emb_test_org,i,fold,word_num_train,word_num_test,lag_index, Y_train,Y_test,srm_features)

                    ww=[]
                    for k in range(len(subjects)):

                        if k!=i:
                    
                            [srm, w_new_subject]=doing_srm_with_replace_with_proportion(ii,emb_train,Y_train,fold,k,proportions[kk])

                            if k>i:
                                w0 = srm.w_[i] 
                            else:
                                w0 = srm.w_[i-1] 

                            Y_train_sub=Y_train[i,fold,ii,:word_num_train[i][fold],:elec_num[i]]
                            Y_train_sub=Y_train_sub-np.mean(Y_train_sub,axis=0)
                            Y_test_sub=Y_test[i,fold,ii,:word_num_test[i][fold],:elec_num[i]]
                            Y_test_sub=Y_test_sub-np.mean(Y_train_sub,axis=0)

                            Y_train_srm=np.matmul(np.matmul(Y_train_sub,w0),w0.T)
                            Y_train_srm=Y_train_srm- np.mean(Y_train_srm,axis=0)
                            Y_test_srm=np.matmul(np.matmul(Y_test_sub,w0),w0.T)
                            Y_test_srm=Y_test_srm- np.mean(Y_train_srm,axis=0)

                            X_train=emb_train[i,:word_num_train[i][fold],:]
                            X_test=emb_test[i,:word_num_test[i][fold],:]

                            reg4 = LinearRegression().fit(X_train,Y_train_srm)


                            emb=emb_test[k,:word_num_test[k][fold],:]
                            Y=Y_test[k,fold,ii,:word_num_test[k][fold],:elec_num[k]] -np.mean(Y_train[k,fold,ii,:word_num_train[k][fold],:elec_num[k]],axis=0)
                            # Y=Y_test[k,ii,:word_num_test[k][fold],:elec_num[k]]
                            # Y=Y_test[:word_num_test[k],:elec_num[k]]

                            Y_pred_srm_2=reg4.predict(emb)
                            
                            Y_test_srm_from_other_sub=np.matmul(np.matmul(Y,w_new_subject),w0.T)

                            

                            a3=[]
                            for elec in range(np.shape(Y_pred_srm_2)[1]):
                                
                                a3.append(np.corrcoef(Y_test_srm_from_other_sub[:,elec],Y_pred_srm_2[:,elec])[0,1])

                            ww.append(np.mean(a3))  
                        else:
                            ww.append(0)

                    across_subject_with_proportion[kk,ii,i,fold,:]=(np.asarray(ww))

proportions=[]

if args.proportion_analysis=='percentage':
    proportions=np.concatenate((np.arange(1,10), np.arange(10,110,10)))
elif args.proportion_analysis=='fixed_number':
    proportions=[100,500,1000,3000, 5000, 7000, 10000,'all']

org_encoding_single_subject_data_for_diff_proportion=np.zeros((len(proportions),len(lags),len(subjects),folds))
srm_denoise_single_subject_data_for_diff_proportion=np.zeros((len(proportions),len(lags),len(subjects),folds))
srm_denoise_with_other_subjects_data_for_diff_proportion=np.zeros((len(proportions),len(lags),len(subjects),folds))


if encoding_with_all_subjects_data:

    print('encoding_with_other_subjects_data')

    
    for kk in range(len(proportions)):
    
        for ii in range(len(lags)):
        
            for i in range(len(subjects)):

                ww1=[]
                ww2=[]
                ww3=[]

                for fold in range(folds):
               
                    [emb_train,emb_test]=pca_reduction(emb_train_org,emb_test_org,i,fold,word_num_train,word_num_test)
                    
                    
                    
                    srm, random_indices = doing_srm_without_replace_with_proportion(ii,emb_train,Y_train,fold,i,proportions[kk],args)

                    w0 = srm.w_[i] 

                    Y_train_sub=Y_train[i,fold,ii,random_indices,:elec_num[i]]
                    Y_train_sub=Y_train_sub-np.mean(Y_train_sub,axis=0)
                    Y_test_sub=Y_test[i,fold,ii,:word_num_test[i][fold],:elec_num[i]]
                    Y_test_sub=Y_test_sub-np.mean(Y_train_sub,axis=0)

                    Y_train_srm=np.matmul(np.matmul(Y_train_sub,w0),w0.T)
                    Y_train_srm=Y_train_srm- np.mean(Y_train_srm,axis=0)
                    Y_test_srm=np.matmul(np.matmul(Y_test_sub,w0),w0.T)
                    Y_test_srm=Y_test_srm- np.mean(Y_train_srm,axis=0)

                    # X_train=emb_train[i,:int(proportions[kk]*word_num_train[i][fold]/100),:]
                    X_train=emb_train[i,random_indices,:]
                    X_test=emb_test[i,:word_num_test[i][fold],:]

                    reg1 = LinearRegression().fit(X_train,Y_train_sub)
                    Y_pred_org=reg1.predict(X_test)

                    reg2 = LinearRegression().fit(X_train,Y_train_srm)
                    Y_pred_srm_single_sub=reg2.predict(X_test)
                    

                    for k in range(len(subjects)):

                        if k!=i:
                    
                            w1= srm.w_[k] 

                            Y1=Y_train[k,fold,ii,:word_num_train[k][fold],:elec_num[k]]
                            Y1=Y1-np.mean(Y1,axis=0)

                            Y1_srm=np.matmul(np.matmul(Y1,w1),w0.T)
                            Y1_srm=Y1_srm-np.mean(Y1_srm,axis=0)

                            X1=emb_train[k,:word_num_train[k][fold],:]

                            Y_train_srm=np.concatenate((Y_train_srm,Y1_srm),axis=0)
                            X_train=np.concatenate((X_train,X1),axis=0)

                    Y_train_srm=Y_train_srm- np.mean(Y_train_srm,axis=0)
                    Y_test_srm2=Y_test_srm- np.mean(Y_train_srm,axis=0)
                    
                    reg3 = LinearRegression().fit(X_train,Y_train_srm)
                    Y_pred_srm_with_other_sub=reg3.predict(X_test)

                    a1=[]
                    a2=[]    
                    a3=[]

                    for elec in range(np.shape(Y_pred_org)[1]):
                        
                        a1.append(np.corrcoef(Y_pred_org[:,elec],Y_test_sub[:,elec])[0,1])
                        a2.append(np.corrcoef(Y_pred_srm_single_sub[:,elec],Y_test_srm[:,elec])[0,1])
                        a3.append(np.corrcoef(Y_pred_srm_with_other_sub[:,elec],Y_test_srm2[:,elec])[0,1])

                    ww1.append(np.mean(a1))  
                    ww2.append(np.mean(a2))  
                    ww3.append(np.mean(a3))  
                        
                org_encoding_single_subject_data_for_diff_proportion[kk,ii,i,:]=(np.asarray(ww1))
                srm_denoise_single_subject_data_for_diff_proportion[kk,ii,i,:]=(np.asarray(ww2))
                srm_denoise_with_other_subjects_data_for_diff_proportion[kk,ii,i,:]=(np.asarray(ww3))


def create_df_for_brainplot(df):

    
    df = df.dropna(subset=['MNI_X'])
    
    # Identify columns that start with 'col_'
    col_columns = [col for col in df.columns if col.startswith('col_')]
    
    # Calculate the average of non-empty 'col_' columns for each row
    df['corr_value'] = df[col_columns].apply(lambda row: row.dropna().mean(), axis=1)

    return df

# def make_brainplot(path,df,criterion,vmin=0,vmax=0.25,cmap='inferno_r'):
def make_brainplot(df,criterion,vmin=0,vmax=0.25,cmap='inferno_r'):

    path='/scratch/gpfs/HASSON/arnab/model_based_srm/result/247/figures'
    values = df.corr_value.to_numpy()

    coords= df[['MNI_X', 'MNI_Y', 'MNI_Z']].to_numpy()
    
    # coords *= 1000  # nilearn likes to plot in meters, not mm
    # print("Coordinate matrix shape: ", coords.shape)
    
    order = values.argsort()
    plot_markers(values[order], coords[order],
                 node_size=20, display_mode='l',
                 node_vmin=vmin,node_vmax=vmax, node_cmap=cmap, colorbar=True)
    filename=path+'/'+'brainplot_'+criterion+'_'+'lags_'+str(lags)+'_'+status+'.svg'
    
    plt.savefig(filename,dpi=600)
    plt.close()
    # plt.show()

def make_bar_plot(df_org, df_srm, df_across, args):


    subjects=df_org.subject.unique()
    
    original = df_org.groupby('subject', sort=False)['corr_value'].mean().reindex(subjects).tolist()
    srm_denoised=df_srm.groupby('subject', sort=False)['corr_value'].mean().reindex(subjects).tolist()
    csrm_across=df_across.groupby('subject', sort=False)['corr_value'].mean().reindex(subjects).tolist()
    # csrm_across_with_3_sub_train=list(np.mean(across_shared_space[0,:,:],axis=1))
    
    
    error_original=df_org.groupby('subject', sort=False)['corr_value'].mean().reindex(subjects).sem().tolist()
    error_srm_denoised=df_srm.groupby('subject', sort=False)['corr_value'].mean().reindex(subjects).sem().tolist()
    error_across_subject=df_srm.groupby('subject', sort=False)['corr_value'].mean().reindex(subjects).sem().tolist()
    # error_csrm_across_with_3_sub_train=list(stats.sem(across_shared_space[0,:,:],axis=1))
        
    # set width of bar 
    barWidth = 0.20
    
    # Set position of bar on X axis 
    br1 = np.arange(len(original)) 
    br2 = [x + barWidth for x in br1] 
    br3 = [x + barWidth for x in br2] 
    # br4 = [x + barWidth for x in br3] 
    
    colors=["#fee8c8","#fdbb84","#e34a33"]
    
    # Make the plot
    plt.bar(br1, original, color =colors[0], width = barWidth, 
            edgecolor ='grey', label ='Original') 
    plt.bar(br2, srm_denoised, color =colors[1], width = barWidth, 
            edgecolor ='grey', label ='SRM-denoised') 
    plt.bar(br3, csrm_across, color =colors[2], width = barWidth, 
            edgecolor ='grey', label ='Across Subject') 
    # plt.bar(br4, csrm_across_with_3_sub_train, color ='darkorange', width = barWidth, 
            # edgecolor ='grey', label ='E_SRM_across (3 sub train)') 
    
    
    
    plt.errorbar(br1, original, yerr=error_original,  fmt = '.', color="k")
    plt.errorbar(br2, srm_denoised, yerr=error_srm_denoised, fmt=".", color="k")
    plt.errorbar(br3, csrm_across, yerr=error_across_subject, fmt=".", color="k")
    # plt.errorbar(br4, csrm_across_with_3_sub_train, yerr=error_csrm_across_with_3_sub_train, fmt=".", color="k")
    
    path='/scratch/gpfs/HASSON/arnab/Embedding-SRM/results/figures_247/'
    
    filename=path+'bar_plot_srm_k_'+str(args.srm_k)+'_lag_'+str(args.lags)+'_'+status+'.svg'
    
    subs=[]
    subs.append(['S'+str(k) for k in range(1 ,len(subjects)+1)])

    # Adding Xticks 
    plt.xlabel('Subjects', fontweight ='bold', fontsize = 15) 
    plt.ylabel('Encoding Performance (r)', fontweight ='bold', fontsize = 15) 
    plt.xticks([r + 1*barWidth for r in range(len(original))], 
            subs[0], fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim(0, 0.25)
    # plt.title(status) 
    plt.legend(frameon=False, fontsize=10)
    # plt.legend(bbox_to_anchor=(1.00, 0.95),prop={'size': 8})
    plt.tight_layout()
    plt.savefig(filename,dpi=600)
    plt.close()
    # plt.show() 


def make_area_bar_plot(df_org, df_srm, df_across, args, areas=None):

    df=pd.DataFrame()
    df['subject']=df_org['subject']
    df['electrode_id']=df_org['electrode_id']
    df['electrode_name']=df_org['electrode_name']
    df['org_corr']=df_org['corr_value']
    df['srm_corr']=df_srm['corr_value']
    df['across_corr']=df_across['corr_value']
    df['area']=df_org['reg1']
    
    df = df.dropna(subset=['area'])
    
    
    if areas is None: 
        areas= df['area'].dropna().unique()
    
    original=[]
    srm_denoised=[]
    across_subject=[]

    error_original=[]
    error_srm_denoised=[]
    error_across_subject=[]
    
    
    for area in areas:
        
        a=df.loc[df.area==area]
        original.append(np.mean(a.org_corr.values))
        srm_denoised.append(np.mean(a.srm_corr.values))
        across_subject.append(np.mean(a.across_corr.values))
    
        error_original.append(stats.sem(a.org_corr.values))
        error_srm_denoised.append(stats.sem(a.srm_corr.values))
        error_across_subject.append(stats.sem(a.across_corr.values))
    
    
    # Increase figure size for better readability
    plt.figure(figsize=(14, 6))
    
    
    # set width of bar 
    barWidth = 0.25
    
    # Set position of bar on X axis 
    br1 = np.arange(len(original)) 
    br2 = [x + barWidth for x in br1] 
    br3 = [x + barWidth for x in br2] 

    colors=["#fee8c8","#fddd84","#e34a33"]
    
    # Make the plot
    plt.bar(br1, original, color =colors[0], width = barWidth, 
            edgecolor ='grey', label ='Original') 
    plt.bar(br2, srm_denoised, color =colors[1], width = barWidth, 
            edgecolor ='grey', label ='SRM-denoised') 
    plt.bar(br3, across_subject, color =colors[2], width = barWidth, 
            edgecolor ='grey', label ='Across Subject') 
    
    plt.errorbar(br1, original, yerr=error_original,  fmt = '.', color="k")
    plt.errorbar(br2, srm_denoised, yerr=error_srm_denoised, fmt=".", color="k")
    plt.errorbar(br3, across_subject, yerr=error_across_subject, fmt=".", color="k")
    
    # Adding Xticks 
    plt.xlabel('Brain Areas', fontweight ='bold', fontsize = 15) 
    plt.ylabel('Encoding Performance (r)', fontweight ='bold', fontsize = 15) 
    plt.xticks([r + barWidth for r in range(len(original))], 
            areas, rotation=45, ha='right', fontsize=12)
     
    plt.legend(fontsize=12,frameon=False)
    plt.tight_layout()
    # plt.savefig('/scratch/gpfs/arnab/Encoding/result/sam_work/csrm_all_elec.png',dpi=600)
    path='/scratch/gpfs/HASSON/arnab/Embedding-SRM/results/figures_247/'
    filename=path+'area_bar_plot_srm_k_'+str(args.srm_k)+'_lag_'+str(args.lags)+'_'+status+'.svg'
    plt.savefig(filename,dpi=600)
    plt.close()
    # plt.show()  


if args.plotting:

    path='/scratch/gpfs/HASSON/arnab/model_based_srm/result/247/csv_files/'
    
    org_file=path+'original_encoding_elec_id_'+args.elec_list+'_'+'srm_k_'+str(args.srm_k)+'_fold_'+str(folds)+'_lag_'+str(lags)+'_'+status+'.csv'
    srm_file=path+'srm_denoising_encoding_elec_id_'+args.elec_list+'_'+'srm_k_'+str(args.srm_k)+'_fold_'+str(folds)+'_lag_'+str(lags)+'_'+status+'.csv'
    across_file=path+'across_subject_with_replacing_srm_elec_id_'+args.elec_list+'_'+'srm_k_'+str(args.srm_k)+'_fold_'+str(folds)+'_lag_'+str(lags)+'_'+status+'.csv'

    df_org=pd.read_csv(org_file)
    df_srm=pd.read_csv(srm_file)
    df_across=pd.read_csv(across_file)


    df_org=create_df_for_brainplot(df_org)
    df_srm=create_df_for_brainplot(df_srm)
    df_across=create_df_for_brainplot(df_across)                

    make_brainplot(df_org,'original_encoding', vmin=0,vmax=0.3,cmap='inferno_r')
    make_brainplot(df_srm,'srm_denoising_encoding', vmin=0,vmax=0.3,cmap='inferno_r')
    make_brainplot(df_across,'across_subject_encoding', vmin=0,vmax=0.3,cmap='inferno_r')

    df=pd.DataFrame()
    df['MNI_X']=df_org['MNI_X']
    df['MNI_Y']=df_org['MNI_Y']
    df['MNI_Z']=df_org['MNI_Z']
    df['corr_value']=df_srm['corr_value']-df_org['corr_value']
    make_brainplot(df,'difference_encoding', vmin=-0.1,vmax=0.1,cmap='RdBu_r')

    make_bar_plot(df_org, df_srm, df_across, args)
    if args.production==1:
        areas=['STG','mMTG','rMTG','cMTG','TP','AG','IFG','MFG','M','SM']
    else:
        areas=['STG','mMTG','cMTG','TP','AG','IFG','MFG']
    make_area_bar_plot(df_org, df_srm, df_across, args, areas=areas)


if args.mat_file_saving:

    path='/scratch/gpfs/HASSON/arnab/Embedding-SRM/results/mat_files_247/'+filename_mat
    

    savemat(path,{'corr_original':original_encoding_all,'lags':lags,'subjects':subjects,'status':status, 'srm_denoise':srm_denoising_all,'across_subject':across_subject,'across_sub_share_space':across_sub_share_space,'across_subject_with_replace':across_subject_with_replace,'across_subject_with_proportion':across_subject_with_proportion,
                        'org_encoding_single_subject_data_for_diff_proportion':org_encoding_single_subject_data_for_diff_proportion,
                        'srm_denoise_single_subject_data_for_diff_proportion':srm_denoise_single_subject_data_for_diff_proportion,
                        'srm_denoise_with_other_subjects_data_for_diff_proportion':srm_denoise_with_other_subjects_data_for_diff_proportion,
                        'proportions':proportions
                        })

    # file_elec=path+'all_elec_fold_'+str(folds)+'_lag_'+str(lags)+'.mat'
    # savemat(file_elec,{'all_elec_org_625':all_elec_org_625,'all_elec_org_676':all_elec_org_676,'all_elec_org_7170':all_elec_org_7170,'all_elec_org_798':all_elec_org_798,'all_elec_srm_625':all_elec_srm_625,'all_elec_srm_676':all_elec_srm_676,'all_elec_srm_7170':all_elec_srm_7170,'all_elec_srm_798':all_elec_srm_798,})

            

    

        


        
    
    
            
        
    