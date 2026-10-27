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

def parse_arguments():
    """Read commandline arguments

    Returns:
        args (Namespace): input as well as default arguments
    """

    parser = argparse.ArgumentParser()
    
    # main args
    parser.add_argument("--production", type=int, required=True)
    parser.add_argument("--subject",  nargs="+", type=int, required=True) 
    parser.add_argument("--lags",  nargs="+", type=int, required=True)
    parser.add_argument("--folds", type=int, required=True) 
    parser.add_argument("--elec_list", type=str, required=True)
    # parser.add_argument("--srm_k", type=int, required=True)
    parser.add_argument("--window_length", type=int, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    

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
model_name=args.model_name
lags=args.lags
min_num_word=0
production=args.production


if production==1:
    status='prod'
else:
    status='comp'

# Set the parameter
plt.rcParams["svg.fonttype"] = "none"

print('Status:',status)
print(args)

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
def base_df_load(subject,args):

    path_base_df='/scratch/gpfs/HASSON/kw1166/247/247-pickling/results/tfs/'  #625
    path_base_df=path_base_df+str(subject)+'/pickles/embeddings/'+args.model_name+'/full/base_df.pkl'
    
    df = pd.read_pickle(path_base_df)

    if args.model_name=='glove50':
        df=df.loc[(~df.adjusted_onset.isnull())]
    else:

        df=df.loc[(~df.adjusted_onset.isnull()) & (df.token_idx==0) & (df.is_nonword==False)]

    # df=df[df.num_words>min_num_word]
    
    
    return df



def get_embeddings_df(df_emb, df, conv_id, production):
        

    index=df[(df.conversation_id==conv_id) & (df.production==production)].index[1:]
    
    df_bert=df_emb.iloc[index]
    
    # onsets=np.squeeze(df[df.conversation_id==conv_id].adjusted_onset.values[1:])
    
    onsets=np.squeeze(df[(df.conversation_id==conv_id) & (df.production==production)].onset.values[1:])
    offsets=np.squeeze(df[(df.conversation_id==conv_id) & (df.production==production)].offset.values[1:])
    words=np.squeeze(df[(df.conversation_id==conv_id) & (df.production==production)].word.values[1:])
    sentence=np.squeeze(df[(df.conversation_id==conv_id) & (df.production==production)].sentence.values[1:])
    speaker=np.squeeze(df[(df.conversation_id==conv_id) & (df.production==production)].speaker.values[1:])
    num_word=np.squeeze(df[(df.conversation_id==conv_id) & (df.production==production)].num_words.values[1:])
    conv_name=np.squeeze(df[(df.conversation_id==conv_id) & (df.production==production)].conversation_name.values[1:])
    
    
    w=(df_bert.embeddings.values)
    
    df_final=pd.DataFrame()
    df_final['onsets']=onsets
    df_final['offsets']=offsets
    df_final['word']=words
    df_final['sentence']=sentence
    df_final['speaker']=speaker
    df_final['num_word']=num_word
    df_final['conversation_id']=conv_id
    df_final['conversation_name']=conv_name
    df_final['embeddings']=w
    
    return df_final


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

def get_elec_data(subject, df, conv_id, onsets, lags, elec_id, args):

    # print('Getting Electrode Data')

    path='/projects/HASSON/247/data/conversations-car/'+str(subject)+'/'
    conv_name=df[df.conversation_id==conv_id].conversation_name.values[0]
    
    elec_num=len(elec_id)
    
    Y_data= np.zeros((len(lags),len(onsets),elec_num)) 

    
    ecogs=[]
    for k1 in elec_id:
        filename=path+'/'+conv_name+'/preprocessed/'+conv_name+'_electrode_preprocess_file_'+str(k1)+'.mat'

        if subject==798:
            filename=path+'/'+conv_name+'/preprocessed_allElec/'+conv_name+'_electrode_preprocess_file_'+str(k1)+'.mat'
        
        if subject==7170:

            if k1==10:
                filename=path+'/'+conv_name+'/preprocessed_allElec/'+conv_name+'_electrode_preprocess_file_'+str(k1)+'.mat'
            else:
                filename=path+'/'+conv_name+'/preprocessed_v2/'+conv_name+'_electrode_preprocess_file_'+str(k1)+'.mat'

        e=loadmat(filename)['p1st'].squeeze().astype(np.float32)
        ecogs.append(e)
    
    ecogs = np.asarray(ecogs).T

    ecogs= stats.zscore(ecogs, axis=0)   ## z-score
    
    window_size=args.window_length
    
    half_window = round((window_size / 1000) * 512 / 2)
    #t = len(brain_signal)
    t=len(ecogs[:,0])
    
    Y= np.zeros((len(onsets), np.shape(ecogs)[1]))
    
        
    for ii in range(len(lags)):
        Y1 = np.zeros((len(onsets), 2 * half_window + 1))
        #print(ii)
        for k in range(np.shape(ecogs)[1]):
    
            brain_signal=ecogs[:,k]
                    
            lag_amount = int(lags[ii]/ 1000 * 512)
            
            index_onsets = np.minimum(
                t - half_window - 1,
                np.maximum(half_window + 1,
                            np.round(onsets, 0, onsets) + lag_amount))
            
            # subtracting 1 from starts to account for 0-indexing
            starts = index_onsets - half_window - 1
            stops = index_onsets + half_window
            
            for i, (start, stop) in enumerate(zip(starts, stops)):
                start=int(start)
                stop=int(stop)
                Y1[i, :] = brain_signal[start:stop].reshape(-1)
                
                                
            #if subject==717:
               
            Y_data[ii,:,k] = np.mean(Y1, axis=-1)

    return Y_data

## data create
def data_create(args):

    print('Creating Data')

    for sub in args.subject:
        for fold in range(args.folds):
            globals()[f"df_train_{sub}_fold_{fold}"] = pd.DataFrame()    
            globals()[f"df_test_{sub}_fold_{fold}"] = pd.DataFrame()    

    for kk in range(len(args.subject)):

        sub=args.subject[kk]
        print(sub)

        base_df=base_df_load(sub,args)

        #df_emb
        if args.model_name=='glove50':

            path_emb='/scratch/gpfs/HASSON/kw1166/247/247-pickling/results/tfs/'+str(sub)+'/pickles/embeddings/glove50/full/cnxt_0001/layer_01.pkl'
        else:
            path_emb1='/scratch/gpfs/HASSON/kw1166/247/247-pickling/results/tfs/' 
            path_emb1=path_emb1+str(sub)+'/pickles/embeddings/'+args.model_name+'/full/cnxt_1024/'
            emb_file='layer_48.pkl'    
            path_emb=path_emb1+emb_file

        # df_emb=load_label(path_emb)
        df_emb = pd.DataFrame(pd.read_pickle(path_emb))

        if args.model_name=='glove50' and sub==798:
            base_df = base_df.reset_index()

        [elec_id, df]=get_elec_id(production,sub,args)

        

        # conv_ids=list(range(1,total_conv_numbers[kk]+1,1))

        # conv_ids=list(np.unique(base_df.conversation_id.values))
        conv_ids = np.unique(base_df.conversation_id.values)   # keep as an array

        # train_conv_ids=list(range(1,q+1,1))
        # test_conv_ids=list(range(q+1,total_conv_numbers[kk]+1,1))

        kf = KFold(n_splits=args.folds)
        fold=-1

        # for train_conv_ids, test_conv_ids in kf.split(conv_ids):
        for train_idx, test_idx in kf.split(conv_ids):
            train_conv_ids = conv_ids[train_idx]
            test_conv_ids = conv_ids[test_idx]

            fold=fold+1
            print('fold:',fold)

            p=0              

            for conv_id in train_conv_ids:

                # print(conv_id)

                df_final=get_embeddings_df(df_emb, base_df, conv_id, production)

                # df_final=get_embeddings_df(sub, model_name, base_df, conv_id, production)
            
                onsets=df_final.onsets.values
                # onsets=[0]

                # print(len(onsets))

                if (len(onsets))>0:

                    Y1=get_elec_data(sub, base_df, conv_id, onsets, lags, elec_id, args)

                    if p==0:

                        globals()[f"Y_train_{sub}_fold_{fold}"]=Y1
                        globals()[f"df_train_{sub}_fold_{fold}"]=df_final

                    else:

                        globals()[f"Y_train_{sub}_fold_{fold}"]=np.concatenate((globals()[f"Y_train_{sub}_fold_{fold}"],Y1), axis=1)
                        globals()[f"df_train_{sub}_fold_{fold}"]=pd.concat([globals()[f"df_train_{sub}_fold_{fold}"], df_final])

                    p=p+1
                    
                    del df_final
                    del onsets
                    del Y1
                
                else:
                    
                    del df_final

                

            p=0
            q=0
            for conv_id in test_conv_ids:

                

                df_final=get_embeddings_df(df_emb, base_df, conv_id, production)

                # df_final=get_embeddings_df(sub, model_name, base_df, conv_id, production)
                onsets=df_final.onsets.values

                # print(len(onsets))

                if (len(onsets))>0:
                    q=1

                    Y1=get_elec_data(sub, base_df, conv_id, onsets, lags, elec_id, args)

                    if p==0:

                        globals()[f"Y_test_{sub}_fold_{fold}"]=Y1
                        globals()[f"df_test_{sub}_fold_{fold}"]=df_final

                    else:

                        globals()[f"Y_test_{sub}_fold_{fold}"]=np.concatenate((globals()[f"Y_test_{sub}_fold_{fold}"],Y1), axis=1)
                        globals()[f"df_test_{sub}_fold_{fold}"]=pd.concat([globals()[f"df_test_{sub}_fold_{fold}"], df_final])

                    p=p+1
                    
                    del df_final
                    del onsets
                    del Y1

                else:
                    del df_final

            if q==0:
                globals()[f"Y_test_{sub}_fold_{fold}"]=globals()[f"Y_test_{sub}_fold_{fold-1}"]
                globals()[f"df_test_{sub}_fold_{fold}"]=globals()[f"df_test_{sub}_fold_{fold-1}"]

                globals()[f"Y_train_{sub}_fold_{fold}"]=globals()[f"Y_train_{sub}_fold_{fold-1}"]
                globals()[f"df_train_{sub}_fold_{fold}"]=globals()[f"df_train_{sub}_fold_{fold-1}"]

    df_words=pd.DataFrame()
    # for sub in args.subject:
    #     for fold in range(args.folds):
    #         df_words[f'words_{sub}_fold_{fold}_train']=globals()[f"df_train_{sub}_fold_{fold}"].word.values
    #         df_words[f'words_{sub}_fold_{fold}_test']=globals()[f"df_test_{sub}_fold_{fold}"].word.values

    # build df_words robustly (handles varying column lengths)
    cols = {}
    for sub in args.subject:
        for fold in range(args.folds):
            cols[f'words_{sub}_fold_{fold}_train'] = pd.Series(
                globals()[f"df_train_{sub}_fold_{fold}"].word.values
            )
            cols[f'words_{sub}_fold_{fold}_test'] = pd.Series(
                globals()[f"df_test_{sub}_fold_{fold}"].word.values
            )
    df_words = pd.DataFrame(cols)
            
    word_num_train=[]
    word_num_test=[]
    elec_num=[]

    for sub in args.subject:
        elec_num.append(np.shape(globals()[f"Y_train_{sub}_fold_0"])[2])
        aa1=[]
        aa2=[]
        for fold in range(args.folds):
            aa1.append(np.shape(globals()[f"Y_train_{sub}_fold_{fold}"])[1])
            aa2.append(np.shape(globals()[f"Y_test_{sub}_fold_{fold}"])[1])
            
        word_num_train.append(np.asarray(aa1))
        word_num_test.append(np.asarray(aa2))

    Y_train=np.zeros((len(args.subject), args.folds, len(lags),np.max(word_num_train),max(elec_num)))
    Y_test=np.zeros((len(args.subject), args.folds, len(lags),np.max(word_num_test),max(elec_num)))

    emb_length=len(globals()[f"df_train_{args.subject[0]}_fold_0"].iloc[0].embeddings)

    emb_train=np.zeros((len(args.subject),args.folds, np.max(word_num_train),emb_length))                  
    emb_test=np.zeros((len(args.subject), args.folds, np.max(word_num_test),emb_length))   

    

    for k in range(len(lags)):

        for i in range(len(args.subject)):

            for fold in range(args.folds):

                Y_train[i,fold,k,:word_num_train[i][fold],:elec_num[i]]=globals()[f"Y_train_{args.subject[i]}_fold_{fold}"][k,:,:]
                Y_test[i,fold,k,:word_num_test[i][fold],:elec_num[i]]=globals()[f"Y_test_{args.subject[i]}_fold_{fold}"][k,:,:]
                
                e1=(globals()[f"df_train_{args.subject[i]}_fold_{fold}"].embeddings.values)
                e2=(globals()[f"df_test_{args.subject[i]}_fold_{fold}"].embeddings.values)
                    
                for ii in range(len(e1)):
                    emb_train[i,fold,ii,:]=e1[ii]

                for ii in range(len(e2)):
                    emb_test[i,fold, ii,:]=e2[ii]


    return Y_train, Y_test, emb_train, emb_test,word_num_train,word_num_test, elec_num, df_words


[Y_train, Y_test, emb_train_org, emb_test_org,word_num_train,word_num_test, elec_num, df_words]=data_create(args)

path='/scratch/gpfs/HASSON/arnab/Embedding-SRM/results/pickle_files/'
filename_pkl=path+'data_zscore_717_out_with_words_new_elec_id_'+args.model_name+'_lag_'+str(lags)+'_'+status+'.pkl'

db = {}
db['elec_train'] =Y_train
db['elec_test'] =Y_test
db['emb_train']=emb_train_org
db['emb_test']=emb_test_org
db['word_num_train']=word_num_train
db['word_num_test']=word_num_test
db['elec_num']=elec_num
db['df_words']=df_words

with open(filename_pkl, "wb") as f:
    pickle.dump(db, f)
