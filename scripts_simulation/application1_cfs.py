import calour as cl
import numpy as np

from scipy.stats import sem
import pickle
cl.set_log_level(40) 


# input biom table and mapping file
cfs = cl.read_amplicon('../data/cfs.biom','../data/cfs.map.txt', sparse=False, normalize=10000, min_reads=1000)

filtlev = [0,1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100,200,300,400,500]
B = 1000

sig_ds_cfs = []
sig_bh_cfs = []
sig_fbh_cfs = []

err_bh_cfs = []
err_ds_cfs = []
err_fbh_cfs = []
for i in filtlev:
    print('filter level...: %s' %(i))
    
    sig_ds = []
    sig_bh = []
    sig_fbh = []
    for j in range(B):
        # filter away low abundant taxa
        cfs_sub = cfs.filter_min_abundance(i)

        # apply FDR methods
        ds_cfs = cfs_sub.diff_abundance('Subject','Control','Patient', fdr_method='dsfdr')
        bh_cfs = cfs_sub.diff_abundance('Subject','Control','Patient', fdr_method='bhfdr')
        fbh_cfs = cfs_sub.diff_abundance('Subject','Control','Patient', fdr_method='filterBH')
        
        sig_ds.append(len(ds_cfs.feature_metadata.index))
        sig_bh.append(len(bh_cfs.feature_metadata.index))
        sig_fbh.append(len(fbh_cfs.feature_metadata.index))

    sig_ds_cfs.append(np.mean(sig_ds))
    sig_bh_cfs.append(np.mean(sig_bh))
    sig_fbh_cfs.append(np.mean(sig_fbh))
    err_ds_cfs.append(sem(sig_ds))
    err_bh_cfs.append(sem(sig_bh))
    err_fbh_cfs.append(sem(sig_fbh))

with open("../results_all/cfs_filtering_updated_B1k.pkl", "wb") as f:
    pickle.dump((filtlev, sig_bh_cfs, sig_ds_cfs, sig_fbh_cfs, err_ds_cfs, err_bh_cfs, err_fbh_cfs), f)
