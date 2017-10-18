import calour as cl
import numpy as np

from scipy.stats import sem
import pickle
cl.set_log_level(40) 

# input biom table and mapping file
mlt = cl.read_amplicon('../data/mlt.biom','../data/mlt.map.txt', sparse=False, normalize=10000, min_reads=1000)

filtlev = [0,1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100,200,300,400,500]
B = 1000

sig_ds_mlt = []
sig_bh_mlt = []
sig_fbh_mlt = []

err_bh_mlt = []
err_ds_mlt = []
err_fbh_mlt = []
for i in filtlev:
    print('filter level...: %s' %(i))

    sig_ds = []
    sig_bh = []
    sig_fbh = []
    for j in range(B):
        # filter away low abundant taxa
        mlt_sub = mlt.filter_min_abundance(i)

        # apply FDR methods
        ds_mlt = mlt_sub.diff_abundance('Description','mouse cecum, TLR5 knockout',
                                    'mouse cecum, wild type', fdr_method='dsfdr')
        bh_mlt = mlt_sub.diff_abundance('Description','mouse cecum, TLR5 knockout',
                                    'mouse cecum, wild type', fdr_method='bhfdr')
        fbh_mlt = mlt_sub.diff_abundance('Description','mouse cecum, TLR5 knockout',
                                    'mouse cecum, wild type', fdr_method='filterBH')
        sig_ds.append(len(ds_mlt.feature_metadata.index))
        sig_bh.append(len(bh_mlt.feature_metadata.index))
        sig_fbh.append(len(fbh_mlt.feature_metadata.index))

    sig_ds_mlt.append(np.mean(sig_ds))
    sig_bh_mlt.append(np.mean(sig_bh))
    sig_fbh_mlt.append(np.mean(sig_fbh))
    err_ds_mlt.append(sem(sig_ds))
    err_bh_mlt.append(sem(sig_bh))
    err_fbh_mlt.append(sem(sig_fbh))

with open("../results_all/mlt_filtering_updated_B1k.pkl", "wb") as f:
    pickle.dump((filtlev, sig_bh_mlt, sig_ds_mlt, sig_fbh_mlt, err_ds_mlt, err_bh_mlt, err_fbh_mlt), f)