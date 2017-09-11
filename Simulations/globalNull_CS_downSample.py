import numpy as np
import pandas as pd
import dsfdr

from biom import load_table
from gneiss.util import match
from scipy.stats import sem
import pickle


# In[2]:

def convert_biom_to_pandas(table):
    otu_table = pd.DataFrame(np.array(table.matrix_data.todense()).T,
                             index=table.ids(axis='sample'),
                             columns=table.ids(axis='observation'))
    return otu_table


# In[3]:

table = load_table('../data/cs.biom')
otu_table = convert_biom_to_pandas(table)


# In[4]:

mapping = pd.read_table("../data/cs.map.txt", sep='\t', header=0, index_col=0)


# In[5]:

mapping = mapping.loc[mapping['smoker'].isin ([False, True])]


# In[6]:

mapping, otu_table = match(mapping, otu_table)


# In[7]:

labels = np.array((mapping['smoker'] == False).astype(int))


# In[8]:

dat = np.transpose(np.array(otu_table))

# normalization
sample_reads = np.sum(dat, axis=0) # colSum: total reads in each sample
norm_length = 10000
dat_norm = dat/sample_reads*norm_length

def fwer(rej):
    if np.sum(rej) >= 1:
        r = 1
    else:
        r = 0    
    return r  

# filter reads whose sum in all samples
def filtering_sum(data, filterLev):
    otu_sum = np.sum(data, axis=1)
    keep = np.array(otu_sum >= filterLev)
    table = data[keep==True, :]
    return(table)

sample_range = [10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80, 90]
B = 100000
filtlev = 1000
same = dat_norm[:, labels==0]
same = filtering_sum(same, filterLev=filtlev)
numbact = same.shape[0] 

p_bh = []
p_fbh = []
p_ds = []

err_bh = []
err_fbh = []
err_ds = []
for nSample in sample_range:
    r_bh = []
    r_fbh = []
    r_ds = []
    for b in range(B):
        # simulated data
        sim = np.zeros([numbact, nSample*2])
        for i in range(numbact):
            sim[i, :] = np.random.choice(same[i, :], nSample*2)
        # simulated labels   
        labels_sim = np.random.randint(2, size=nSample*2)
        healthy = sim[:, labels_sim==0]
        sick = sim[:, labels_sim==1]
        dat_sim = np.hstack((healthy, sick))

        # filtering    
        #dat_sim = filtering_sum(dat_sim, filterLev=filtlev)

        # apply FDR methods
        rej_bh = dsfdr.dsfdr(dat_sim, labels_sim, transform_type = 'rankdata', method = 'meandiff',
                             alpha=0.1, numperm=1000, fdr_method ='bhfdr')
        rej_fbh = dsfdr.dsfdr(dat_sim, labels_sim, transform_type = 'rankdata', method = 'meandiff',
                                     alpha=0.1, numperm=1000, fdr_method ='filterBH')
        rej_ds = dsfdr.dsfdr(dat_sim, labels_sim, transform_type = 'rankdata', method = 'meandiff',
                                     alpha=0.1, numperm=1000, fdr_method ='dsfdr')

        # total sum of fwer
        r_bh.append(fwer(rej_bh[0]))
        r_fbh.append(fwer(rej_fbh[0]))
        r_ds.append(fwer(rej_ds[0]))

    print('FDR...: %s' %(nSample))
    p_bh.append(np.mean(r_bh))
    p_fbh.append(np.mean(r_fbh))
    p_ds.append(np.mean(r_ds))

    err_bh.append(sem(r_bh))
    err_fbh.append(sem(r_fbh))
    err_ds.append(sem(r_ds))

with open("../results_downsampling/simulation_cs_norm_downSample_fl1000_B100k.pkl", "wb") as f:
    pickle.dump((filtlev, B, sample_range, numbact, p_bh, p_fbh, p_ds, err_bh, err_fbh, err_ds), f)   
