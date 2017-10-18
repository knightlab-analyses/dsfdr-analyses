import numpy as np
import scipy as sp
import scipy.stats
from scipy.stats import sem

import pickle

import transform
import statistics
import simulation
import dsfdr

# simple simulation II
# setting: 100 truly differential taxa, 100 truly similar taxa, sample size to be 50 in each group
# varying rare taxa from 500 to 10000 to create different discreteness

# simulation parameters
np.random.seed(31)
B = 100
d = [500, 1000, 2000, 4000, 6000, 8000, 10000]

otu_bh2 = []
otu_fbh2 = []
otu_ds2 = []
otu_gb2 = []

FDR_bh2 = []
FDR_fbh2 = []
FDR_ds2 = []
FDR_gb2 = []

pval_bh2 = []
pval_fbh2 = []
pval_ds2 = []
pval_gb2 = []

PWR_bh2 = []
PWR_fbh2 = []
PWR_ds2 = []
PWR_gb2 = []

err_bh2 = []
err_fbh2 = []
err_ds2 = []
err_gb2 = []

sd_bh2 = []
sd_fbh2 = []
sd_ds2 = []
sd_gb2 = []

for j in d:
    fdr_bh2 = []
    fdr_fbh2 = []
    fdr_ds2 = []
    fdr_gb2 = []

    sig_bh2 = []
    sig_fbh2 = []
    sig_ds2 = []
    sig_gb2 = []

    pwr_bh2 = []
    pwr_fbh2 = []
    pwr_ds2 = []
    pwr_gb2 = []

    for i in range(B):
        # simulated data
        data2,labels2 = simulation.simulatedat2(numsamples = 50, numdiff=100, numc =100, numd=j)

        # apply FDR methods
        rej_bh2 = dsfdr.dsfdr(data2, labels2, transform_type = 'rankdata', method = 'meandiff',
                             alpha=0.1, numperm=1000, fdr_method ='bhfdr')
        rej_fbh2 = dsfdr.dsfdr(data2, labels2, transform_type = 'rankdata', method = 'meandiff',
                             alpha=0.1, numperm=1000, fdr_method ='filterBH')
        rej_ds2 = dsfdr.dsfdr(data2, labels2, transform_type = 'rankdata', method = 'meandiff',
                             alpha=0.1, numperm=1000, fdr_method ='dsfdr')
        rej_gb2 = dsfdr.dsfdr(data2, labels2, transform_type = 'rankdata', method = 'meandiff',
                             alpha=0.1, numperm=1000, fdr_method ='gilbertBH')

        v_bh2 = np.sum(np.where(rej_bh2[0])[0]>=100)
        r_bh2 = np.sum(rej_bh2[0])
        sig_bh2.append(r_bh2)
        if r_bh2 == 0:
            fdr_bh2.append(0)
        else:
            fdr_bh2.append(v_bh2/r_bh2)
        pval_bh2.append(rej_bh2[2])
        pwr_bh2.append(np.sum(np.where(rej_bh2[0])[0]<100)/100)
        
        v_fbh2 = np.sum(np.where(rej_fbh2[0])[0]>=100)
        r_fbh2 = np.sum(rej_fbh2[0])
        sig_fbh2.append(r_fbh2)
        if r_fbh2 == 0:
            fdr_fbh2.append(0)
        else:
            fdr_fbh2.append(v_fbh2/r_fbh2)
        pval_fbh2.append(rej_fbh2[2])
        pwr_fbh2.append(np.sum(np.where(rej_fbh2[0])[0]<100)/100)
        
        v_ds2 = np.sum(np.where(rej_ds2[0])[0]>=100)
        r_ds2 = np.sum(rej_ds2[0])
        sig_ds2.append(r_ds2)
        if r_ds2 == 0:
            fdr_ds2.append(0)
        else:
            fdr_ds2.append(v_ds2/r_ds2)
        pval_ds2.append(rej_ds2[2])
        pwr_ds2.append(np.sum(np.where(rej_ds2[0])[0]<100)/100)

        v_gb2 = np.sum(np.where(rej_gb2[0])[0]>=100)
        r_gb2 = np.sum(rej_gb2[0])
        sig_gb2.append(r_gb2)
        if r_gb2 == 0:
            fdr_gb2.append(0)
        else:
            fdr_gb2.append(v_gb2/r_gb2)
        pval_gb2.append(rej_gb2[2])
        pwr_gb2.append(np.sum(np.where(rej_gb2[0])[0]<100)/100)

    #print('otu...: %s' %(j)) 
    otu_bh2.append(np.mean(sig_bh2))
    otu_fbh2.append(np.mean(sig_fbh2))
    otu_ds2.append(np.mean(sig_ds2))
    otu_gb2.append(np.mean(sig_gb2))

    #print('FDR...: %s' %(j)) 
    FDR_bh2.append(np.nanmean(fdr_bh2))
    FDR_fbh2.append(np.nanmean(fdr_fbh2))
    FDR_ds2.append(np.nanmean(fdr_ds2))
    FDR_gb2.append(np.nanmean(fdr_gb2))

    #print('PWR...: %s' %(j))
    PWR_bh2.append(np.nanmean(pwr_bh2))
    PWR_fbh2.append(np.nanmean(pwr_fbh2))
    PWR_ds2.append(np.nanmean(pwr_ds2))
    PWR_gb2.append(np.nanmean(pwr_gb2))

    err_bh2.append(sem(fdr_bh2))
    err_fbh2.append(sem(fdr_fbh2))
    err_ds2.append(sem(fdr_ds2))
    err_gb2.append(sem(fdr_gb2))

    sd_bh2.append(np.std(fdr_bh2, ddof=1))
    sd_fbh2.append(np.std(fdr_fbh2, ddof=1))
    sd_ds2.append(np.std(fdr_ds2, ddof=1))
    sd_gb2.append(np.std(fdr_gb2, ddof=1))


with open("../results_all/simulation2_B100.pkl", "wb") as f:
    pickle.dump((d, otu_bh2, otu_fbh2, otu_ds2, otu_gb2,
                 FDR_bh2, FDR_fbh2, FDR_ds2, FDR_gb2,
                 pval_bh2, pval_fbh2, pval_ds2, pval_gb2,
                 PWR_bh2, PWR_fbh2, PWR_ds2, PWR_gb2,
                 err_bh2, err_fbh2, err_ds2, err_gb2,
                 sd_bh2, sd_fbh2, sd_ds2, sd_gb2), f)
