import numpy as np
import scipy as sp
import scipy.stats
from scipy.stats import sem

import pickle

import transform
import statistics
import simulation
import dsfdr


# simple simulation I
# setting: 800 truly differential taxa, 100 truly similar taxa and 100 rare taxa
# varying sample size from 10 to 100 to create different discreteness

np.random.seed(31)

B = 100
ss1 = [10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80, 90, 100]


otu_bh1 = []
otu_fbh1 = []
otu_ds1 = []
otu_gb1 = []

pval_bh1 = []
pval_fbh1 = []
pval_ds1 = []
pval_gb1 = []

FDR_bh1 = []
FDR_fbh1 = []
FDR_ds1 = []
FDR_gb1 = []

PWR_bh1 = []
PWR_fbh1 = []
PWR_ds1 = []
PWR_gb1 = []

err_bh1 = []
err_fbh1 = []
err_ds1 = []
err_gb1 = []

sd_bh1 = []
sd_fbh1 = []
sd_ds1 = []
sd_gb1 = []

for j in ss1:
    fdr_bh1 = []
    fdr_fbh1 = []
    fdr_ds1 = []
    fdr_gb1 = []

    sig_bh1 = []
    sig_fbh1 = []
    sig_ds1 = []
    sig_gb1 = []

    pwr_bh1 = []
    pwr_fbh1 = []
    pwr_ds1 = []
    pwr_gb1 = []

    for i in range(B):
        data1,labels1 = simulation.simulatedat2(numsamples = j, sigma=2, numdiff=100, numc =100, numd=800)
        rej_bh1 = dsfdr.dsfdr(data1, labels1, transform_type = 'rankdata', method = 'meandiff',
                             alpha=0.1,numperm=1000, fdr_method ='bhfdr')
        rej_fbh1 = dsfdr.dsfdr(data1, labels1, transform_type = 'rankdata', method = 'meandiff',
                             alpha=0.1,numperm=1000, fdr_method ='filterBH')
        rej_ds1 = dsfdr.dsfdr(data1, labels1, transform_type = 'rankdata', method = 'meandiff',
                             alpha=0.1,numperm=1000, fdr_method ='dsfdr')
        rej_gb1 = dsfdr.dsfdr(data1, labels1, transform_type = 'rankdata', method = 'meandiff',
                             alpha=0.1,numperm=1000, fdr_method ='gilbertBH')

        v_bh1 = np.sum(np.where(rej_bh1[0])[0]>=100)
        r_bh1 = np.sum(rej_bh1[0])
        sig_bh1.append(r_bh1)
        if r_bh1 == 0:
            fdr_bh1.append(0)
        else:
            fdr_bh1.append(v_bh1/r_bh1)
        pval_bh1.append(rej_bh1[2])
        pwr_bh1.append(np.sum(np.where(rej_bh1[0])[0]<100)/100)

        v_fbh1 = np.sum(np.where(rej_fbh1[0])[0]>=100)
        r_fbh1 = np.sum(rej_fbh1[0])
        sig_fbh1.append(r_fbh1)
        if r_fbh1 == 0:
            fdr_fbh1.append(0)
        else:
            fdr_fbh1.append(v_fbh1/r_fbh1)
        pval_fbh1.append(rej_fbh1[2])
        pwr_fbh1.append(np.sum(np.where(rej_fbh1[0])[0]<100)/100)
        
        v_ds1 = np.sum(np.where(rej_ds1[0])[0]>=100)
        r_ds1 = np.sum(rej_ds1[0])
        sig_ds1.append(r_ds1)
        if r_ds1 == 0:
            fdr_ds1.append(0)
        else:
            fdr_ds1.append(v_ds1/r_ds1)
        pval_ds1.append(rej_ds1[2])
        pwr_ds1.append(np.sum(np.where(rej_ds1[0])[0]<100)/100)

        v_gb1 = np.sum(np.where(rej_gb1[0])[0]>=100)
        r_gb1 = np.sum(rej_gb1[0])
        sig_gb1.append(r_gb1)
        if r_gb1 == 0:
            fdr_gb1.append(0)
        else:
            fdr_gb1.append(v_gb1/r_gb1)
        pval_gb1.append(rej_gb1[2])
        pwr_gb1.append(np.sum(np.where(rej_gb1[0])[0]<100)/100)

 
    #print('otu...: %s' %(j))   
    otu_bh1.append(np.mean(sig_bh1))
    otu_fbh1.append(np.mean(sig_fbh1))
    otu_ds1.append(np.mean(sig_ds1))
    otu_gb1.append(np.mean(sig_gb1))

    #print('FDR...: %s' %(j))
    FDR_bh1.append(np.nanmean(fdr_bh1))
    FDR_fbh1.append(np.nanmean(fdr_fbh1))
    FDR_ds1.append(np.nanmean(fdr_ds1))
    FDR_gb1.append(np.nanmean(fdr_gb1))

    #print('PWR...: %s' %(j))
    PWR_bh1.append(np.nanmean(pwr_bh1))
    PWR_fbh1.append(np.nanmean(pwr_fbh1))
    PWR_ds1.append(np.nanmean(pwr_ds1))
    PWR_gb1.append(np.nanmean(pwr_gb1))

    err_bh1.append(sem(fdr_bh1))
    err_fbh1.append(sem(fdr_fbh1))
    err_ds1.append(sem(fdr_ds1))
    err_gb1.append(sem(fdr_gb1))

    sd_bh1.append(np.std(fdr_bh1, ddof=1))
    sd_fbh1.append(np.std(fdr_fbh1, ddof=1))
    sd_ds1.append(np.std(fdr_ds1, ddof=1))
    sd_gb1.append(np.std(fdr_gb1, ddof=1))


with open("../results_all/simulation1_B100.pkl", "wb") as f:
    pickle.dump((ss1, otu_bh1, otu_fbh1, otu_ds1, otu_gb1, 
                 FDR_bh1, FDR_fbh1, FDR_ds1, FDR_gb1, 
                 pval_bh1, pval_fbh1, pval_ds1, pval_gb1, 
                 PWR_bh1, PWR_fbh1, PWR_ds1, PWR_gb1, 
                 err_bh1, err_fbh1, err_ds1, err_gb1, 
                 sd_bh1, sd_fbh1, sd_ds1, sd_gb1), f)
