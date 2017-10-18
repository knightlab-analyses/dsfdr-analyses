import numpy as np
import pandas as pd
import dsfdr

from biom import load_table
from gneiss.util import match
from scipy.stats import sem
import pickle


# input biom table
def convert_biom_to_pandas(table):
    otu_table = pd.DataFrame(np.array(table.matrix_data.todense()).T,
                             index=table.ids(axis='sample'),
                             columns=table.ids(axis='observation'))
    return otu_table

table = load_table('../data/cs.biom')
otu_table = convert_biom_to_pandas(table)


# input mapping file
mapping = pd.read_table("../data/cs.map.txt", sep='\t', header=0, index_col=0)


# choose interested group for comparison
mapping = mapping.loc[mapping['smoker'].isin([False, True])]


# match mapping file with biom table
mapping, otu_table = match(mapping, otu_table)
labels = np.array((mapping['smoker'] == False).astype(int))
dat = np.transpose(np.array(otu_table))

# normalization
sample_reads = np.sum(dat, axis=0) # colSum: total reads in each sample
norm_length = 10000
dat_norm = dat/sample_reads*norm_length

# one group of the data
same = dat_norm[:, labels==0]

def filtering(data, filterLev):
    otu_sum = np.sum(data, axis=1)
    keep = np.array(otu_sum >= filterLev)
    table = data[keep==True, :]
    return(table) 

# round to nearest even integer
def round_even(f):
    f = np.ceil(f / 2.) * 2 
    return(np.int(f))

# prepare data for multiple comparisons
reads = np.sum(same, axis=1)
labels_large = np.array(reads >= np.median(reads))
labels_small = np.array(reads < np.median(reads))
same_sorted = np.vstack((same[labels_large, :], same[labels_small, :]))

# simulation parameters
filtlev = [0, 2, 4, 6, 8, 10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80, 90, 100]
B = 1000
nSample = 15
numbact = same_sorted.shape[0]
c = 0.1 # proprtion of non nulls

FDR_bh1 = []
FDR_fbh1 = []
FDR_ds1 = []
FDR_gb1 = []

PWR_bh1 = []
PWR_fbh1 = []
PWR_ds1 = []
PWR_gb1 = []

OTU_bh1 = []
OTU_fbh1 = []
OTU_ds1 = []
OTU_gb1 = []

err_bh1 = []
err_fbh1 = []
err_ds1 = []
err_gb1 = []

sd_bh1 = []
sd_fbh1 = []
sd_ds1 = []
sd_gb1 = []

num_nulls = []
for f in filtlev:
    fdr_bh1 = []
    fdr_fbh1 = []
    fdr_ds1 = []
    fdr_gb1 = []

    pwr_bh1 = []
    pwr_fbh1 = []
    pwr_ds1 = []
    pwr_gb1 = []

    otu_bh1 = []
    otu_fbh1 = []
    otu_ds1 = []
    otu_gb1 = []

    num_bact = []

    for b in range(B):   
        p1 = round_even(c*numbact) # number of truly different bacteria
        p0 = numbact - p1

        healthy = np.zeros([numbact, nSample]) 
        sick = np.zeros([numbact, nSample]) 
        for i in range(numbact):
            healthy[i, :] = np.random.choice(same_sorted[i, :], nSample)
            sick[i, :] = np.random.choice(same_sorted[i, :], nSample)
        dat_sim = np.hstack((healthy, sick))

        # changes in both group for p1 bacteria
        for i in range(int(p1/2)):
            mu_H = np.random.uniform(5, 6)
            k = 2
            mu_S = mu_H + k
            sigma = 2
            h1 = healthy[i, :] + np.random.normal(mu_H, sigma, nSample)
            h2 = healthy[i + int(p1/2), :] + np.random.normal(mu_S, sigma, nSample)
            s1 = sick[i, :] + np.random.normal(mu_S, sigma, nSample)
            s2 = sick[i + int(p1/2), :] + np.random.normal(mu_H, sigma, nSample)
            
            dat_sim[i, :] = np.hstack((h1, s1))
            dat_sim[i + int(p1/2), :] = np.hstack((h2, s2))
        
        # labels
        labels_sim = np.hstack((np.repeat(0, nSample), np.repeat(1, nSample)))
        
        # filtering
        dat_sim = filtering(dat_sim, filterLev=f)
        num_bact.append(dat_sim.shape[0])

        # apply FDR methods
        rej_bh1 = dsfdr.dsfdr(dat_sim, labels_sim, transform_type = 'rankdata', method = 'meandiff',
                             alpha=0.1, numperm=1000, fdr_method ='bhfdr')
        rej_fbh1 = dsfdr.dsfdr(dat_sim, labels_sim, transform_type = 'rankdata', method = 'meandiff',
                                     alpha=0.1, numperm=1000, fdr_method ='filterBH')
        rej_ds1 = dsfdr.dsfdr(dat_sim, labels_sim, transform_type = 'rankdata', method = 'meandiff',
                                     alpha=0.1, numperm=1000, fdr_method ='dsfdr')
        rej_gb1 = dsfdr.dsfdr(dat_sim, labels_sim, transform_type = 'rankdata', method = 'meandiff',
                                     alpha=0.1, numperm=1000, fdr_method ='gilbertBH')
        # compute FDR

        v_bh1 = np.sum(np.where(rej_bh1[0])[0]>=p1)
        r_bh1 = np.sum(rej_bh1[0])
        if r_bh1 == 0:
            fdr_bh1.append(0)
        else:
            fdr_bh1.append(v_bh1/r_bh1)
        pwr_bh1.append(np.sum(np.where(rej_bh1[0])[0]<p1)/p1)
        otu_bh1.append(np.sum(np.where(rej_bh1[0])[0]<p1))

        v_fbh1 = np.sum(np.where(rej_fbh1[0])[0]>=p1)
        r_fbh1 = np.sum(rej_fbh1[0])
        if r_fbh1 == 0:
            fdr_fbh1.append(0)
        else:
            fdr_fbh1.append(v_fbh1/r_fbh1)
        pwr_fbh1.append(np.sum(np.where(rej_fbh1[0])[0]<p1)/p1)
        otu_fbh1.append(np.sum(np.where(rej_fbh1[0])[0]<p1))
        
        v_ds1 = np.sum(np.where(rej_ds1[0])[0]>=p1)
        r_ds1 = np.sum(rej_ds1[0])
        if r_ds1 == 0:
            fdr_ds1.append(0)
        else:
            fdr_ds1.append(v_ds1/r_ds1)
        pwr_ds1.append(np.sum(np.where(rej_ds1[0])[0]<p1)/p1)
        otu_ds1.append(np.sum(np.where(rej_ds1[0])[0]<p1))

        v_gb1 = np.sum(np.where(rej_gb1[0])[0]>=p1)
        r_gb1 = np.sum(rej_gb1[0])
        if r_gb1 == 0:
            fdr_gb1.append(0)
        else:
            fdr_gb1.append(v_gb1/r_gb1)
        pwr_gb1.append(np.sum(np.where(rej_gb1[0])[0]<p1)/p1)
        otu_gb1.append(np.sum(np.where(rej_gb1[0])[0]<p1))

    num_nulls.append(np.mean(num_bact))

    FDR_bh1.append(np.nanmean(fdr_bh1))
    FDR_fbh1.append(np.nanmean(fdr_fbh1))
    FDR_ds1.append(np.nanmean(fdr_ds1))
    FDR_gb1.append(np.nanmean(fdr_gb1))

    PWR_bh1.append(np.nanmean(pwr_bh1))
    PWR_fbh1.append(np.nanmean(pwr_fbh1))
    PWR_ds1.append(np.nanmean(pwr_ds1))
    PWR_gb1.append(np.nanmean(pwr_gb1))

    OTU_bh1.append(np.nanmean(otu_bh1))
    OTU_fbh1.append(np.nanmean(otu_fbh1))
    OTU_ds1.append(np.nanmean(otu_ds1))
    OTU_gb1.append(np.nanmean(otu_gb1))

    err_bh1.append(sem(fdr_bh1))
    err_fbh1.append(sem(fdr_fbh1))
    err_ds1.append(sem(fdr_ds1))
    err_gb1.append(sem(fdr_gb1))

    sd_bh1.append(np.std(fdr_bh1, ddof=1))
    sd_fbh1.append(np.std(fdr_fbh1, ddof=1))
    sd_ds1.append(np.std(fdr_ds1, ddof=1))
    sd_gb1.append(np.std(fdr_gb1, ddof=1))

with open("../results_all/simulation_CS_amnon_mixed_fl_s15.pkl", "wb") as f:
    pickle.dump((filtlev, B, c, nSample, num_nulls,
        FDR_bh1, FDR_fbh1, FDR_ds1, FDR_gb1,
        PWR_bh1, PWR_fbh1, PWR_ds1, PWR_gb1,
        OTU_bh1, OTU_fbh1, OTU_ds1, OTU_gb1,
        err_bh1, err_fbh1, err_ds1, err_gb1,
        sd_bh1, sd_fbh1, sd_ds1, sd_gb1), f)
