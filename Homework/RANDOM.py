# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 12:14:38 2017

@author: Guillaume Monarcha
"""

import numpy as np
import pandas as pd
from sklearn import preprocessing
import scipy.stats as stats
import statsmodels.api as sm
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from matplotlib import pyplot as plt
from statsmodels.multivariate.pca import PCA as PCA

SECTORS=np.array(pd.read_excel('HOMEWORK_3.xlsx', 'Feuil1',usecols='B:K'))
MARKET=np.array(pd.read_excel('HOMEWORK_3.xlsx', 'Feuil1',usecols='L'))
RF=np.array(pd.read_excel('HOMEWORK_3.xlsx', 'Feuil1',usecols='M'))
FUNDS=np.array(pd.read_excel('HOMEWORK_3.xlsx', 'Feuil1',usecols='N:O'))


n_sim=1000
ret_sim=np.zeros((len(SECTORS),n_sim))
stats_sim=np.zeros((3,n_sim))

for i in range(n_sim):
    print(i)
    score_t=np.random.normal(0,1,SECTORS.shape)
    for t in range(len(SECTORS)):
        select=score_t[t,:]>=np.quantile(score_t[t,:],0.7)
        ret_sim[t,i]=np.mean(SECTORS[t,select])
        
# annualized returns        
stats_sim[0,:]=np.mean(ret_sim,axis=0)*12
# Sharpe ratios
stats_sim[1,:]=(np.mean(ret_sim,axis=0)*12-np.mean(RF))/(np.std(ret_sim,axis=0)*12**0.5)
# alpha vs. SP500
for i in range(n_sim):
    model = sm.OLS(ret_sim[:,i],sm.add_constant(MARKET))
    results = model.fit()
    stats_sim[2,i]=results.params[0]*12

### STATISTICS LUCKY THRESHOLDS
threshlods=np.quantile(stats_sim,0.99,axis=1)

### FUND STATS
print("Annualized performances", np.mean(FUNDS,axis=0)*12)
print("Sharpe Ratios", (np.mean(FUNDS,axis=0)*12-np.mean(RF))/(np.std(FUNDS,axis=0)*12**0.5))
model = sm.OLS(FUNDS[:,0],sm.add_constant(MARKET))
results = model.fit()
print("Alpha fund 1:", results.params[0]*12)
model = sm.OLS(FUNDS[:,1],sm.add_constant(MARKET))
results = model.fit()
print("Alpha fund 1:", results.params[0]*12)