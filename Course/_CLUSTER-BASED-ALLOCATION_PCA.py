
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 12:14:38 2017

@author: Guillaume Monarcha
"""

import numpy as np
import pandas as pd
import scipy as sp
import scipy.stats as stats
import statsmodels.api as sm

from sklearn import preprocessing
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from matplotlib import pyplot as plt
from statsmodels.multivariate.pca import PCA as PCA

import statsmodels.api as sm

# LOADING #####################################################################
arp=np.array(pd.read_excel('ARP_INDICES.xlsx', 'IND',usecols='B:AT'))
macro=np.array(pd.read_excel('ARP_INDICES.xlsx', 'MACRO',usecols='B:D'))
labels=pd.read_excel('ARP_INDICES.xlsx', 'IND',usecols='B:AT')
LABl=list(labels.columns.values)

### 1. HIERARCHICAL CLUSTERING
## FEATURES BASED ON EXPOSURES TO MARKET RISKS

# ARP STANDARDIZATION
arpn=preprocessing.scale(arp)


## FEATURES BASED ON EXPOSURES UNDERLYING RISKS (PCA DECOMPOSITION)
# PCA DECOMPOSITION OF THE COVARIANCE MATRIX
pca_out = PCA(arp)
pca_ic=pca_out.ic
pca_var=pca_out.eigenvals/np.sum(pca_out.eigenvals)
pca_eigv=pca_out.eigenvals/np.mean(pca_out.eigenvals)
# NUMBER OF PRINCIPAL COMPONENTS?
n_comp=5
pca_comp=pca_out.factors[:,0:n_comp]
# FEATURES COMPUTATION
features=np.zeros((n_comp,arp.shape[1]))
pca_comp_n=preprocessing.scale(pca_comp,axis=0,with_mean=False)
for i in range (0,arp.shape[1],1):
    model = sm.OLS(arpn[:,i],sm.add_constant(pca_comp_n))
    results = model.fit()
    features[:,i]=results.params[1:]
features_n=preprocessing.scale(features,axis=1)

## DENDOGRAM
# DISTANCES
dist=pdist(features.T,'euclidean')
# LINKAGE
output = linkage(dist,method='ward')
# GRAPH
plt.figure(figsize=(10, 10))
csfont = {'fontname':'Calibri', 'fontsize' : '10'}
csfont2 = {'fontname':'Calibri', 'fontsize' : '8'}
csfont3 = {'fontname':'Calibri', 'fontsize' : '6'}
plt.xlabel('Distance',**csfont2)
plt.xticks(**csfont3)
plt.yticks(**csfont3)
dendrogram(output,color_threshold=1,truncate_mode='level',orientation='right',leaf_font_size=10,labels=LABl)
plt.savefig('books_read.jpeg',bbox_inches='tight', dpi=300)
plt.show()

## CLUSTERS
# CLUSTERING ##################################################################
n_clust=6
clusters=fcluster(output,n_clust,criterion='maxclust')

## WITHIN CLUSTER ALLOCATIONS - EQUAL VOL
w_within=np.zeros((arp.shape[1],n_clust))
for i in range (1,n_clust+1,1):
    vol=np.std(arp[:,clusters==i],axis=0)
    w_within[clusters==i,i-1]=1/vol
w_within=w_within/np.sum(w_within,axis=0)

## ACROSS CLUSTER ALLOCATION - EQUAL CVaR(95%)
# 4.1 RETURNS OF CLUSTER PORTFOLIOS
r_clusters=np.zeros((arp.shape[0],n_clust))
for i in range (0,n_clust,1):
    r_clusters[:,i]=arp.dot(w_within[:,i])
# 4.2 CLUSTER CVaR(95%)
CVaR_c=np.zeros((n_clust))
for i in range (0,n_clust,1):
    CVaR_c[i]=abs(np.mean(r_clusters[r_clusters[:,i]<np.quantile(r_clusters[:,i],0.05),i],axis=0))
# 4.3 CLUSTER ALLOCATION
w_c=(1/CVaR_c)/np.sum(1/CVaR_c)

## GLOBAL ALLOCATION
w=w_within.dot(w_c)