
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 12:14:38 2017

@author: Guillaume Monarcha
"""

import numpy as np
import pandas as pd
import scipy as sp
import scipy.stats as stats
from sklearn import preprocessing
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from matplotlib import pyplot as plt
from statsmodels.multivariate.pca import PCA as PCA
import statsmodels.api as sm



### DATA LOADING #####################################################################
# ARP returns
arp=np.array(pd.read_excel('ARP_FX.xlsx', 'FX',usecols='B:AT', skiprows=3))
# ARP LABELS
LABELSl=pd.read_excel('ARP_FX.xlsx', 'FX',usecols='B:AT')
LABELSs=list(LABELSl.columns.values)

#RETURN STANDARDIZATION
ARPs=preprocessing.scale(arp,axis=0,with_mean=False)

# PCA DECOMPOSITION OF THE COVARIANCE MATRIX
pca_out = PCA(ARPs)
pca_ic=pca_out.ic
pca_var=pca_out.eigenvals/np.sum(pca_out.eigenvals)
pca_eigv=pca_out.eigenvals/np.mean(pca_out.eigenvals)
# NUMBER OF PRINCIPAL COMPONENTS?
n_comp=6
pca_comp=pca_out.factors[:,0:n_comp]
# FEATURES COMPUTATION
features=np.zeros((n_comp,arp.shape[1]))
pca_comp_n=preprocessing.scale(pca_comp,axis=0,with_mean=False)
for i in range (0,arp.shape[1],1):
    model = sm.OLS(ARPs[:,i],sm.add_constant(pca_comp_n))
    results = model.fit()
    features[:,i]=results.params[1:]
features_n=preprocessing.scale(features,axis=1)

alphas=np.zeros((ARPs.shape[1],1))
### FEATURES ESTIMATION: MACRO-BASED
for i in range(0,ARPs.shape[1],1):
    # OLS ESTIMATION OF MACRO EXPOSURES
    model = sm.OLS(ARPs[:,i],sm.add_constant(pca_comp_n))
    # FEATURES MATRIX = MACRO EXPOSURES
    results = model.fit()
    alphas[i]=results.params[0]
    
### DISTANCE MEASURES
features=preprocessing.scale(features,axis=0)
dist=pdist(features.T,'euclidean')

### CLUSTERING ALGO
output = linkage(dist,method='ward')

### CHART DENDOGRAM
plt.figure(figsize=(10, 15))
csfont = {'fontname':'Calibri', 'fontsize' : '10'}
csfont2 = {'fontname':'Calibri', 'fontsize' : '8'}
csfont3 = {'fontname':'Calibri', 'fontsize' : '6'}
plt.xlabel('Distance',**csfont2)
plt.xticks(**csfont3)
plt.yticks(**csfont3)
dendrogram(output,color_threshold=5,truncate_mode='level',orientation='right',leaf_font_size=12,labels=LABELSs)
plt.savefig('books_read.jpeg',bbox_inches='tight', dpi=900)
plt.show()


### IDENTIFY THE CLUSTER COMPOSITION
nclusters=3
cluster=fcluster(output,nclusters,criterion='maxclust')

ARP_select=np.zeros((ARPs.shape[1],3))
### SELECT THE MOST REPRESENTATIVE ARPs
for i in range (1,nclusters+1,1):
    features_c=features.T[cluster==i,:]
    center=np.mean(features_c,axis=0)
    features_c2=np.concatenate((features_c,center),axis=0)
    dist_c=pdist(features_c2,'euclidean')
    dist_m_c=squareform(dist_c)
    dist_cent=dist_m_c[0:dist_m_c.shape[1]-1,dist_m_c.shape[1]-1]
    alpha_c=alphas[cluster==i,:]
    #select the 50% most representative ARPs
    select=(dist_cent<np.quantile(dist_cent,0.5))
    # select the 50% best performing ARPs within the most representative ones
    select_alphas=alpha_c[select,0]>np.quantile(alpha_c[select,0],0.5)
    select[select==1]=select_alphas
    ARP_select[cluster==i,i-1]=select