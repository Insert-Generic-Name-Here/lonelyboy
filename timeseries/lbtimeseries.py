
import pandas as pd
import multiprocessing as mp
import numpy as np
import os


# returns a list of lists having the indices that correspond to linear parts
def find_lin_parts(x, skiplen = 5, include_zeros=False):
	
	xzer = np.argwhere(x==0)
	xdif2=np.diff(x,n=2)
	xdif2[abs(xdif2)<0.00004]=0 # consider numerical issues xzerdif = np.argwhere(xdif2==0)
	if not include_zeros:
		xlin = np.setdiff1d(xzerdif,xzer)
		xlin =xlin.astype('int')
	else:
		xlin = np.setdiff1d(xzerdif,[])
		xlin = xlin.astype('int')
	aa = group_consecutives(xlin)
	linparts_index = []
	for i in range(len(aa)):
		if len(aa[i])>=1:
			lastm = min([aa[i][-1]+2, len(x)])
			# print lastm
			ppp = list([aa[i][0], lastm])
			linparts_index.append(ppp)
	 
	linparts_index = merge_lin_parts(linparts_index)
	
	linparts_out = []
	for i in range(len(linparts_index)):
		if linparts_index[i][1]-linparts_index[i][0]>=skiplen+2:
			linparts_out.append(linparts_index[i])
	
	
	return linparts_out




def find_zero_parts(x):
	''''''
	xzer = np.argwhere(x==0)
	aa = group_consecutives(xzer)
	zeroparts_index = []
	for i in range(len(aa)):
		zeroparts_index.append([aa[i][0][0], aa[i][-1][0]])
		
	return zeroparts_index




def detrending(X):
	'''Return the detrended part of X and the separate trend
	THIS CODE IS SHIT. PLEASE GOD FIX IT. YOU ARE THE ONLY CAPABLE.
	'''
	
	if len(X.shape)==1:
		X=X.reshape((X.shape[0],1))

		Xd = np.zeros(X.shape)    
	Xt = np.zeros(X.shape)
	
	
	for line in range(X.shape[1]):
			
		x = X[:,line].copy()
		
		trend=pd.Series(x).rolling(window=12*12,center=False).median().values
		detrend = np.nan_to_num(x-trend)
		
		Xd[:,line] = detrend
		Xt[:,line] = trend
		
	return Xd, Xt



def find_outlier_indices(df):
    quantiles = df.quantile([0.25, 0.75])
    feature_outlier_indices = []
    for feature in quantiles:
        q25, q75 = quantiles[feature].values
        iqr = q75 - q25
        
        outlier_indices = df.index[(df[feature] < q25 - (1.5 * iqr)) | (df[feature] > q75 + (1.5 * iqr))].tolist()
        feature_outlier_indices.append((feature, outlier_indices))
    return feature_outlier_indices

def drop_outliers(df, outliers):
    indices = set([])
    for col, col_indices in outliers:
        indices = indices.union(set(col_indices))
    return df.drop(list(indices))

