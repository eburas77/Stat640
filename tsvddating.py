# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 17:25:18 2015

@author: ericburas
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import TruncatedSVD
import random

ratings = pd.read_csv('ratings.csv')
Kfolds = 5
kvec = np.array([10,15,20,25,30,35,40])

rows = random.sample(ratings.index,3279759)
l = len(ratings)/Kfolds
err_array = np.zeros((len(kvec),Kfolds))
for i in range(0,Kfolds):
    cvrows = rows[i*l:(i+1)*l]
    if i == Kfolds-1:
        cvrows = rows[i*l:len(ratings)-1]
    cv_test = ratings.ix[cvrows]    
    cv_train = ratings.drop(cvrows)
    cv_ratings = cv_test['Rating']
    
    #drop the rating from the test set
    cv_test = cv_test[['UserID','ProfileID']]
    user_profile_matrix = pd.concat([cv_train,cv_test]).pivot('UserID','ProfileID','Rating')
    profile_means = user_profile_matrix.mean()
    user_means = user_profile_matrix.mean(axis=1)
    mzm = user_profile_matrix-profile_means
    mz = mzm.fillna(0)
    mask = -mzm.isnull()
    for j in range(0,len(kvec)):
        k = kvec[j]
        print(k)
        iteration = 0
        mse_last = 999
        while iteration<10:
            iteration += 1
            svd = TruncatedSVD(n_components=k,random_state=42)
            svd.fit(mz)
            mzsvd = pd.DataFrame(svd.inverse_transform(svd.transform(mz)),columns=mz.columns,index=mz.index)

            mse = mean_squared_error(mzsvd[mask].fillna(0),mzm[mask].fillna(0))
            print('%i %.5f %.5f'%(iteration,mse,mse_last-mse))
            mzsvd[mask] = mzm[mask]

            mz = mzsvd
            if mse_last-mse<0.00001: break
            mse_last = mse

        m = mz+profile_means
        m = m.clip(lower=1,upper=10)

    #creating rating for cv_test
        cv_test['newrating'] = cv_test.apply(lambda x:m[m.index==x.UserID][x.ProfileID].values[0],axis=1)

    # There are some movies who did not have enough info to make prediction, so just used average value for user
        missing = np.where(cv_test.newrating.isnull())[0]
        cv_test.ix[missing,'newrating'] = user_means[cv_test.loc[missing].UserID].values

        rmse = np.sqrt(sum(pow((cv_ratings-cv_test['newrating']),2))/len(cv_ratings))
        err_array[j,i]=rmse
        print(err_array)

foldsavg = np.array((len(kvec),1))
for p in range(0,len(kvec)):   
    foldsavg[p,1] = np.mean(err_array[p,:])

kmin =kvec[np.argmin(foldsavg)]

IDmap = pd.read_csv('IDmap.csv')
ID = IDmap['KaggleID']
IDmapids = IDmap[['UserID','ProfileID']]
user_profile_matrix = pd.concat([ratings,IDmapids]).pivot('UserID','ProfileID','Rating')
profile_means = user_profile_matrix.mean()
user_means = user_profile_matrix.mean(axis=1)
mzm = user_profile_matrix-profile_means
mz = mzm.fillna(0)
mask = -mzm.isnull()

iteration = 0
mse_last = 999
while iteration<10:
    iteration += 1
    svd = TruncatedSVD(n_components=kmin,random_state=42)
    svd.fit(mz)
    mzsvd = pd.DataFrame(svd.inverse_transform(svd.transform(mz)),columns=mz.columns,index=mz.index)

    mse = mean_squared_error(mzsvd[mask].fillna(0),mzm[mask].fillna(0))
    print('%i %.5f %.5f'%(iteration,mse,mse_last-mse))
    mzsvd[mask] = mzm[mask]

    mz = mzsvd
    if mse_last-mse<0.00001: break
    mse_last = mse

m = mz+profile_means
m = m.clip(lower=1,upper=10)

#creating rating for test set
IDmapids['newrating'] = IDmapids.apply(lambda x:m[m.index==x.UserID][x.ProfileID].values[0],axis=1)

missing = np.where(IDmapids.newrating.isnull())[0]
IDmapids.ix[missing,'newrating'] = user_means[IDmapids.loc[missing].UserID].values

IDmap['Prediction'] = IDmapids['newrating']
IDmap = IDmap[['KaggleID','Prediction']]
IDmap.columns = ['ID','Prediction']
IDmap.to_csv('submissiontsvd.csv',index=False,columns=['ID','Prediction'])