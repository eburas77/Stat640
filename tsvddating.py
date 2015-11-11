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

#want 80/20 split for initial cross validation
rows = random.sample(ratings.index,2623807)
ratings_80 = ratings.ix[rows]
ratings_20 = ratings.drop(rows)

user_profile_matrix = pd.concat([ratings_80,ratings_20]).pivot('UserID','ProfileID','Rating')
profile_means = user_profile_matrix.mean()
user_means = user_profile_matrix.mean(axis=1)
mzm = user_profile_matrix-profile_means
mz = mzm.fillna(0)
mask = -mzm.isnull()

iteration = 0
mse_last = 999
while iteration<10:
    iteration += 1
    svd = TruncatedSVD(n_components=15,random_state=42)
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
ratings_20['newrating'] = ratings_20.apply(lambda x:m[m.index==x.UserID][x.ProfileID].values[0],axis=1)

# There are some movies who did not have enough info to make prediction, so just used average value for user
missing = np.where(ratings_20.newrating.isnull())[0]
ratings_20.ix[missing,'newrating'] = user_means[ratings_20.loc[missing].UserID].values



