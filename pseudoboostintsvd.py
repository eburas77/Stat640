'''
this script is to play around with schemes to create prediction rule from samples then modify prediction rule
from poorly predicted cv samples. Can be multiple rounds of prediction rules
'''

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import TruncatedSVD
import random

ratings = pd.read_csv('ratings.csv')

rows = random.sample(ratings.index,655951)
cv_train=ratings.drop(rows)
cv_test = ratings.ix[rows]
cv_ratings = cv_test['Rating']
train_rows = random.sample(cv_train.index,1311904)

cv_train_1 = cv_train.ix[train_rows]
cv_train_2 = cv_train.drop(train_rows)    
    #drop the rating from the test set
cv_test = cv_test[['UserID','ProfileID']]


cv_train_ratings = cv_train_2['Rating']
cv_train_2 = cv_train_2[['UserID','ProfileID']]
user_profile_matrix = pd.concat([cv_train_1,cv_train_2]).pivot('UserID','ProfileID','Rating')
profile_means = user_profile_matrix.mean()
user_means = user_profile_matrix.mean(axis=1)
mzm = user_profile_matrix-profile_means
mz = mzm.fillna(0)
mask = -mzm.isnull()
iteration = 0
mse_last = 999
    

while iteration<10:
    iteration += 1
    svd = TruncatedSVD(n_components=25,random_state=42)
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

    #creating rating for cv_train_2
cv_train_2['newrating'] = cv_train_2.apply(lambda x:m[m.index==x.UserID][x.ProfileID].values[0],axis=1)

    # There are some profiles who did not have enough info to make prediction, so just used average value for user
missing = np.where(cv_train_2.newrating.isnull())[0]
cv_train_2.ix[missing,'newrating'] = user_means[cv_train_2.loc[missing].UserID].values
cv_train_2['Rating']=cv_train_ratings
#only find the samples where the prediction was off by 2 or more
cv_add = cv_train_2[abs(cv_train_2['newrating']-cv_train_2['Rating'])>1.5]
#drop newrating
cv_add = cv_add[['UserID','ProfileID','Rating']]

cv_train_final = cv_train_1.append(cv_add)
#do same algorithm to find prediction rule on only those samples that were poorly predicted on
user_profile_matrix1 = pd.concat([cv_train_final,cv_test]).pivot('UserID','ProfileID','Rating')
profile_means1 = user_profile_matrix1.mean()
user_means1 = user_profile_matrix1.mean(axis=1)
mzm1 = user_profile_matrix1-profile_means1
mz1 = mzm1.fillna(0)
mask1 = -mzm1.isnull()
iteration1 = 0
mse_last1 = 999
    

while iteration1<10:
    iteration1 += 1
    svd1 = TruncatedSVD(n_components=25,random_state=42)
    svd1.fit(mz1)
    mzsvd1 = pd.DataFrame(svd1.inverse_transform(svd1.transform(mz1)),columns=mz1.columns,index=mz1.index)
        
    mse1 = mean_squared_error(mzsvd1[mask1].fillna(0),mzm1[mask1].fillna(0))
    print('%i %.5f %.5f'%(iteration1,mse1,mse_last1-mse1))
    mzsvd1[mask1] = mzm1[mask1]
    
    mz1 = mzsvd1
    if mse_last1-mse1<0.00001: break
    mse_last1 = mse1

m1 = mz1+profile_means1
m1 = m1.clip(lower=1,upper=10)

#create prediction rule from original m and newly found m1 from badly predicted samples
#this weighting should be mixed up to find best optimal 
mfinal=(m+m1)/2
    #creating rating for cv_test
cv_test['newrating'] = cv_test.apply(lambda x:m1[m1.index==x.UserID][x.ProfileID].values[0],axis=1)

    # There are some profiles who did not have enough info to make prediction, so just used average value for user
missing1 = np.where(cv_test.newrating.isnull())[0]
cv_test.ix[missing1,'newrating'] = user_means1[cv_test.loc[missing1].UserID].values

rmse = np.sqrt(sum(pow((cv_ratings-cv_test['newrating']),2))/len(cv_ratings))
print('testing rmse')
print(rmse)

IDmap = pd.read_csv('IDmap.csv')
ID = IDmap['KaggleID']

IDmap['newrating'] = IDmap.apply(lambda x:m1[m1.index==x.UserID][x.ProfileID].values[0],axis=1)

missing3 = np.where(IDmap.newrating.isnull())[0]
IDmap.ix[missing3,'newrating'] = user_means[IDmap.loc[missing].UserID].values

IDmap['Prediction'] = IDmap['newrating']
IDmap = IDmap[['KaggleID','Prediction']]
IDmap.columns = ['ID','Prediction']
IDmap.to_csv('submissiontsvdboost.csv',index=False,columns=['ID','Prediction'])









#cv_train_final = cv_train_1.append(cv_add)
#
#user_profile_matrix = pd.concat([cv_train_1,cv_train_2]).pivot('UserID','ProfileID','Rating')
#profile_means = user_profile_matrix.mean()
#user_means = user_profile_matrix.mean(axis=1)
#mzm = user_profile_matrix-profile_means
#mz = mzm.fillna(0)
#mask = -mzm.isnull()
#iteration = 0
#mse_last = 999
#    
#
#while iteration<10:
#    iteration += 1
#    svd = TruncatedSVD(n_components=40,random_state=42)
#    svd.fit(mz)
#    mzsvd = pd.DataFrame(svd.inverse_transform(svd.transform(mz)),columns=mz.columns,index=mz.index)
#        
#    mse = mean_squared_error(mzsvd[mask].fillna(0),mzm[mask].fillna(0))
#    print('%i %.5f %.5f'%(iteration,mse,mse_last-mse))
#    mzsvd[mask] = mzm[mask]
#    
#    mz = mzsvd
#    if mse_last-mse<0.00001: break
#    mse_last = mse
#
#m = mz+profile_means
#m = m.clip(lower=1,upper=10)
#
#    #creating rating for cv_test
#cv_train_2['newrating'] = cv_train_2.apply(lambda x:m[m.index==x.UserID][x.ProfileID].values[0],axis=1)
#
#    # There are some movies who did not have enough info to make prediction, so just used average value for user
#missing = np.where(cv_train_2.newrating.isnull())[0]
#cv_train_2.ix[missing,'newrating'] = user_means[cv_train_2.loc[missing].UserID].values




    