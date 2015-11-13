from __future__ import print_function, division, absolute_import, unicode_literals
import pandas as pd
import numpy as np
import csv
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import TruncatedSVD
import random

ratings = pd.read_csv('ratings.csv')

rows = random.sample(ratings.index,655951)  
cv_test = ratings.ix[rows]    
cv_train = ratings.drop(rows)
cv_ratings = cv_test['Rating']
small_data_train = np.array(cv_train)

    
    #drop the rating from the test set
    
    #TSVD PORTION
cv_test = cv_test[['UserID','ProfileID']]
small_data_test = np.array(cv_test)
user_profile_matrix = pd.concat([cv_train,cv_test]).pivot('UserID','ProfileID','Rating')
profile_means = user_profile_matrix.mean()
user_means = user_profile_matrix.mean(axis=1)
mzm = user_profile_matrix-profile_means
mz = mzm.fillna(0)
mask = -mzm.isnull()
iteration = 0
mse_last = 999
while iteration<10:
    iteration += 1
    svd = TruncatedSVD(n_components=20,random_state=42)
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
print(rmse)

user_ids = [row[0] for row in small_data_train]
user_ids = set(user_ids)
user_ids = sorted(user_ids)
number_of_users = len(user_ids)

profile_ids = [row[1] for row in small_data_train]
profile_ids = set(profile_ids)
profile_ids = sorted(profile_ids)
number_of_profiles = len(profile_ids)
    
user_profile_matrix_1 = np.zeros((10000,10000))
 
for row in small_data_train:
    user_id = user_ids.index(row[0])
    profile_id = profile_ids.index(row[1])
    user_profile_matrix_1[user_id,profile_id] = row[2]
    
K=50

km = MiniBatchKMeans(n_clusters = K)
km.fit(user_profile_matrix_1)

labels = km.labels_
print(str(labels))

cluster_num_users=np.zeros(K)

cluster_list_users=[]
for i in range(K):
    cluster_list_users.append([])
    
prediction = km.predict(user_profile_matrix_1)

for i in range(len(prediction)):
    cluster_num_users[prediction[i]]+=1
    list_of_users = []
    list_of_users = cluster_list_users[prediction[i]]
    list_of_users.append(i)
    cluster_list_users[prediction[i]]=list_of_users
    
f=open('cluster_num_users','w')
for i in range(K):
    f.write(str(i))
    f.write('\t')
    f.write(str(cluster_num_users[i]))
    f.write('\n')
f.close()


Prediction_kaggle = []
nonrated=0
z=0
for row in small_data_test:
    z = z+1
    print(z)
    #print('Testing for 1st user and profile in test : ' + str(row))    
    profile = row[1]
    #print('Cluster for this user : ')
    user = row[0]
    #print(user)
    user_id = user_ids.index(user)
    #print(user_id)
    #print(labels)
    cluster_index = labels[user_id]
    #print(cluster_index)
    
    
    #print('Other user ids  in this cluster : ')
    #print(cluster_num_users[cluster_index])
    #print(len(cluster_list_users[cluster_index]))
    other_user_ids_in_same_cluster=cluster_list_users[cluster_index]
    #print(other_user_ids_in_same_cluster)
    #print('Have they rated profile ')
    #print(profile)
    if profile in profile_ids:
        profile_id=profile_ids.index(profile)
    else:
        continue

    number_of_users_who_rated_profile=0
    sum_total_rating=0
    for i in other_user_ids_in_same_cluster:
        if user_profile_matrix_1[i][profile_id] > 0:
            #print(i)
            #print('index has rated profile ')
            #print(profile_id)
            #print(user_profile_matrix[i][profile_id])
            number_of_users_who_rated_profile+=1
            sum_total_rating+=user_profile_matrix_1[i][profile_id]
    #print('Predicted Rating for this profile :')
    #print(sum_total_rating)
    if(number_of_users_who_rated_profile > 0):
        rating_predicted = sum_total_rating/number_of_users_who_rated_profile
        print(rating_predicted)
        print("")
        Prediction_kaggle = np.append(Prediction_kaggle,rating_predicted)
        #root_mean_accuracy += Decimal(pow(Decimal(rating_predicted-rating),2))
    else:
         print("NO PREVIOUS RATING")
         Prediction_kaggle = np.append(Prediction_kaggle,np.mean(small_data_train[:,2]))
         nonrated=nonrated+1
         
Prediction_kaggle = Prediction_kaggle.T
