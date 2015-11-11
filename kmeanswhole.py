# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 11:39:28 2015

@author: ericburas
"""
from __future__ import print_function, division, absolute_import, unicode_literals
import numpy as np
import pandas as pd

from decimal import *
import csv
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.cluster import v_measure_score
from math import *

ratings = pd.read_csv('ratings.csv',header=0)

r = np.array(ratings)

sub = 10000


small_data = np.array(ratings)

small_data_train = small_data
small_data_test = pd.read_csv('Idmap.csv',header=0)
small_data_test = np.array(small_data_test)

user_ids = [row[0] for row in small_data_train]
user_ids = set(user_ids)
user_ids = sorted(user_ids)
number_of_users = len(user_ids)

profile_ids = [row[1] for row in small_data_train]
profile_ids = set(profile_ids)
profile_ids = sorted(profile_ids)
number_of_profiles = len(profile_ids)
    
user_profile_matrix = np.zeros((sub,sub))
 
for row in small_data_train:
    user_id = user_ids.index(row[0])
    profile_id = profile_ids.index(row[1])
    user_profile_matrix[user_id,profile_id] = row[2]
    
K=50

km = MiniBatchKMeans(n_clusters = K)
km.fit(user_profile_matrix)

labels = km.labels_
print(str(labels))

cluster_num_users=np.zeros(K)

cluster_list_users=[]
for i in range(K):
    cluster_list_users.append([])
    
prediction = km.predict(user_profile_matrix)

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
    Kaggle_id = row[2]
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
        if user_profile_matrix[i][profile_id] > 0:
            #print(i)
            #print('index has rated profile ')
            #print(profile_id)
            #print(user_profile_matrix[i][profile_id])
            number_of_users_who_rated_profile+=1
            sum_total_rating+=user_profile_matrix[i][profile_id]
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
         Prediction_kaggle = np.append(Prediction_kaggle,np.mean(small_data[:,2]))
         nonrated=nonrated+1
         
Prediction_kaggle = Prediction_kaggle.T
Prediction_kaggle = np.reshape(Prediction_kaggle,(len(Prediction_kaggle),1))
ID = small_data_test[:,2]
ID = ID.T
ID = np.reshape(ID,(len(ID),1))
final = np.hstack((ID,Prediction_kaggle))


np.savetxt('kmeans.csv',final)
np.savetxt('kmeansdating.csv',final,header="ID,Prediction",delimiter=',')

print(nonrated)