
from __future__ import print_function, division, absolute_import, unicode_literals
from decimal import *

#other stuff we need to import
import csv
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms import bipartite
from scipy.sparse import csc_matrix
from sklearn.cluster import KMeans
from sklearn.cluster.bicluster import SpectralCoclustering
from sklearn.metrics.cluster import v_measure_score
from math import *

# Input: M scipy.sparse.csc_matrix
# Output: NetworkX Graph
def nx_graph_from_biadjacency_matrix(M):
    # Give names to the nodes in the two node sets
    U = [ "u{}".format(user_ids[i]) for i in range(M.shape[0]) ]
    V = [ "v{}".format(profile_ids[i]) for i in range(M.shape[1]) ]
    
    # Create the graph and add each set of nodes
    G = nx.Graph()
    G.add_nodes_from(U, bipartite=0)
    G.add_nodes_from(V, bipartite=1)
    
    # Find the non-zero indices in the biadjacency matrix to connect
    # those nodes
    G.add_edges_from([ (U[i], V[j]) for i, j in zip(*M.nonzero()) ])
    
    return G

#beginning of main program



ratings = pd.read_csv('ratings.csv',header=0)
IDmap = pd.read_csv('IDmap.csv',header=0)

ratings = np.array(ratings)
IDmap = np.array(IDmap)

user_profile_matrix = np.zeros((10000,10000))



user_ids = [row[0] for row in ratings]
user_ids = set(user_ids)
user_ids = sorted(user_ids)
number_of_users = len(user_ids)


profile_ids = [row[1] for row in ratings]
profile_ids = set(profile_ids)
profile_ids = sorted(profile_ids)
number_of_profiles = len(profile_ids)


for row in ratings:
    user_id = user_ids.index(row[0])
    profile_id = profile_ids.index(row[1])
    user_profile_matrix[user_id,profile_id] = row[2]

#find number of users and movies in each bicluster
'''G = nx_graph_from_biadjacency_matrix(user_movie_matrix)
nx.draw(G)
plt.show()'''

#initialize and carry out clustering
K=50


scc = SpectralCoclustering(n_clusters = K,svd_method='arpack')
scc.fit(user_profile_matrix)

#labels
row_labels = scc.row_labels_
column_labels = scc.column_labels_

bicluster_num_users=np.zeros(K)
bicluster_num_profiles=np.zeros(K)

bicluster_list_users=[]

bicluster_list_profiles=[]

for i in range(K):
    bicluster_list_users.append([])
    bicluster_list_profiles.append([])

for i in range(len(row_labels)):
    bicluster_num_users[row_labels[i]]+=1
    list_of_users = []
    list_of_users = bicluster_list_users[row_labels[i]]
    list_of_users.append(i)
    bicluster_list_users[row_labels[i]]=list_of_users


for i in range(len(column_labels)):
    bicluster_num_profiles[column_labels[i]]+=1
    list_of_profiles = []
    list_of_profiles = bicluster_list_profiles[column_labels[i]]
    list_of_profiles.append(i)
    bicluster_list_profiles[column_labels[i]]=list_of_profiles

#print(str(row_labels))
#print(len(row_labels))
print('\n--------Number of users and profiles in each bicluster--------')
print('{:<15}\t{}\t{}'.format('Cluster','Users','Profiles'))
temp=0
for i in range(K):
    print('{:<15}\t{}\t{}'.format(i,bicluster_num_users[i],bicluster_num_profiles[i]))
print(sum(bicluster_num_users))
print(sum(bicluster_num_profiles))

f=open('cluster_num_users_biclustering','w')
for i in range(K):
    f.write(str(i))
    f.write('\t')
    f.write(str(bicluster_num_users[i]))
    f.write('\n')
f.close()

Prediction_kaggle = []
nonrated=0
z=0
for row in IDmap:
    z=z+1
    print(z)
    #print('Testing for 1st user and profile in test : ' + str(row))
    profile = row[1]

    #print('Bi Cluster for this user : ')
    user = row[0]
    #print(user)
    user_id = user_ids.index(user)
    #print(user_id)
    #print(labels)
    bicluster_index = row_labels[user_id]
    #print(bicluster_index)
    
    #print('Other user ids  in this cluster : ')
   # print(bicluster_num_users[bicluster_index])
   # print(len(bicluster_list_users[bicluster_index]))
    other_user_ids_in_same_cluster=bicluster_list_users[bicluster_index]
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

    #print('Bi Cluster for this profile : ')
    profile = row[1]
    #print(profile)
    profile_id = profile_ids.index(profile)
    #print(profile_id)
    #print(labels)
    bicluster_index_profile = column_labels[profile_id]
    #print(bicluster_index_profile)
    
    #print('Other profile ids in this cluster : ')
    #print(bicluster_num_profile[bicluster_index_profile])
    #print(len(bicluster_list_profile[bicluster_index_profile]))
    other_profile_ids_in_same_cluster=bicluster_list_profiles[bicluster_index_profile]
    #print(other_profile_ids_in_same_cluster)

    number_of_profiles_rated_by_user=0
    sum_total_rating_1=0
    for i in other_profile_ids_in_same_cluster:
        if user_profile_matrix[user_id][i] > 0:
            #print(i)
            #print('profile has been rated')
            #print(profile_id)
            #print(user_profile_matrix[user_id][i])
            number_of_profiles_rated_by_user+=1
            sum_total_rating_1+=user_profile_matrix[user_id][i]

    #print('Predicted Rating for this profile :')
    #print(sum_total_rating)
    if(number_of_users_who_rated_profile > 0):
        rating_predicted = sum_total_rating/number_of_users_who_rated_profile
        
    if(number_of_profiles_rated_by_user > 0):
        rating_predicted = (rating_predicted + (sum_total_rating_1/number_of_profiles_rated_by_user))/2
        
    if((number_of_profiles_rated_by_user == 0) & (number_of_users_who_rated_profile == 0)):
        print("NO PREVIOUS RATING")
        print(row)
        rating_predicted = np.mean(ratings[:,2])
        nonrated=nonrated+1 
    Prediction_kaggle = np.append(Prediction_kaggle,rating_predicted)    
        
Prediction_kaggle = Prediction_kaggle.T
Prediction_kaggle = np.reshape(Prediction_kaggle,(len(Prediction_kaggle),1))
ID = IDmap[:,2]
ID = np.reshape(ID,(len(ID),1))
final = np.hstack((ID,Prediction_kaggle))
print(nonrated)
h = 'ID, Prediction'
np.savetxt('biclustering.csv',final,delimiter = ',',header = h)
