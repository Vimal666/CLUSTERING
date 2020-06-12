# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 22:55:25 2020

@author: Vimal PM
"""
#importing neccessary libraries

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn.cluster import	KMeans
from scipy.spatial.distance import cdist 

#imporing the dataset using pd.read_csv()
Airlines=pd.read_csv("D:\DATA SCIENCE\ASSIGNMENT\Assignment7\Airlines.csv")
#coulmns names
Airlines.columns
Index(['ID', 'Balance', 'Qual_miles', 'cc1_miles', 'cc2_miles', 'cc3_miles',
       'Bonus_miles', 'Bonus_trans', 'Flight_miles_12mo', 'Flight_trans_12',
       'Days_since_enroll', 'Award']

#Normalizing the datasets using normalizing equation to make our data unitless and scale free

def norm_func(i):
    x=(i-i.min()) / (i.max()-i.min())
    return (x)
# Normalized data frame (considering the numerical part of data)
#now i'm going to apply this normalizing function to make values looks normalized
df_norm=norm_func(Airlines.iloc[:,:])
#Next I'm going to calculate the mean,median,mode,etc... using describe function
df_norm.describe()
               ID      Balance  ...  Days_since_enroll       Award
count  3999.000000  3999.000000  ...        3999.000000  3999.000000
mean      0.500950     0.043172  ...           0.496330     0.370343
std       0.288747     0.059112  ...           0.248991     0.482957
min       0.000000     0.000000  ...           0.000000     0.000000
25%       0.251119     0.010868  ...           0.280685     0.000000
50%       0.501244     0.025279  ...           0.493610     0.000000
75%       0.751119     0.054201  ...           0.697914     1.000000
max       1.000000     1.000000  ...           1.000000     1.000000
#Next I'm going for the visualizations using different methods
plt.plot(Airlines.ID,Airlines.Balance,"ro");plt.xlabel("ID");plt.ylabel("Balance")
plt.plot(Airlines.Qual_miles,Airlines.cc1_miles,"ro");plt.xlabel("Qual_miles");plt.ylabel("cc1_miles")
plt.plot(Airlines.cc2_miles,Airlines.cc3_miles,"bo");plt.xlabel("cc2_miles");plt.ylabel("cc3_miles")
plt.plot(Airlines.Bonus_trans,Airlines.Bonus_miles,"go");plt.xlabel("Bonus_trans");plt.ylabel("Bonus_miles")
plt.plot(Airlines.Flight_miles_12mo,Airlines.Flight_trans_12,"go");plt.xlabel("Flight_miles_12mo");plt.ylabel("Flight_trans_12")
plt.plot(Airlines.Days_since_enroll,Airlines.Award,"go");plt.xlabel("Days_since_enroll");plt.ylabel("Award")

#first five obsevation of my normalized dataframe
df_norm.head()
         ID   Balance  Qual_miles  ...  Flight_trans_12  Days_since_enroll  Award
0  0.000000  0.016508         0.0  ...         0.000000           0.843742    0.0
1  0.000249  0.011288         0.0  ...         0.000000           0.839884    0.0
2  0.000498  0.024257         0.0  ...         0.000000           0.847842    0.0
3  0.000746  0.008667         0.0  ...         0.000000           0.837955    0.0
4  0.000995  0.057338         0.0  ...         0.075472           0.835905    1.0

#First I'm going to perform KMEANS clustering

k=list(range(2,15))#here i'm defining my clusters range randomly from 2 to 15

#Next I need to identify the total sum of square using TWSS[] 
TWSS=[]
for i in k:
    kmeans=KMeans(n_clusters=i)
    kmeans.fit(df_norm)
    WSS=[]#With in sum of squares
    for j in range(i):
         WSS.append(sum(cdist(df_norm.iloc[kmeans.labels_==j,:],kmeans.cluster_centers_[j].reshape(1,df_norm.shape[1]),"euclidean")))
    TWSS.append(sum(WSS))
#Here I have to deal with 4000 observation so I can't sellect my clusters number
#Therefor I'm going for screeplot to identify the elbo point
plt.plot(k,TWSS,"ro-");plt.xlabel("No_of_clusters");plt.ylabel("total_within_ss");plt.xticks(k)    
#From the graph I can see the Elbo point lying on 10th datapoint.
#So I'm going to choose my cluster number as 10
model=KMeans(n_clusters=10)
model.fit(df_norm)
#KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
      # n_clusters=10, n_init=10, n_jobs=None, precompute_distances='auto',
     #  random_state=None, tol=0.0001, verbose=0)
#Getting the labels of clusters     
model.labels_     
#array([8, 8, 8, ..., 9, 4, 4])
md=pd.Series(model.labels_)
md.head()#first five label of cluster
0    8
1    8
2    8
3    8
4    7
md.tail()#last five label of cluster
3994    0
3995    0
3996    9
3997    4
3998    4
#adding my labels of clusters to my original dataset
Airlines["clust"]=md
Airlines=Airlines.iloc[:,[12,0,1,2,3,4,5,6,7,8,9,10,11]]
Airlines.head()

     clust  ID  Balance  ...  Flight_trans_12  Days_since_enroll  Award
0      8   1    28143  ...                0               7000      0
1      8   2    19244  ...                0               6968      0
2      8   3    41354  ...                0               7034      0
3      8   4    14776  ...                0               6952      0
4      7   5    97752  ...                4               6935      1
#
#getting the aggregate mean of each cluster
Airlines.iloc[:,:].groupby(Airlines.clust).mean()
                ID        Balance  ...  Days_since_enroll  Award
clust                              ...                          
0      2822.132394   64350.585915  ...        2766.659155    1.0
1      1887.896321   43653.541806  ...        4238.285953    0.0
2       898.657459  190592.662983  ...        6149.292818    1.0
3       890.488449  135895.914191  ...        6088.016502    0.0
4      3359.668187   32297.475485  ...        1753.137970    0.0
5       880.657233  104939.188679  ...        6094.006289    1.0
6      2754.247619  101369.158730  ...        2814.650794    0.0
7       875.122977   93671.262136  ...        6113.135922    1.0
8       609.983529   54255.738824  ...        6628.520000    0.0
9      2736.625786   75719.361635  ...        2917.226415    1.0

#Next I'm going to perform Hierarchical clustering

from scipy.cluster.hierarchy import linkage#here I'm importing the linkage function from hierarchy of cluster from scipy module
#for seeing dendrogram i'm going to hierarchy as sch
import scipy.cluster.hierarchy as sch
#Here I'm going to calculate the Euclidean distance using complete linkage function
C=linkage(df_norm,method="complete",metric="Euclidean")
plt.figure(figsize=(15,5));plt.title("hierarchical clustering dendogram");plt.xlabel("index");plt.ylabel("distance")
sch.dendrogram(
        C,
        leaf_rotation=0.,
        leaf_font_size=8.,
)
plt.show()
#Before I was calculated cluster number as 10 from the screeplot,Here also I'm going to use this same number of clusters as 10
#Impoting the agglomerative clustering for how many clusters that we need to see or cut
from sklearn.cluster import AgglomerativeClustering
C_linkage=AgglomerativeClustering(n_clusters=10,linkage="complete",affinity="euclidean").fit(df_norm)
#labels of clusters
C_linkage.labels_
#array([2, 2, 2, ..., 0, 1, 1],
cluster_labels=pd.Series(C_linkage.labels_)#here transforming the series of labels to a new a dataset called "cluster_labels"
#Next I would like to add these variable to my original dataset
Airlines["clust"]=cluster_labels
Airlines=Airlines.iloc[:,[12,0,1,2,3,4,5,6,7,8,9,10,11]]
Airlines.head()
      clust  ID  Balance  ...  Flight_trans_12  Days_since_enroll  Award
0      2   1    28143  ...                0               7000      0
1      2   2    19244  ...                0               6968      0
2      2   3    41354  ...                0               7034      0
3      2   4    14776  ...                0               6952      0
4      4   5    97752  ...                4               6935      1
Airlines.tail()
        clust    ID  Balance  ...  Flight_trans_12  Days_since_enroll  Award
3994      0  4017    18476  ...                1               1403      1
3995      0  4018    64385  ...                0               1395      1
3996      0  4019    73597  ...                0               1402      1
3997      1  4020    54899  ...                1               1401      0
3998      1  4021     3016  ...                0               1398      0

#getting the aggregate mean of each cluster
Airlines.groupby(Airlines.clust).mean()

                ID        Balance  ...  Days_since_enroll  Award
clust                              ...                          
0      2716.744817   65604.537480  ...        2943.051037    1.0
1      2902.386220   46942.534766  ...        2528.217446    0.0
2       900.652031   82596.805708  ...        6072.614709    0.0
3      2398.840000   43494.400000  ...        3544.400000    0.0
4      1198.029907  120814.998131  ...        5557.158879    1.0
5      3128.000000  131999.500000  ...        2200.250000    1.0
6       676.930796  119201.719723  ...        6494.588235    1.0
7      1477.000000   73699.142857  ...        4897.857143    1.0
8      1401.400000   85120.900000  ...        5626.300000    1.0
9       931.111111  180122.444444  ...        6117.666667    1.0

