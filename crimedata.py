# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 14:18:32 2020

@author: Vimal PM
"""
#importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
#imorting the dataset using pd.read_csv()
crimedata=pd.read_csv("D://DATA SCIENCE//ASSIGNMENT//Assignment7//crimedata.csv")
#coulmns names
crimedata.columns
#Index(['Unnamed: 0', 'Murder', 'Assault', 'UrbanPop', 'Rape'], dtype='object')
#first five observations
crimedata.head()
      Unnamed: 0  Murder  Assault  UrbanPop  Rape
0     Alabama    13.2      236        58    21.2
1      Alaska    10.0      263        48    44.5
2     Arizona     8.1      294        80   31.0
3    Arkansas     8.8      190        50    19.5
4  California     9.0      276        91    40.6

#last five observation from my dataset
crimedata.tail()
        Unnamed: 0  Murder  Assault  UrbanPop  Rape
45       Virginia     8.5      156        63  20.7
46     Washington     4.0      145        73  26.2
47  West Virginia     5.7       81        39   9.3
48      Wisconsin     2.6       53        66  10.8
49        Wyoming     6.8      161        60  15.6

#Normalizing the dataset using normalizing function
def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return(x)
    
#normalizng the variables from my dataset using norm_func() then store it in a newdata set called df_norm
#here i'm taking only numerical parts from my variables    
df_norm=norm_func(crimedata.iloc[:,1:])

#Next i'm going for the EDA parts
#various visualizations
plt.hist(df_norm["Murder"])
plt.hist(df_norm["Assault"])
plt.hist(df_norm["UrbanPop"])
plt.hist(df_norm["Rape"])
plt.plot(df_norm.Murder,crimedata.Assault,"ro");plt.xlabel("Muder");plt.ylabel("Assault")
plt.plot(df_norm.UrbanPop,df_norm.Rape,"ro");plt.xlabel("UrbanPop");plt.ylabel("Rape")
#calculating the mean,median,mode,sd,variance,max value and min value using describe function
df_norm.describe()

          Murder    Assault   UrbanPop       Rape
count  50.000000  50.000000  50.000000  50.000000
mean    0.420964   0.430685   0.568475   0.360000
std     0.262380   0.285403   0.245335   0.242025
min     0.000000   0.000000   0.000000   0.000000
25%     0.197289   0.219178   0.381356   0.200904
50%     0.388554   0.390411   0.576271   0.330749
75%     0.629518   0.698630   0.775424   0.487726
max     1.000000   1.000000   1.000000   1.000000

#imporiting the linkage function of Hierarchical from clustering scipy module
from scipy.cluster.hierarchy import linkage
#for dendrogram 
import scipy.cluster.hierarchy as sch
#Next i'm calculating the Euclidean distance using single linkage function
L=linkage(df_norm,method="single",metric="Euclidean")
plt.figure(figsize=(15,5));plt.title("Hierarchical clustering dendogram");plt.xlabel("index");plt.ylabel("distance")
sch.dendrogram(
    L,
    leaf_rotation=0.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
plt.show()
#here we have 50  obervations,so root of (50/2)=3..So consider the cluster as 3 
#Next i'm going to use agglomerative clustering() for how many clusters that I need to see
from sklearn.cluster import AgglomerativeClustering
L_single= AgglomerativeClustering(n_clusters=3,linkage="single",affinity="euclidean").fit(df_norm)
#labels
L_single.labels_

array([0, 1, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0]
#from the labels I can see most of the datapoints are belongs to cluster'0'
cluster_labels=pd.Series(L_single.labels_)
cluster_labels.head()
0    0
1    1
2    0
3    0
4    0
cluster_labels.tail()
45    0
46    0
47    0
48    0
49    0
#next i would like to add my cluster_variable to the original dataset
crimedata['cluster']=cluster_labels
crimedata=crimedata.iloc[:,[5,0,1,2,3,4]]
crimedata.head()
    cluster  Unnamed: 0  Murder  Assault  UrbanPop  Rape
0        0     Alabama    13.2      236        58  21.2
1        1      Alaska    10.0      263        48  44.5
2        0     Arizona     8.1      294        80  31.0
3        0    Arkansas     8.8      190        50  19.5
4        0  California     9.0      276        91  40.6
crimedata.tail()
    cluster     Unnamed: 0  Murder  Assault  UrbanPop  Rape
45        0       Virginia     8.5      156        63  20.7
46        0     Washington     4.0      145        73  26.2
47        0  West Virginia     5.7       81        39   9.3
48        0      Wisconsin     2.6       53        66  10.8
49        0        Wyoming     6.8      161        60  15.6
#
#getting the aggregate mean of each cluster
crimedata.groupby(crimedata.cluster).mean()
             Murder     Assault   UrbanPop    Rape
cluster                                          
0         7.583333  165.416667  65.604167  20.525
1        10.000000  263.000000  48.000000  44.500
2        15.400000  335.000000  80.000000  31.900