#!/usr/bin/env python
# coding: utf-8

# # TASK 2 : Prediction Using Unsupervised ML

# # NAME : Ritik Manro

# ### From the given ‘Iris’ dataset,we've to predict the optimum number of clusters and represent it visually.

# ### Importing Libraries

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, accuracy_score
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Reading Data Set

# In[2]:


df = pd.read_csv('E:\\TSF Internship\\Iris.csv')
df.head()


# ### Checking shape of Data

# In[3]:


df.shape


# In[4]:


df.info()


# ### Data Types

# In[5]:


data_types = pd.DataFrame(df.dtypes, columns = ['Types'])
data_types


# ### Missing Values

# In[6]:


missing_values = pd.DataFrame(df.isna().sum(), columns = ['Missing Values'])
missing_values


# ### Numerical Analysis

# In[7]:


df.describe()


# ### Number of Species

# In[8]:


df['Species'].value_counts()


# ### Converting Species into Number

# In[9]:


df['Species'] = df['Species'].map({'Iris-versicolor' : 0, 'Iris-setosa' : 1, 'Iris-virginica' : 2}).astype(int)
df


# ### Dropping Id

# In[10]:


df = df.drop('Id', axis = 1)
df.head()


# ### Finding Optimum Number of Clusters

# In[11]:


X = df.iloc[:, [0, 1, 2, 3]].values
wcss = []
plt.figure(figsize = (10,12))
from matplotlib import style
style.use("ggplot")
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, random_state = 17)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss, color = 'm',linewidth = 5)
plt.title('ELBOW METHOD', fontsize = 20)
plt.xlabel('Number of clusters', fontsize = 15)
plt.ylabel('WCSS', fontsize = 15)
plt.grid(color='k', linestyle='-', linewidth=2)
plt.show()


#    ### So the optimum number of clusters is 3.

# ### Clustering

# In[12]:


kmc = KMeans(n_clusters = 3)
kmc.fit_predict(X)
df['Cluster'] = pd.Series(kmc.labels_)
df.head()


# ### Checking with Confusion Matrix

# In[13]:


confusion_matrix(df['Species'], df['Cluster'])


#         48 Iris-versicolor are correctly predicted and 2 are incorrectly predicted as Iris-virginica.
#         All 50 Iris-setosa are correctly predicted.
#         36 Iris-virginica are correctly predicted and 14 are incorrectly predicted as Iris-versicolor.

# ### Accuracy

# In[14]:


accuracy_score(df['Species'], df['Cluster'])


# ### Visual Representation

# In[15]:


plt.figure(figsize=(15,10))

from matplotlib import style
style.use("ggplot")
plt.scatter(X[df['Cluster'] == 0, 0], X[df['Cluster'] == 0, 1], c = 'm' ,label = 'Iris-versicolour')
plt.scatter(X[df['Cluster'] == 1, 0], X[df['Cluster'] == 1, 1], c = 'c', label = 'Iris-setosa')
plt.scatter(X[df['Cluster'] == 2, 0], X[df['Cluster'] == 2, 1], c = 'b', label = 'Iris-virginica')
plt.scatter(kmc.cluster_centers_[:,0], kmc.cluster_centers_[:,1], s = 100, c = 'purple', label = 'Centroids')
plt.grid(color='k', linestyle='--', linewidth=2)
plt.legend()
plt.show()


# In[ ]:




