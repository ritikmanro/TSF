#!/usr/bin/env python
# coding: utf-8

# # Name - Ritik Manro

# # Task1- Prediction using Supervised ML

# We have to predict the percentage of marks that a student is expected to score based on the number of hours they studied.
# This is a simple linear regression task as it involves just two variables.

# In[1]:


# Importing the libraries. 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# Reading data.
link="http://bit.ly/w-data"
df = pd.read_csv(link)
print("Data loaded successfully.")
df.head()


# In[3]:


# Exploring the dataset.
df.shape


# In[4]:


df.describe()


# In[5]:


df.info()


# In[6]:


df.isnull()


# In[7]:


df.isnull().sum()


# # Data visualisation

# In[8]:


from matplotlib import style
plt.figure(figsize = (16,9))
style.use("ggplot")

x = 'Hours'
y = 'Scores'
df.plot(x,y,style = 's',marker = 'o',linewidth = 5,markersize = 10)
plt.title("Hours vs Percentage", fontsize = 15)
plt.xlabel("Hours Studied", fontsize = 10)
plt.ylabel("Percentage Score", fontsize =10)
plt.grid(color='k', linestyle='-', linewidth=2)
plt.show()


# It is clear from the graph that there is a positive linear relation between the number of hours studied and 
# percentage of score.
# 
# 

# In[9]:


import seaborn as sns
sns.lineplot(x = 'Hours', y = 'Scores', data = df)
plt.show()


# In[10]:


sns.heatmap(df.corr())
plt.show()


# # Data Preparing

# In[11]:


x = df['Hours']
y = df['Scores']


# In[12]:


x.head()


# In[13]:


y.head()


# In[14]:


x = df.iloc[:,:-1].values
y = df.iloc[:,1].values
x
y


# # Spliting Data

# In[15]:


from sklearn.model_selection import train_test_split


# In[16]:


x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.2, random_state=0)


# In[17]:


x_train.shape


# In[18]:


x_test.shape


# In[19]:


y_train.shape


# In[20]:


y_test.shape


# # Model Training Using Linear Regression

# In[21]:


from sklearn.linear_model import LinearRegression # for training Machine learning model
from sklearn.metrics import accuracy_score # for checking the accuracy of model

model = LinearRegression()


# In[22]:


model.fit(x_train, y_train)


# # Accuracy 

# In[23]:


model.score(x_train,y_train)


# In[24]:


y_pred=model.predict(x_test)


# In[25]:


#visualization
plt.figure(figsize=(10,6))
plt.scatter(x_test,y_test,color="m",s=75,label="Actual Score")
plt.scatter(x_test,y_pred,color="b",s=75,label="Predicted Score")
plt.plot(x_test,y_pred,color="k",label="Line of Best fit")
plt.title("Number of hours vs Percentage of marks(Actual vs predicted)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.legend(loc = 4)
plt.show()


# In[26]:


# Making predictions.

print(x_test)
y_pred= model.predict(x_test)


# In[27]:


error = pd.DataFrame({"Actual":y_test,"Predicted":y_pred,"Absolute Error":abs(y_test-y_pred)})
error


# What will be the predicted score if a student studies for 9.25 hrs/ day?

# In[28]:


hours =[[9.25]]
Predicted_score = model.predict(hours)
print('Predicted_Score for 9.25 Hours = ',Predicted_score)


# # Hence we conclude that the  predicted score of a student for studying 9.25 hours per day is 93.69%

# In[ ]:




