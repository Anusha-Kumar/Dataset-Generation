import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn import preprocessing
import random

# A dataset with two balanced classes and 5 features

X, Y = datasets.make_classification(n_samples=1000, n_features=5,n_classes=2,
                             shuffle=False, random_state=123, n_clusters_per_class=2, class_sep=5,
                             shift=[20,10,0,5,30])

print('A dataset with two balanced classes and 5 features')
print('Shape of X matrix: ',np.shape(X))
print('Shape of Y vector: ',np.shape(Y))

X = pd.DataFrame(X)
Y = pd.DataFrame(Y)

print('Description of input features:')
print(X.describe())
print('----------------------------------------------')

####################################################################################################
####################################################################################################

# A dataset with unbalanced classes and 5 features, with 3 features scaled by 2.
# Additional parameters: 
# Use the 'weights' parameter to assign the imbalance between the two classes

X, Y = datasets.make_classification(n_samples=1000, n_features=5,n_classes=2, weights=[0.7,0.3],
                             shuffle=False, random_state=123, n_clusters_per_class=2, class_sep=5,
                             shift=[20,10,0,5,30], scale=[1,2,2,1,2])

print('A dataset with unbalanced classes and 5 features, with 3 features scaled by 2')
print('Shape of X matrix: ',np.shape(X))
print('Shape of Y vector: ',np.shape(Y))

X = pd.DataFrame(X)
Y = pd.DataFrame(Y)

print('Description of input features:')
print(X.describe())
print('----------------------------------------------')

####################################################################################################
####################################################################################################

# A dataset with unbalanced classes and 4 out of 7 features are influential
# Additional parameters: 
# 'n_informative' is the number of influential features 
# 'n_redundant' is the number of features that are randomly given a linear combination of the informative features
# 'n_repeated' are the number of features that are randomly duplicated from other informative or redundant features.

X, Y = datasets.make_classification(n_samples=1000, n_features=7,n_informative=4, n_redundant=2, n_repeated=1,
                             n_classes=2, weights=[0.7,0.3],
                             shuffle=False, random_state=123, n_clusters_per_class=2, class_sep=5,
                             shift=[20,10,0,5,30,15,5])

print('A dataset with unbalanced classes and 4 out of 7 features are influential')
print('Shape of X matrix: ',np.shape(X))
print('Shape of Y vector: ',np.shape(Y))

X = pd.DataFrame(X)
Y = pd.DataFrame(Y)

print('Description of input features:')
print(X.describe())
print('----------------------------------------------')

####################################################################################################
####################################################################################################

# Introducing noise to a dataset
# Additional parameters:
# The parameter 'flip_y' gives the proportion of observations in which the label 'y' is randomly flipped. 
# This is meant to introduce some noise to the dataset.

X, Y = datasets.make_classification(n_samples=1000, n_features=7,n_informative=4, n_redundant=2, n_repeated=1,
                             n_classes=2, weights=[0.7,0.3], flip_y=0.02,
                             shuffle=False, random_state=123, n_clusters_per_class=2, class_sep=5,
                             shift=[20,10,0,5,30,15,5])

print('A dataset with unbalanced classes and 4 out of 7 features are influential, with added noise')
print('Shape of X matrix: ',np.shape(X))
print('Shape of Y vector: ',np.shape(Y))

X = pd.DataFrame(X)
Y = pd.DataFrame(Y)

print('Description of input features:')
print(X.describe())
print('----------------------------------------------')

####################################################################################################
####################################################################################################

#Output with imbalanced classes
print('Output Y (2 classes) : Imbalanced classes')
y_1 = np.where(Y==1)
print('No. of samples labeled Y=1 : ',np.shape(y_1)[1])
y_0 = np.where(Y==0)
print('No. of samples labeled Y=0 : ',np.shape(y_0)[1])
print('----------------------------------------------')

####################################################################################################
####################################################################################################

#Printing correlation matrix
print('Correlation matrix :')
print(X.corr())
print('----------------------------------------------')

####################################################################################################
####################################################################################################

# Converting features to categorical values

temp = np.array(X[2])
temp = temp.reshape(-1, 1)
est = preprocessing.KBinsDiscretizer(n_bins=2, encode='ordinal', strategy='kmeans').fit(temp)
est
X2_binned = est.transform(temp)
print('Categories after transforming X2: ',np.unique(X2_binned))
X.insert(2,'2_new',X2_binned) #Inserting transformation as a new column
print(X)
X = X.drop(columns=[2]) #Removing old column which was continuous 
print(X.columns)
print('----------------------------------------------')

####################################################################################################
####################################################################################################

#Creating missing data

X_missing = X.copy(deep=True)
row_ind = np.random.randint(0,1000,int(0.05*1000)) #Randomly choosing 5% rows for inserting missing data 
print(row_ind)
l = [0,1,2,3,4,5,6] #Column indices
for i in range(0,int(0.05*1000)):
    col_ind = np.random.choice(l) 
    X_missing.iloc[row_ind[i],col_ind] = np.nan

print('Number of missing values introduces in each column:')
print(X_missing.isnull().sum())
print('----------------------------------------------')
