
#MiniBatchKMeans clustering + list of models
"""
here we are using MiniBatchKMeans clustering as it is better than KMeans
"""
# generalise model

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from sklearn.cluster import MiniBatchKMeans

# Importing the dataset
dataset = pd.read_csv('combine.csv',low_memory=False)
#X = dataset.iloc[:, 1:].values
#y = dataset.iloc[:, 0].values

#applying clustering on the preprocessed data 
x_cl=dataset.iloc[:40000,3:].values
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
x_cl[:,1:] = sc_X.fit_transform(x_cl[:,1:])

#feature selection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
kbest=SelectKBest(score_func=f_regression,k=146)
x_cl1=kbest.fit_transform(x_cl[:,1:],x_cl[:,0])
features=kbest.get_support()
# Using the elbow method to find the optimal number of clusters
wcss = []
for i in range(1, 14):
    kmeans = MiniBatchKMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(x_cl1)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 14), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# Fitting K-Means to the dataset
n_cluster=8
kmeans = MiniBatchKMeans(n_clusters = n_cluster, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(x_cl1)

#Seggregating  the clusters into n_cluster dataframes
dic={}
for i in range(n_cluster):    
    l=[]
    for index in range(len(y_kmeans)):
        if(y_kmeans[index]==i):
            l.append(index)
    dic[i]=l
        
#Now our new datasets are:
dataset_dic={}
for i in range(n_cluster):
    dataset_dic[i]=x_cl1[dic[i],:]

"""
specifying the list of models corresponding to each cluster

"""
model_list=[]
model_dic={}
print "\nOrder of the models:\nBayesianRidge\tHuber\tSVR\tSGD\tDecisionTree\tLinear\tRandomForest\n"
print "\nNote: the no of clusters :\t"+str(n_cluster)
model_list=[int(w) for w in raw_input("please enter a list of numbers starting from 1 to 7 :\n").split()]

from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import HuberRegressor 
from sklearn.svm import SVR
from sklearn.linear_model import SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

model=[BayesianRidge(),HuberRegressor(alpha = .5),SVR(kernel = 'linear'),SGDRegressor(loss="huber"),DecisionTreeRegressor(random_state=0),LinearRegression(),RandomForestRegressor(n_estimators = 1000, random_state = 0,n_jobs=1)]
for i in range(n_cluster):
    model_dic[i] = model[model_list[i]-1]

"""
now from above we can customise the models in the cluster that we want to run
and if we want to take same model for each cluster we can set the same index in model list 
"""


n_value={}
from sklearn.decomposition import PCA
def cluster(i):    
    # Splitting the dataset of 0th cluster into the Training set and Test set
    from sklearn.cross_validation import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(dataset_dic[i][:,:],x_cl[dic[i],0], test_size = 0.30, random_state = 0)
    from sklearn.decomposition import PCA
    from sklearn.tree import DecisionTreeRegressor
    best_accuracy=0
    best_component=0
    iterations=50
    count=0
#    best_pca=PCA()
    best_regressor=DecisionTreeRegressor()
    while(count<=iterations):
        count=count+1
        rows,column=X_train.shape
        n_value[i]=random.randint(100,column)
        pca=PCA(n_components=n_value[i])
        X_train1=pca.fit_transform(X_train,y_train)
        X_test1=pca.transform(X_test)
#        explained_variance=pca.explained_variance_ratio_
        # Fitting  Regression model to the dataset
        model_dic[i]
        model_dic[i].fit(X_train1, y_train)
        # Predicting a new result
        err_allowed=0.1
        y_pred = model_dic[i].predict(X_test1)
        accuracy=float(sum(abs(y_test-y_pred)<=err_allowed))/len(y_test)*100
        if(accuracy>=best_accuracy):
            best_accuracy=accuracy
            best_component=n_value[i]
            best_X=X_train1
            best_y=y_train
            best_pca=pca
            best_regressor=model_dic[i]
    return [best_regressor,best_accuracy,best_component,best_X,best_y,best_pca]

#cluster dictionary
clus_dic={}
for i in range(n_cluster):
    clus_dic[i]=cluster(i)


#checking on the unseen dataset

#1.preprocessing
unseen_x=dataset.iloc[50000:60000:200, 3:].values
#feature scaling
unseen_x[:,1:]=sc_X.fit_transform(unseen_x[:,1:])
#feature selection
unseen1_x=unseen_x
unseen1_x=unseen1_x[:,np.append(np.array([True]),features)]

"""
now our new unseen dataset is unseen1_x with feature scaling and selection
"""
# Predicting unseen data

"""
here for clustering we will use the same object "kmeans"
"""

#2.clustring prediction
y_kmeans_unseen = kmeans.fit_predict(unseen1_x[:,1:])

#Seggregating  the clusters into n_cluster dataframes
dic_unseen={}
for i in range(n_cluster):    
    l=[]
    for index in range(len(y_kmeans_unseen)):
        if(y_kmeans_unseen[index]==i):
            l.append(index)
    dic_unseen[i]=l
    
#Now our new datasets are:
dataset_dic_unseen={}
for i in range(n_cluster):
    dataset_dic_unseen[i]=unseen1_x[dic_unseen[i],:]
    
#predicting the unseen dataset according to each cluster
result={}
data_x={}
data_y={}
accuracy_unseen={}
for i in range(n_cluster):
    if( (dataset_dic_unseen[0].shape)[0]!=0):
        reg=clus_dic[i][0]
        ext_component=clus_dic[i][2]
        data=dataset_dic_unseen[i]
        data_x[i]=data[:,1:]
        data_y[i]=data[:,0]
#        pca_unseen=PCA(n_components=ext_component)
        pca_unseen=clus_dic[i][5]
        data_x1=pca_unseen.fit_transform(clus_dic[i][3],clus_dic[i][4])
#        data_x1=pca_unseen.fit_transform(data_x[i],data_y[i])
        data_x[i]=pca_unseen.transform(data_x[i])
        result[i]=reg.predict(data_x[i])
        err_allowed=0.1
        accuracy_unseen[i]=float(sum(abs(data_y[i]-result[i])<=err_allowed))/len(data_y[i])*100
    else:
        accuracy_unseen[i]='NULL'
        continue
    


        
