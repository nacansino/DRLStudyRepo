# Importing the libraries
#shift+enter to execute blocks of selected code
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Part 1 - Data Preprocessing

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X1 = LabelEncoder() #for the geographical location
labelencoder_X2 = LabelEncoder() #for the gender
X[:, 1] = labelencoder_X1.fit_transform(X[:, 1])
X[:, 2] = labelencoder_X2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
#since one of the categories in our categorical feature 
#(originally the geographical location) is already represented as the
#zero of all the other categories. remove then the dummy variable
#A categorical feature with distinct elements n can be
#represented by (n-1) One-Hot encoded columns.
X = X[:,1:] #array selection

# Splitting the dataset into the Training set and Test set
#train_test_split was moved from sklearn.cross_validation to sklearn.model_selection
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
#compulsory for ANNs!!
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Making the ANN

#import the Keras library
import keras
from keras.models import Sequential #used to initialize our NN
from keras.layers import Dense #used to create the layers of our ANN
from keras.layers import Dropout #for dropout regularization

#Initializing the ANN
#2 ways: 1) defining the sequence of layers, or 2) defining a graph
#for now, use 1)
classifier = Sequential()

# Adding the input layer and the first hidden layer
# for us to have the probability information for each customer.
# For the hidden layers, use the rectifier function
# Press Ctrl+I for help
#a. create input layer and 1st hidden layer with the input layer having 11 nodes
#and the 1st hidden layer having 6 nodes
#note that by default, bias is set to true (check __init__ for more details)
classifier.add(Dense(units=6, activation = "relu",input_dim=11))
classifier.add(Dropout(rate=0.1)) #apply Dropout regularization to first layer
#use rate=0.1, increment by 0.1 if it doesn't solve the problem
#avoid using rate>0.5 (underfitting)

#add the 2nd hidden layer
classifier.add(Dense(units=6, activation = "relu"))
classifier.add(Dropout(rate=0.1)) #apply Dropout too here
# Use the sigmoid activation function for the output layer
#the output units depend on the number of distinct categories
#use Softmax activation fxn for more than one output units (similar to sigmoid)
classifier.add(Dense(units=1, activation = "sigmoid"))

#compile the whole ANN by applying stochastic gradient descent
#use a very efficient stochastic gradient descent algorithm, adam
#for binary outcome classification, use "binary_crossentropy" (research more on this)
classifier.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])

# Fit the ANN to the training set
#select batch size and epochs
classifier.fit(x = X_train, y = y_train, batch_size = 10, epochs=100)

# Fitting classifier to the Training set
# Create your classifier here

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#experiment with one data point
x_samp=np.asarray([0,0,600,1,40,3,60000,2,1,1,50000])
x_samp=sc.transform(x_samp.reshape(1,-1))

ysamp_pred = classifier.predict(x_samp)

# Part 4 - Evaluating, improving and tuning the ANN

#evaluating the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

#we now create a function that defines the
#architecture of the classifier.
#This will act as the argument for the KerasClassifier
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units=6, activation = "relu",input_dim=11))
    classifier.add(Dense(units=6, activation = "relu"))
    classifier.add(Dense(units=1, activation = "sigmoid"))
    classifier.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])
    return classifier

#create a new classifier
#that will act as our global classifier.
#this will be built using K-fold cross validation
#we will call this classifier_kfold
classifier_kfold = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 100)
#create variables that will contain the accuracies of each instances
#that used each folds for k-cross validation
#use cv = 10, and n?_jobs = -1 (run parallel computations)
accuracies = cross_val_score(estimator = classifier_kfold, X = X_train, y = y_train, cv = 10)

#improve ANN by using dropout regularization
#dropping one neuron randomly to avoid interdependencies
#implement this in our classifier

# Part 5 - Tuning the ANN
#tuning the ANN
#define HYPERPARAMETERS - number of epoch, batch size, optimizer, # of neurons, layers
#use grid search to find the best set of HYPERPARAMETERS 
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV #use GridSearchCV for model selection

#we now create a function that defines the
#architecture of the classifier.
#This will act as the argument for the KerasClassifier
def build_classifier_gridsearchCV(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units=6, activation = "relu",input_dim=11))
    classifier.add(Dense(units=6, activation = "relu"))
    classifier.add(Dense(units=1, activation = "sigmoid"))
    classifier.compile(optimizer = optimizer, loss = "binary_crossentropy", metrics = ["accuracy"])
    return classifier
#no indication of batch size and epochs; this will be determined in the process of GridSearchCV
classifier_kfold = KerasClassifier(build_fn = build_classifier_gridsearchCV)

#create dictionary for the hyperparameters
#IMPT: To tune any parameter in the architecture (units, optimizer, activation),
#pass the name as parameter in the build_fn then use that as a key in the dictionary below.
parameters = {'batch_size' : [25, 32],
              'epochs' : [100, 500],
              'optimizer' : ['adam', 'rmsprop']}
#create grid search object
grid_search = GridSearchCV(estimator = classifier_kfold,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)
#fit the ANN to the training set using a variety of hyperparameters
grid_search = grid_search.fit(X_train, y_train)
#extract the best hyperparameters and score (attribute of grid_search)
best_hyperparameters = grid_search.best_params_
best_accuracy = grid_search.best_score_