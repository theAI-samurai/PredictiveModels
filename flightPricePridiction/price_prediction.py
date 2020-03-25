#Flight Preice Prediction - United Airlines

import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import matplotlib.pyplot as plt  
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import math
import matplotlib.pyplot as plt  
import statsmodels.api as sm
import numpy
from scipy import stats
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras import backend as K


fares = pd.read_csv(r'D:\myProject\predictiveModelling\training\training\train_fares.csv')
fare_test = pd.read_csv(r'D:\myProject\predictiveModelling\test\test\test_fares_data.csv')


data =['L1','L2','L3']
df_fares = fares[fares.carrier.isin(data)].reset_index(drop=True)
del[fares]

# this function extracts the month-year-date
def month_extraction(df): 
    dep_month = []
    obser_month = []
    for i in range(len(df)):
        if '-' in df.loc[i,'flt_departure_dt']:
            dep_month_val = ((df.loc[i,'flt_departure_dt']).replace('-',''))
            obser_month_val = ((df.loc[i,'observation_date']).replace('-',''))
        else:
            dep_month_val = ((df.loc[i,'flt_departure_dt']).replace('/',''))
            obser_month_val = ((df.loc[i,'observation_date']).replace('/',''))
        dep_month.append(dep_month_val)      
        obser_month.append(obser_month_val)
    df['month_flt_depar'] = dep_month
    df['month_observation'] = obser_month


#month_extraction(df_fares)
df_fare_test= fare_test[fare_test.carrier.isin(data)].reset_index(drop=True)
del [fare_test]
#month_extraction(df_fare_test)

# droping the key column seems unnecessary
Y = df_fares['total_fare'].tolist()
df_fares.drop(['Unnamed: 0','total_fare'], inplace=True, axis =1)
df_fare_test.drop(['Unnamed: 0'], inplace=True, axis =1)


#validation dataset creation
X_train, X_validate, Y_train, Y_validate = train_test_split(df_fares, Y, test_size = .20, random_state = 40)

# encoding the data
le = preprocessing.LabelEncoder()
X_train_ = X_train.apply(le.fit_transform)
X_validate_ = X_validate.apply(le.fit_transform)
fare_test_ =df_fare_test.apply(le.fit_transform)


#*******************************************************************

"""
Chi - square test between different features of 
p-value extraction for 2 variables

"""
def chi_test(df):
    features = list(df)
    for ele in features:
        for ele2 in features:
            if ele != ele2:
                print(str(ele) + ' vs '+str(ele2))
                cross_table = pd.crosstab(df[ele], df[ele2])
                chi_value = stats.chi2_contingency(observed= cross_table)
                print('P_value :', chi_value[1])
                print("\n")

chi_test(X_train_)
#***********************************************************************

#***********************************************************************
'''
ANOVA Test for relation between Catagoricala nd continous variable
'''
def anova_test(df):
    features = list(df)
    for ele in features:
        print(str(ele)+' vs' + 'Total_fare')
        temp = stats.f_oneway(X_train_[ele], Y_train)
        print(temp)
        print('\n')
        
anova_test(X_train_)
#*************************************************************************
#******************************* Linear Regression Model *********************
regressorLS = LinearRegression()  # Regressor
regressorLS.fit(X_train_, Y_train) #training the algorithm

# Prediction regressor on Validation data
y_pred_validation = regressorLS.predict(X_validate_)

# Mean sqaured error in validation set
error = math.sqrt(mean_squared_error(Y_validate, y_pred_validation))
print("\nRoot mean square error on validation set is  :", error)
print("\n")

# Predicting regressor on Test data
df_fare_test['predicted_LinearRegression'] = regressorLS.predict(fare_test_)

#*****************************************************************************

#******************************* Random Forest Regressor Model ***************
# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 10, random_state = 42)
# Train the model on training data
rf.fit(X_train_, Y_train) # Random forest regressor 
y_pred_validation_RF = rf.predict(X_validate_)

error_ = math.sqrt(mean_squared_error(Y_validate, y_pred_validation_RF))
print("\nRoot mean square error on validation set is  :", error_)
print("\n")
# Predicting regressor on Test data
df_fare_test['predicted_RandomForest'] = rf.predict(fare_test_)
#*****************************************************************************

#******************************* SVM Regressor *****************************
'''
regressor = SVR(kernel='poly', epsilon=.1, gamma='auto',degree=3)
regressor.fit(X_train_, Y_train)
y_pred_validation_svm = regressor.predict(X_validate_)
# Mean sqaured error in validation set
error = math.sqrt(mean_squared_error(Y_validate, y_pred_validation_svm))
print("\nRoot mean square error on validation set is  :", error)
print("\n")
# Predicting regressor on Test data
df_fare_test['predicted_svm'] = regressor.predict(fare_test_)
'''
#***************************************************************************

#******************************* Neural Network Model ***********************

def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true))) 

NN_model = Sequential()

# The Input Layer :
NN_model.add(Dense(128, kernel_initializer='normal',input_dim = X_train_.shape[1], activation='relu'))

# The Hidden Layers :
NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))

# The Output Layer :
NN_model.add(Dense(1, kernel_initializer='normal',activation='linear'))

# Compile the network :
#NN_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
NN_model.compile(loss=root_mean_squared_error, optimizer='adam', metrics=['accuracy'])
NN_model.summary()

NN_model.fit(X_train_, Y_train, epochs=150, batch_size=10240, validation_split = 0.2)
predictions_NN = NN_model.predict(X_validate_)

error__ = math.sqrt(mean_squared_error(Y_validate, predictions_NN))
print("\nRoot mean square error on validation set is  :", error__)
print("\n")
# Predicting regressor on Test data
df_fare_test['predicted_NNt'] = NN_model.predict(fare_test_)
#****************************************************************************

df_fare_test.to_csv('D:/myProject/predictiveModelling/test/test_fares_data_ankit_mishra.csv')

# statistical model description
X_train_ = sm.add_constant(X_train_) # adding a constant
model = sm.OLS(Y_train, X_train_).fit()
predictions = model.predict(X_train_) 
print_model = model.summary()
print("\n",print_model)

print('----------task completed---------')


