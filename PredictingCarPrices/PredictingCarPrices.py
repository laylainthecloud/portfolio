import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset = pd.read_csv('Datasets/Automobile_data.csv')
print(dataset.head())

# mark all missing values (?) as NaN:
dataset.replace('?', np.nan, inplace=True)
print(dataset.head())

print(dataset.shape)

print(dataset.isnull().sum())

# drop all records with missing data and save it as a new dataframe:
dataset.dropna(inplace=True)
print(dataset.shape)

print(dataset.columns)

print(dataset.dtypes)

# let's first convert values in some of the columns to float (because they were recognized as 'object' initially):

dataset['normalized-losses'] = pd.to_numeric(dataset['normalized-losses'], downcast="float")
dataset['bore'] = pd.to_numeric(dataset['bore'], downcast="float")
dataset['stroke'] = pd.to_numeric(dataset['stroke'], downcast="float")
dataset['horsepower'] = pd.to_numeric(dataset['horsepower'], downcast="float")
dataset['peak-rpm'] = pd.to_numeric(dataset['peak-rpm'], downcast="float")
dataset['price'] = pd.to_numeric(dataset['price'], downcast="float")

print(dataset['make'].unique())

print(dataset['num-of-doors'].unique())

print(dataset['num-of-cylinders'].unique())

# now let's replace numerical data written in words with actual numbers:

dataset.replace({"num-of-doors":{"four": 4, "two": 2},
                "num-of-cylinders":{"three":3, "four": 4, "five": 5, "six": 6, "eight": 8}}, 
                inplace=True)
print(dataset.head())


# Let's look at all the columns with categorical data so we can decide how to encode them:

print(dataset['fuel-type'].unique())

print(dataset['aspiration'].unique())

print(dataset['body-style'].unique())

print(dataset['drive-wheels'].unique())

print(dataset['engine-location'].unique())

# as it appears that all cars in our dataset have front engines, we can drop this column:
dataset.drop(['engine-location'], axis=1, inplace=True)

print(dataset['engine-type'].unique())

print(dataset['fuel-system'].unique())

# replace categorical data with dummy variables
dataset = pd.get_dummies(dataset,
                         columns = ['make', 'fuel-type', 'aspiration','body-style', 'drive-wheels',\
                                    'engine-type', 'fuel-system'],
                         prefix = ['make', 'fuel-type', 'aspiration','body', 'drive','engine', 'fuel-system'])
print(dataset.head())


# Now that we've prepared our dataset, we can move on to creating our ML models and choosing the best one.

# separate our predictor and result variables:
X = dataset.loc[:, dataset.columns != 'price'].values
y = dataset['price'].values


# # Multiple Linear Regression model

# split the dataset into Training and Test sets:
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# train our model on the Training set:
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# predict Test set results:
y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# evaluate the model performance:
from sklearn.metrics import r2_score
print(r2_score(y_test, y_pred))


# # Polynomial Regression model

# split the dataset into Training and Test sets:
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# train our model on the Training set:
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

poly_reg = PolynomialFeatures(degree = 3)
X_poly = poly_reg.fit_transform(X_train)
regressor = LinearRegression()
regressor.fit(X_poly, y_train)

# predict Test set results:
y_pred = regressor.predict(poly_reg.transform(X_test))
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# evaluate the model performance:
from sklearn.metrics import r2_score
print(r2_score(y_test, y_pred))


# # Support Vector Regression (SVR) model

y1 = y.reshape(len(y),1)

# split the dataset into Training and Test sets:
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y1, test_size=0.2, random_state=0)

# feature scaling:
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X_train = sc_X.fit_transform(X_train)
y_train = sc_y.fit_transform(y_train)

# train our model on the Training set:
from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(X_train, y_train)

# predict Test set results:
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(X_test)))
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# evaluate the model performance:
from sklearn.metrics import r2_score
print(r2_score(y_test, y_pred))


# # Decision Tree Regression model

# split the dataset into Training and Test sets:
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# train our model on the Training set:
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X_train, y_train)

# predict Test set results:
y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# evaluate the model performance:
from sklearn.metrics import r2_score
print(r2_score(y_test, y_pred))


# # Random Forest Regression model

# split the dataset into Training and Test sets:
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# train our model on the Training set:
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X_train, y_train)

# predict Test set results:
y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# evaluate the model performance:
from sklearn.metrics import r2_score
print(r2_score(y_test, y_pred))


# # So the Random Forest Regression model takes the win!
