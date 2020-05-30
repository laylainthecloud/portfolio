import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from yellowbrick.target import FeatureCorrelation

cars = pd.read_csv('Datasets/Automobile_data.csv')
print(cars.head())

# mark all missing values (?) as NaN:
cars.replace('?', np.nan, inplace=True)
print(cars.head())

# inspect the dataframe:
print(cars.columns)

print(cars.shape)

print(cars.isnull().sum())


# There is a significant number of records with missing values.

# so let's drop all such records and save it as a new dataframe:

cars.dropna(inplace=True)
print(cars.shape)

# look at summary statistics for this dataframe:
print(cars.describe())


# Interesting - there is no 'price' or 'horsepower' column (and some other ones we won't be needing right now).

# let's convert values in those 2 columns to float, so we can use them for our analysis:

cars['horsepower'] = pd.to_numeric(cars['horsepower'], downcast="float")
cars['price'] = pd.to_numeric(cars['price'], downcast="float")

# now let's have another look at summary statistics for this dataframe:
print(cars.describe())

# save this new dataframe as a .csv:
cars.to_csv('Datasets/cars_processed.csv', index=False)


# visualize price distribution (with the KDE curve):

plt.figure(figsize=(12,8))

sns.distplot(cars['price'], color='blue')

plt.title('Cars Data')

plt.show()


# We can see that most cars have prices in the range (approximately) $5K-$10K.


# let's look at that distribution in a bit more detail - visualize a rug plot together with the KDE curve
# and label price intervals in more detail:

plt.figure(figsize=(12,8))

sns.distplot(cars['price'], hist=False, rug=True, color='blue')

plt.xticks(np.arange(4000, 40000, 2000), rotation=60)

plt.title('Cars Data')

plt.show()


# We can now see (more precisely) that most of the individual data points (cars) lie between $6K and $8K.

# We can assume that horsepower and fuel consumption probably affect the price, so let's check those relationships:



# let's look at the relationship between horsepower and price:

plt.figure(figsize=(12,8))

sns.regplot(x='horsepower', y='price', data=cars)

plt.title('Prices Based on Horsepower')

plt.show()


# We can see a positive correlation between horsepower and price - as horsepower increases, so does the price.


# let's add another variable here - number of cylinders:

plt.figure(figsize=(12,8))

sns.scatterplot(x='horsepower', y='price', data=cars, hue='num-of-cylinders', s=120)

plt.title('Prices Based on Horsepower and Number of Cylinders')

plt.show()


# Here we can notice that most cars have 4 cylinders, rarely 3 or 8, and some have 5 or 6.
# Apparently, those with 5 cylinders tend to be a little pricier than those with 4 or 6.
# We can also notice that cars with 4 or fewer cylinders tend to have lower horsepower,
# those with 5 - between 100 and 140, while those with 6 cylinders range between 100 and 200 horsepower.



# now let's inspect how fuel consumption affects the price:

plt.figure(figsize=(12,8))

sns.scatterplot(x='highway-mpg', y='price', data=cars, hue='fuel-type', s=100)

plt.title('Prices Based on Fuel Consumption')

plt.show()


# We can see that there's a negative correlation between these 2 variables (price and fuel consumption),
# and also that there are far more cars that use gas than the ones that run on diesel



# let's inspect the relationship between prices and fuel consumption further:

plt.figure(figsize=(10,6))

sns.regplot(x='highway-mpg', y='price', data=cars)

plt.title('Price vs. Fuel Consumption')

plt.show()



# now let's go back to what we saw in the 'Prices Based on Fuel Consumption' scatter plot
# when it comes to distribution of cars that run on gas and diesel:

plt.figure(figsize=(6,6))

gas = round(cars['fuel-type'].value_counts()[0]/len(cars)*100, 2)
diesel = round(cars['fuel-type'].value_counts()[1]/len(cars)*100, 2)
sizes = [gas, diesel]
labels = 'gas', 'diesel'
explode = (0, 0.1)

plt.pie(sizes, labels=labels, explode=explode, autopct='%1.1f%%', shadow=True, startangle=90)
plt.axis('equal')

plt.show()



# let's look at this distribution together with prices:

plt.figure(figsize=(15,8))

fg = sns.FacetGrid(cars, col='fuel-type', height=7, aspect=1)

fg.map(plt.hist, 'price', color='green')

plt.show()


# this doesn't tell us much when it comes to price, there is no apparent trend,
# considering we saw earlier that most cars are in the $6-8K price range anyway, 
# but let's check if fuel consumption vs. price correlation changes with fuel type:

fg = sns.FacetGrid(cars, col='fuel-type', height=7, aspect=1)

fg.map(sns.lineplot, 'highway-mpg', 'price')

plt.show()


# So the answer is no - fuel type does not affect fuel consumption vs. price relationship, 
# we can see that there is a negative correlation in both cases.



# let's check whether engine size has any effect on the price:

plt.figure(figsize=(10,6))

sns.regplot(x='engine-size', y='price', data=cars)

plt.title('Price vs. Engine Size')

plt.show()


# The above regression plot shows us that as engine size grows - so does the price.


# we could also calculate Pearson correlation coefficients for the relationships inspected so far
# to double-check the findings:

target = cars['price']
features = cars[['highway-mpg', 'city-mpg', 'horsepower', 'engine-size']]
feature_names = list(features.columns)

# instantiate the FeatureCorrelation Python object:
visualizer = FeatureCorrelation(labels = feature_names, 
                                title='Features Correlation With Price', 
                                color=['r','r','green','green'])
# calculate correlation coefficients:
visualizer.fit(features, target)
# display them:
visualizer.poof()


# The above visualization confirms the previous findings:
# there is a strong positive correlation between engine size and price, and between horsepower and price,
# and there is a negative correlation between fuel consumption and price.
