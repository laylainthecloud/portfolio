import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('Datasets/births_by_age_of_mother_1960_2013.csv')
print(data.head(10))

# first remove all records with age of mother "unknown":
data = data[data.age != 'UNKNOWN']
print(data.head(10))

# save the age groups for later use:
age_groups = data['age'].unique()
print(age_groups)


# line plots for all years and all age groups:

plt.figure(figsize=(12,8))

sns.lineplot(x='year', y='count', hue='age', data=data, legend='full')

plt.title('Births in CA by Year and Age of Mother')

plt.show()


# let's say we want to see births for mothers of ages 20-29 and 30-39 as percentages of total births by year
# first, pivot the dataframe so 1 year = 1 row:
data_by_year = pd.pivot_table(data, values="count", index="year", columns=data['age'], aggfunc=np.sum)
print(data_by_year.head(10))

# add a total column:
data_by_year['total'] = data_by_year.sum(axis=1)
print(data_by_year.head())

# sum the data we are interested in inspecting:
data_by_year['20-29'] = data_by_year['20-24'] + data_by_year['25-29']
data_by_year['30-39'] = data_by_year['30-34'] + data_by_year['35-39']
print(data_by_year.head())

# remove columns we don't need anymore: 
data_by_year.drop(columns=age_groups, axis=1, inplace=True)
print(data_by_year.head())

# calculate the percentages:
for column in ['20-29', '30-39']:
    data_by_year[column] = round((data_by_year[column]/data_by_year['total'])*100, 2)
print(data_by_year.head())


# now we can visualize those percentages over the years:

fig, ax1 = plt.subplots()

ax1.set_xlabel('year')
ax1.set_ylabel('percentage of total', color='blue')
ax1.plot('20-29', data=data_by_year, color='blue', linewidth=2, label="20-29")
ax1.plot('30-39', data=data_by_year, color='green', linewidth=2, label="30-39")
ax1.tick_params(axis='y', labelcolor='blue')
ax1.legend()

ax2 = ax1.twinx()

ax2.set_ylabel('total', color='orange')
ax2.plot('total', data=data_by_year, color='orange', linewidth=2, linestyle='dashed', label="total")
ax2.tick_params(axis='y', labelcolor='orange')

fig.tight_layout()

plt.title('Births in CA by Year by Mothers of Ages 20-39')

plt.show()


# if we want to look at more detailed visualizations, let's first divide the original dataset into "chunks":

data1960to1969 = data.iloc[:90,:]
data1970to1979 = data.iloc[90:180,:]
data1980to1989 = data.iloc[180:270,:]
data1990to1999 = data.iloc[270:360,:]
data2000to2009 = data.iloc[360:450,:]
data2010to2013 = data.iloc[450:,:]


# now we can visualize those "chunks" separately
# for example, let's say we're interested in the period between 1990 and 1999:

plt.figure(figsize=(12,8))

sns.catplot(x='year', y='count', hue='age', palette='bright',
            data=data1990to1999, kind='bar', height=6, aspect=2)

plt.title('Births in CA by Year and Age of Mother 1990-1999')

plt.show()