from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd
import matplotlib as plt
import tensorflow as tf

# x = [1, 2, 2.5, 3, 4]
# y = [1, 4, 7, 9, 15]
# plt.plot(x, y, 'ro')
# plt.axis([0, 6, 0, 20])
# plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)))
# plt.show()

# Load dataset.
dftrain = pd.read_csv('train.csv')  # training data
dfeval = pd.read_csv('eval.csv')  # testing data
y_train = dftrain.pop('survived')  # removing the survived column
y_eval = dfeval.pop('survived')

print(dftrain.head())  # This will show us the first 5 items in our dataframe.
print(dftrain.describe())  # statistical analysis of our data
print(dftrain.shape)
print(y_train.head())

# dftrain.age.hist(bins=20)
# dftrain.sex.value_counts().plot(kind='barh')
# dftrain['class'].value_counts().plot(kind='barh')
# pd.concat([dftrain, y_train], axis=1).groupby('sex').survived.mean().plot(kind='barh').set_xlabel('% survive')

# Before we continue and create/train a model we must convet our categorical data into numeric data.
# We can do this by encoding each category with an integer (ex. male = 1, female = 2).
CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck', 'embark_town', 'alone']
NUMERIC_COLUMNS = ['age', 'fare']

feature_columns = []
for feature_name in CATEGORICAL_COLUMNS:
    vocabulary = dftrain[feature_name].unique()  # gets a list of all unique values from given feature column
    feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

for feature_name in NUMERIC_COLUMNS:
    feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

print(feature_columns)
