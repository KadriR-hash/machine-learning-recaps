from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
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


# The Training Process

# Input function
def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
    def input_function():  # inner function, this will be returned
        # create tf.data.Dataset object with data and label
        ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))
        if shuffle:
            ds = ds.shuffle(1000)  # randomize order of data
        # split dataset into batches of 32 and repeat process for number of epochs
        ds = ds.batch(batch_size).repeat(num_epochs)
        return ds  # return a batch of the dataset

    return input_function  # return a function object for use


# here we will call the input_function that was returned to us to get a dataset object we can feed to the model
train_input_fn = make_input_fn(dftrain, y_train)
eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)

# Creating the Model ( linear regression algorithm)
linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)
# We create a linear estimtor by passing the feature columns we created earlier

# Training the Model
linear_est.train(train_input_fn)  # train
result = linear_est.evaluate(eval_input_fn)  # get model metrics/stats by testing on testing data
print(result['accuracy'])  # the result variable is simply a dict of stats about our model (0.74242425)

# Predict using the eval  data
pred_dicts = list(linear_est.predict(eval_input_fn))
probs = pd.Series([pred['probabilities'][1] for pred in pred_dicts])

probs.plot(kind='hist', bins=20, title='predicted probabilities')
plt.show()
