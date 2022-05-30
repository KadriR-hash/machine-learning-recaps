from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')

print(dftrain.head())
print(dftrain.describe())
print(dftrain.shape)
print(y_train.head())

#dftrain.age.hist(bins=20)
#dftrain.sex.value_counts().plot(kind='barh')

#dftrain['class'].value_counts().plot(kind='barh')
#pd.concat([dftrain, y_train], axis=1).groupby('sex').survived.mean().plot(kind='barh').set_xlabel('% survive')
