# split_train_test.py
import pandas as pd
import config

from sklearn.model_selection import train_test_split

# Read the raw data source 
df = pd.read_csv(config.RAW_DATA)


# split into features and target variables
y = df['User Rating']
x = df.drop('User Rating', axis = 1)

# split into training and test sets
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)


# output data files 
 df.to_csv(config.TRAIN_FOLDS, index = False)



print(x_train)