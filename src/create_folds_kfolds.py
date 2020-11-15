# create_folds.py
import pandas as pd
from sklearn import model_selection
import config

if __name__ == "__main__":
   # Training data is in a CSV file called train.csv
   df = pd.read_csv(config.TRAIN_NO_FOLDS)

   # we create a new column called kfold and fill it with -1
   df['kfold'] = -1 

   # the next step is to randomize the rows of the data
   df = df.sample(frac=1).reset_index(drop=True)

   # initiate the kfold class from model_selection module
   kf = model_selection.KFold(n_splits=5)

   # fill in the new kfold column
   for fold, (trn_,val_) in enumerate(kf.split(X=df)):
      df.loc[val_, 'kfold'] = fold
   
   # save the new csv with kfold column
   df.to_csv(config.TRAIN_FOLDS, index = False)