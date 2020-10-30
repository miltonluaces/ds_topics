# BINNING: NUMERICAL TO CATEGORICAL

import numpy as np
import pandas as pd 

# Convert numerical columns to categorical binning ranges
def bin_num2cat(df, column, bins, letters=False):
    labels=None
    if letters: 
        letters = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        labels = letters[0:len(bins)-1]
    cats = pd.cut(x=df.age, bins=bins, labels=labels) 
    cats = cats.to_frame() 
    cats.columns = ['range'] 
    df = pd.concat([df,cats], axis=1) 
    return df

if __name__ == "__main__":
    df = pd.DataFrame(np.random.randint(low=1, high=9, size=(100, 1)),columns = ['age']) 
    
    df1 = bin_num2cat(df=df, column='Age', bins=[0,3,6,9])
    print(df1)

    df2 = bin_num2cat(df=df, column='Age', bins=[0,3,6,9], letters=True)
    print(df2)