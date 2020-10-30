# DATA_BALANCE UTILS

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

# Show distribution of categorical labels (ds : dataset, column: label)
def catlabel_dist(ds, column):
    plt.figure(figsize=(12,8))
    ax = sb.countplot(x=column, data=ds)
    ax.set_title('Distribution of Label ' + column)
    for p in ax.patches:
        ax.text(p.get_x()+p.get_width()/2., p.get_height() + 3, '{:1.2%}'.format(p.get_height()/ds.shape[0]), ha="center") 

# Show distribution of numerical labels (ds : dataset, column: label)
def numlabel_dist(ds, column):
    plt.figure(figsize=(12,8))
    plt.title('Distribution of Label ' + column)
    ax = sb.histplot(ds[column]).grid(axis='x');



if __name__ == "__main__":
    ds = pd.read_csv('D:/data/csv/balance-scale.csv')
    catlabel_dist(ds=ds, column='target')

    ds = pd.read_csv('D:/data/csv/airline_passengers.csv')
    numlabel_dist(ds=ds, column='Passengers')
    plt.show()