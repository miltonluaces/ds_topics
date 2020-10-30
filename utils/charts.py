# CHARTS

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def Pareto(data, labels=[], cumPlot=True, limit=1.0, axes=None):
    assert 0.0 <= limit <= 1.0, 'limit must be a positive scalar between 0.0 and 1.0'
    dataArgs=(); data_kw={}; line_args=('g',); line_kw={}; limit_kw={}
    # re-order the data in descending order
    data = list(data)
    n = len(data)
    if n!=len(labels):
        labels = range(n)
    ordered = sorted(zip(data, labels), key=itemgetter(0), reverse=True)
    ordData = [dat for dat, lab in ordered]
    ordLabels = [lab for dat, lab in ordered]
    
    # create the cumulative line data
    line_data = [0.0]*n
    total_data = float(sum(ordData))
    for i, dat in enumerate(ordData):
        if i==0: line_data[i] = dat/total_data
        else: line_data[i] = sum(ordData[:i+1])/total_data

    # determine where the data will be trimmed based on the limit
    ltcount = 0
    for ld in line_data:
        if ld<limit:
            ltcount += 1
    limLoc = range(ltcount+1)
    
    limData = [ordData[i] for i in limLoc]
    limLabels = [ordLabels[i] for i in limLoc]
    limLine = [line_data[i] for i in limLoc]
    
    # if axes is specified, grab it and focus on its parent figure; otherwise create a new figure
    if axes:
        plt.sca(axes)
        ax1 = axes
        fig = plt.gcf()
    else:
        fig = plt.gcf()
        ax1 = plt.gca()
    
    # Create the second axis
    if cumPlot: ax2 = ax1.twinx()
    
    # Plotting
    if 'align' not in data_kw: data_kw['align'] = 'center'
    if 'width' not in data_kw: data_kw['width'] = 0.9
    ax1.bar(limLoc, limData, *dataArgs, **data_kw)
    if cumPlot: ax2.plot(limLoc, [ld*100 for ld in limLine], *line_args, **line_kw)
    ax1.set_xticks(limLoc)
    ax1.set_xlim(-0.5,len(limLoc)-0.5)
    
    # Formatting
    if cumPlot:
        # since the sum-total value is not likely to be one of the tick marks, let's make it the top-most one, regardless of label closeness
        ax1.set_ylim(0, total_data)
        loc = ax1.get_yticks()
        newloc = [loc[i] for i in range(len(loc)) if loc[i]<=total_data]
        newloc += [total_data]
        ax1.set_yticks(newloc)
        ax2.set_ylim(0, 100)
        if limit<1.0:
            xmin,xmax = ax1.get_xlim()
            if 'linestyle' not in limit_kw:
                limit_kw['linestyle'] = '--'
            if 'color' not in limit_kw:
                limit_kw['color'] = 'r'
            ax2.axhline(limit*100, xmin-1, xmax-1, **limit_kw)
    
    # set the x-axis labels
    ax1.set_xticklabels(limLabels)
    
    # adjust the second axis if cumplot=True
    if cumPlot:
        yt = [str(int(it))+r'%' for it in ax2.get_yticks()]
        ax2.set_yticklabels(yt)

    if cumPlot: return fig,ax1,ax2
    else: return fig,ax1


if __name__ == "__main__":
    print('')