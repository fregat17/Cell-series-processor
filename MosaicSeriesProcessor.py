import pandas as pd
import numpy as np
import re
import os

from matplotlib import pyplot as plt
from pylab import *
import matplotlib.colors as col
import matplotlib.patheffects as pe

def errorfill(x, y, yerr, color=None, alpha_fill=0.3, ax=None):
    ax = ax if ax is not None else plt.gca()
    if color is None:
        color = ax._get_lines.color_cycle.next()
    if np.isscalar(yerr) or len(yerr) == len(y):
        ymin = y - yerr
        ymax = y + yerr
    elif len(yerr) == 2:
        ymin, ymax = yerr
    ax.plot(x, y, color=color)
    ax.fill_between(x, ymax, ymin, color=color, alpha=alpha_fill)

class series:
    def __init__(self, path, fps):
        self.path = path
        self.name = os.path.basename(path).replace(".csv", "")
        self.fps = fps

    def xtix(self, mins):
        '''
        translates frames into time (min) and create ticks
        for X-axis
        '''
        coef = 1/60/self.fps
        fame_in_min = int(mins/coef)
        labs = [i*mins for i in range(0,len(self.x()[::fame_in_min]))]
        positions = self.x()[::fame_in_min]
        return positions, labs

    def fig_init(self, dpi, figsize, X, Y, mins):
        '''
        creates figure object with given properties
        '''
        fig = plt.figure(figsize=figsize, dpi=dpi)
        plt.xlim([0,len(self.x())]) #plt.xlim([0,len(self.x())])
        plt.ylim([0,1])
        plt.xlabel(X)
        plt.ylabel(Y)
        plt.xticks(*self.xtix(mins))
        return fig

    def lines(self):
        '''
        creates list of lists with Y values 
        of each selected ROI
        '''
        listik = []
        res = pd.read_csv(self.path)
        vals = []
        for i in res.columns:
            if 'Mean' in i:
                vals.append(i)
        ready_res = res[vals[:-1]].subtract(res[vals[-1]], axis=0)
        data = ready_res.apply(lambda x: x/x.max(), axis=0)
        for k in range(0,len(data.columns)):
            listik.append(data.values[:,k].tolist())
        return listik

    def mean(self):
        '''
        calculates average values for all ROIs
        '''
        res = pd.read_csv(self.path)
        vals = []
        for i in res.columns:
            if 'Mean' in i:
                vals.append(i)
        ready_res = res[vals[:-1]].subtract(res[vals[-1]], axis=0)
        data = ready_res.apply(lambda x: x/x.max(), axis=0)
        data_mean = np.mean(data, axis=1)
        return np.array(list(data_mean))

    def x(self):
        '''
        just returns the number of points for graph creation
        '''
        res = pd.read_csv(self.path)
        return np.array(list(range(0,len(res))))

    def std(self):
        '''
        std calculation for plot graph 
        with errorfill
        '''
        res = pd.read_csv(self.path)
        vals = []
        for i in res.columns:
            if 'Mean' in i:
                vals.append(i)
        ready_res = res[vals[:-1]].subtract(res[vals[-1]], axis=0)
        data = ready_res.apply(lambda x: x/x.max(), axis=0)
        data_std = np.std(data, axis=1)
        return np.array(list(data_std))

    def plot(self, mins=2, dpi=100, figsize=(6, 4), X='X', Y='Y'):
        '''
        plots mean for all ROIs 
        '''
        self.fig_init(dpi, figsize, X, Y, mins)
        plt.plot(self.x(), self.mean(), lw=2, color='#83A83B')

    def plot_errorfill(self, mins=2, dpi=100, figsize=(6, 4), X='X', Y='Y'):
        '''
        plots errorfill graph
        '''
        self.fig_init(dpi, figsize, X, Y, mins)
        errorfill(self.x(), self.mean(), self.std(), color='#83A83B', alpha_fill=0.25)

    def animate(self, path, mins=2, dpi=100, figsize=(6, 4), X='X', Y='Y'):
        '''
        Animates graph, renders each frame, 
        save all frames to a separate specified folder, 
        create a text file with paths to all frames for 
        further animation creation in imageJ -> import -> stack from list ...
        '''
        self.fig_init(dpi, figsize, X, Y, mins)

        for i in range(0,len(self.x())):
            for k in range(0, len(self.lines())):
                plt.plot(self.x()[:i], self.lines()[k][:i], color='#83A83B', lw=2,
                        path_effects=[pe.Stroke(linewidth=2.15, foreground='black'), pe.Normal()])

            plt.savefig(path +'\{}_frame_{}.png'.format(self.name,str(i).zfill(3)), dpi=dpi,
                        bbox_inches = 'tight', pad_inches = 0.2)


        files = sorted(glob.glob(os.path.join(path, '*.png')))
        with open(path + '\\' + self.name + '.txt', 'w') as in_files:
            in_files.writelines(fn + '\n' for fn in files)