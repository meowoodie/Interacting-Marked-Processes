#!/usr/bin/env python
# -*- coding: utf-8 -*-

import arrow
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np



def avg(myArray, N=2):
    cum = np.cumsum(myArray,0)
    result = cum[N-1::N]/float(N)
    result[1:] = result[1:] - result[:-1]

    remainder = myArray.shape[0] % N
    if remainder != 0:
        if remainder < myArray.shape[0]:
            lastAvg = (cum[-1]-cum[-1-remainder])/float(remainder)
        else:
            lastAvg = cum[-1]/float(remainder)
        result = np.vstack([result, lastAvg])

    return result

class AnimatedScatter(object):
    """An animated scatter plot using matplotlib.animations.FuncAnimation."""
    def __init__(self, numpoints=50):
        self.locs      = np.load("data/geolocation.npy")
        self.data      = np.load("data/maoutage.npy")[2000:4000, :]
        # self.data      = avg(self.data, N=7) # 34589
        print(self.data.shape)
        self.numpoints = numpoints
        self.stream    = self.data_stream(self.locs, self.data)

        # Setup the figure and axes...
        self.fig, self.ax = plt.subplots()
        # Then setup FuncAnimation.
        self.ani = animation.FuncAnimation(self.fig, self.update, interval=1, 
                                          init_func=self.setup_plot, blit=True)

    def setup_plot(self):
        """Initial drawing of the scatter plot."""
        x, y, s, c = next(self.stream).T
        self.scat = self.ax.scatter(x, y, c=c, s=s, vmin=np.log(1), vmax=np.log(self.data.max()), cmap="hot")
        self.ax.axis([self.locs[:,1].min()-0.3, self.locs[:,1].max()+0.3, self.locs[:,0].min()-0.3, self.locs[:,0].max()+0.3])
        # For FuncAnimation's sake, we need to return the artist we'll be using
        # Note that it expects a sequence of artists, thus the trailing comma.
        return self.scat,

    def data_stream(self, locs, data):
        """
        Generate a random walk (brownian motion). Data is scaled to produce
        a soft "flickering" effect.
        """
        xy = locs
        for t in range(data.shape[0]):
            s = np.ones((data.shape[1])) * 10
            c = np.log(data[t] + 1)
            print(t)
            yield np.c_[xy[:,1], xy[:,0], s, c]

    def update(self, i):
        """Update the scatter plot."""
        data = next(self.stream)

        # Set x and y data...
        self.scat.set_offsets(data[:, :2])
        # Set sizes...
        # self.scat.set_sizes(300 * abs(data[:, 2])**1.5 + 100)
        # Set colors..
        self.scat.set_array(data[:, 3])
        # Setup title.
        # self.ax.set_title

        # We need to return the updated artist for FuncAnimation to draw..
        # Note that it expects a sequence of artists, thus the trailing comma.
        return self.scat,



if __name__ == '__main__':
    a = AnimatedScatter()
    plt.show()

    # y = np.load("data/maoutage.npy")
    # print(y.shape)
    # plt.plot(y.sum(axis=1))
    # plt.show()