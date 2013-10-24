#!/usr/bin/env python

import getopt
import json
import sys
import math
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullFormatter


class Usage(Exception):
    def __init__(self, msg):
        self.msg = msg

def help_msg():
    return """
  Usage: qsr_to_gmm.py [-h] <qsrmodel> <obj>

    qsrmodel        file including the QSR model for generationg the GMMs 

    -h, --help for seeing this msg
"""
    

if __name__ == "__main__":

    argv = None
    if argv is None:
        argv = sys.argv
    try:
        try:
            opts, args = getopt.getopt(argv[1:], "h", ["help"])
        except getopt.error as msg:
            raise Usage(msg)

        if ('-h','') in opts or ('--help', '') in opts or len(args) != 2:
            raise Usage(help_msg())

        #print('Parsing QSR model')
        obj = args[1]

        with open(args[0]) as qsr_file:    
            qsr_model = json.load(qsr_file)

            rel_pos = list()
            for scn in qsr_model:
                scn_types = scn[1]['type'].values()
                if obj in scn_types:
                    obj_pos = scn[1]['position'][obj.lower()]
                    table_pos = scn[1]['position']['table']
                    rel_pos.append([a - b for a, b in zip(obj_pos, table_pos)])

            x = list()
            y = list()
            for pos in rel_pos:
                x_rel = pos[0]
                y_rel = pos[1]
                x_rel_rotated = x_rel * math.cos(math.pi * 3/2) - y_rel * math.sin(math.pi * 3/2)
                y_rel_rotated = x_rel * math.sin(math.pi * 3/2) + y_rel * math.cos(math.pi * 3/2)
                x.append(x_rel_rotated)
                y.append(y_rel_rotated)

            H, xedges, yedges = np.histogram2d(y, x, bins=20)
            extent = [yedges[0], yedges[-1], xedges[-1], xedges[0]]
            plt.imshow(H, extent=extent,interpolation='nearest')
            plt.colorbar()
            plt.xlim(-0.9,0.9)
            plt.ylim(-0.5,0.5)
            plt.show()
                
            plt.plot(x, y, 'o')
            #fig, ax = plt.subplots()
            #im = ax.hexbin(x, y, gridsize=20)
            #fig.colorbar(im, ax=ax)
            plt.xlim(-0.9,0.9)
            plt.ylim(-0.5,0.5)
            plt.show()


            nullfmt   = NullFormatter()
            # definitions for the axes
            left, width = 0.1, 0.65
            bottom, height = 0.1, 0.65
            bottom_h = left_h = left+width+0.02

            rect_scatter = [left, bottom, width, height]
            rect_histx = [left, bottom_h, width, 0.2]
            rect_histy = [left_h, bottom, 0.2, height]

            # start with a rectangular Figure
            plt.figure(1, figsize=(8,8))

            axScatter = plt.axes(rect_scatter)
            axHistx = plt.axes(rect_histx)
            axHisty = plt.axes(rect_histy)

            # no labels
            axHistx.xaxis.set_major_formatter(nullfmt)
            axHisty.yaxis.set_major_formatter(nullfmt)

            # the scatter plot:
            axScatter.scatter(x, y)

            # now determine nice limits by hand:
            binwidth = 0.1
            xymax = np.max( [np.max(np.fabs(x)), np.max(np.fabs(y))] )
            lim = ( int(xymax/binwidth) + 1) * binwidth

            axScatter.set_xlim( (-0.9, 0.9) )
            axScatter.set_ylim( (-0.5, 0.5) )

            bins = np.arange(-lim, lim + binwidth, binwidth)
            axHistx.hist(x, bins=bins)
            axHisty.hist(y, bins=bins, orientation='horizontal')

            axHistx.set_xlim( axScatter.get_xlim() )
            axHisty.set_ylim( axScatter.get_ylim() )

            plt.show()
            
    except Usage as err:
        print(err.msg)
        print("for help use --help")

