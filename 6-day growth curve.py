# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 14:21:16 2020

@author: jordan
"""


import matplotlib.pyplot as plt
import numpy as np

### point graph

y= [0, 100, 1870, 1910, 3460, 2360]
x = [1 ,2 ,3 ,4, 5, 6]
plt.plot(x,y, linewidth = 3, marker='.', markersize = 17, markeredgecolor='yellow')
plt.title("6-day Growth Curve")

plt.xlabel('number of cells x1000')
plt.ylabel('number of days')


    
    