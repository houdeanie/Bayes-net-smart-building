from __future__ import division
from __future__ import print_function

# Allowed libraries 
import numpy as np
import pandas as pd
import scipy as sp
import scipy.special
import heapq as pq
import matplotlib as mp
import matplotlib.pyplot as plt
import math
from itertools import product, combinations
from collections import OrderedDict as odict
import collections
from graphviz import Digraph, Graph
from tabulate import tabulate
import copy
import sys
import os
import datetime
import sklearn
import ast
import re
import pickle
import json

# generate proability of reliability of sensors reliability of sensors
# we can use this to create the emissions/probabilities
# for whether which rooms the sensors are directly in have people or not
data = pd.read_csv("data.csv", index_col = 0)

reliable_sensors = {
    'reliable_sensor1' : 'r16',
    'reliable_sensor2' : 'r5',   
    'reliable_sensor3' : 'r25',
    'reliable_sensor4' : 'r31',
    }
unreliable_sensors = {
    'unreliable_sensor1' : 'o1',
    'unreliable_sensor2' : 'c3',   
    'unreliable_sensor3' : 'r1',
    'unreliable_sensor4' : 'r24',
}
door_sensors = {
    'door_sensor1' : ['r8','r9'],
    'door_sensor2' : ['c1','c2'],
    'door_sensor3' : ['r26','r27'],
    'door_sensor4' : ['c4','r35'],
}

sensors_dict = {**reliable_sensors,**unreliable_sensors}

# set up reliability dictionary, initially all 0
reliability_dict = {}
for i in sensors_dict.keys():
    reliability_dict[i] = [0,0,0,0]

# count number of times reliable/unrealiable sensor is correct
for sensor in sensors_dict.keys():
    # data to be used for calculating the reliability of 
    # all the sensors against the ground truth
    print(sensor, sensors_dict[sensor])
    data_cols = list([sensor] + [sensors_dict[sensor]])
    # count times when motion / no motion and is correct
    correct = 0
    on_correct = 0
    total_on = 0
    off_correct = 0
    total_off = 0
    for index, row in data[data_cols].iterrows():
        if row[sensor] == "motion":
            total_on += 1
            if row[sensors_dict[sensor]] != 0:
                correct += 1
                on_correct += 1
        elif row[sensor] == "no motion":
            total_off += 1
            if row[sensors_dict[sensor]] == 0:
                correct += 1
                off_correct += 1

    TP = on_correct/total_on
    FP = 1 - TP
    TN = off_correct/total_off
    FN = 1 - TN
    print("TP: ", TP, "FP: ", FP)
    print("TN: ", TN, "FN: ", FN)
    print("TP and TN: ", correct/len(data))
    reliability_dict[sensor] = [TP, TN ,correct/len(data), 1 - (correct/len(data))]

print("\n")

# accuracy of door sensors
for sensor in door_sensors.keys():
    # data to be used for calculating the reliability of 
    # all the sensors against the ground truth
    data_cols = list([sensor] + door_sensors[sensor])
    # count times when motion / no motion and is correct
    # count times when on left or right door side
    motion_l_count = 0
    motion_r_count = 0
    total_motion = 0
    no_motion_l_count = 0
    no_motion_r_count = 0
    total_no_motion = 0
    correct = 0
    for index, row in data[data_cols].iterrows():
        if row[sensor] != 0:
            total_motion += 1
            if row[door_sensors[sensor][0]] != 0:
                motion_l_count += 1
            if row[door_sensors[sensor][1]] != 0:
                motion_r_count += 1
            if row[door_sensors[sensor][0]] != 0 or row[door_sensors[sensor][1]] != 0:
                correct += 1
        elif row[sensor] == 0:
            total_no_motion += 1
            if row[door_sensors[sensor][0]] == 0:
                no_motion_l_count += 1
            if row[door_sensors[sensor][1]] == 0:
                no_motion_r_count += 1
            if row[door_sensors[sensor][0]] == 0 or row[door_sensors[sensor][1]] == 0:
                correct += 1
    # true positives and true negatives for people on left and right side of doors
    TP_l = motion_l_count/total_motion
    TN_l = no_motion_l_count/total_no_motion
    TP_r = motion_r_count/total_motion
    TN_r = no_motion_r_count/total_no_motion
    print(sensor, row[door_sensors[sensor][0]], row[door_sensors[sensor][1]])
    print("TP left: ", TP_l, "TN left: ", TN_l)
    print("TP right: ", TP_r, "TN right: ", TN_r)
    print("correct: ", correct/len(data), 1 - correct/len(data))
    reliability_dict[sensor] = [[TP_l, TN_l], [TP_r, TN_r], correct/len(data), 1 - (correct/len(data))]

pd.DataFrame(reliability_dict).to_csv("reliability.csv")