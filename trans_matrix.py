# Make division default to floating-point, saving confusion
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


# transition probabilities
# create graph of connected rooms (approximated by either rooms that have doors next to each other or the belief that a person can in 15 seconds travel to that room)
# this graph outlines where a person can go in a state
floor_graph = {
    'r1' : ['r2', 'r3','r4','r7'],
    'r2' : ['r1', 'r3','r4', 'r7'],
    'r3' : ['r1', 'r2', 'r4', 'r7'],
    'r4' : ['r1', 'r2', 'r3', 'r8', 'r9'],
    'r5' : ['r6', 'r8', 'r9', 'r13', 'c3'],
    'r6' : ['r5', 'r9', 'r10', 'r11', 'r15', 'c3'],
    'r7' : ['r1', 'r2', 'r3', 'c1'],
    'r8' : ['r4', 'r5', 'r9','r13'],
    'r9' : ['r4', 'r5', 'r8', 'r13'],
    'r10': ['c3', 'r11', 'r15'],
    'r11': ['c3', 'r10', 'r15'],
    'r12': ['r22', 'r25', 'outside'],
    'r13': ['r8', 'r9', 'r14', 'r24'],
    'r14': ['r24','r13'],
    'r15': ['c3', 'r10', 'r11', 'r16', 'r17'],
    'r16': ['c3', 'r15', 'r18', 'r17'],
    'r17': ['c3', 'r16', 'r18'],
    'r18': ['c3', 'r16', 'r17', 'r19'],
    'r19': ['c3', 'r18', 'r20', 'r21'],
    'r20': ['c3', 'r19', 'r21'],
    'r21': ['c3', 'r19', 'r20'],
    'r22': ['r12', 'r25', 'c1'],
    'r23': ['r24'],
    'r24': ['r13', 'r14', 'r23'],
    'r25': ['r22', 'r26', 'r27', 'c1', 'c2'],
    'r26': ['r25', 'r27','c1'],
    'r27': ['r26', 'r32'],
    'r28': ['r35', 'c4'],
    'r29': ['r30', 'c2', 'c4', 'o1'],
    'r30': ['r29', 'c4'],
    'r31': ['r27', 'r32', 'r33'],
    'r32': ['r27', 'r31', 'r33'],
    'r33': ['r27', 'r31', 'r33'],
    'r34': ['c2','c4'],
    'r35': ['r28', 'c2', 'c4'],
    'c1' : ['r7', 'r25', 'c2'],
    'c2' : ['r34', 'c1', 'c4'],
    'c3' : ['r5', 'r6', 'r10', 'r11', 'r15', 'r16', 'r17', 'r18', 'r19', 'r20', 'r21', 'o1'],
    'c4' : ['r28', 'r29', 'r35', 'c2', 'o1'],
    'o1' : ['r21', 'c3', 'c4'],
    'outside': ['r12']
}

# check % transitions from the current room to another from graph and data to gauge staying prob
def movement_check(data, room):
    room_col = data[room].to_numpy()
    room_col_shift = np.append(data[room][1:].to_numpy(), 0)
    # we find the difference of people in a room between timesteps and sum them as a % of total pop
    # shift_diff = np.absolute(room_col - room_col_shift)
    shift_diff = room_col - room_col_shift
    # print(shift_diff)
    sum = 0
    for diff in shift_diff:
        # if it is bigger implies moving away from room
        if diff > 0:
            sum += diff
    if room_col.sum() == 0:
        return 0
    else:
        return sum/room_col.sum()

# generate transition matrices
def create_transition_matrix(floor_graph, room_data, split_count):
    room_cols = list(room_data.columns)
    transition = {}
    # for each room, find prob of moving from that room to other rooms
    # find prob of staying in that room
    for room in floor_graph.keys():
        # initialize all 0 transition probabilities for that room
        trans = [0] * 41
        # print(room)
        # count the possiblity of moving to other room while going row by row in data
        for i in range(len(room_data) - 1):
            r1 = room_data.iloc[i] # current row
            r2 = room_data.iloc[i+1] # current row + 1

            # if the number of people in the room has changed, if the future is less than present
            # for one room e.g. r1
            if int(r1[room]) > int(r2[room]):
                people_change = int(r1[room]) - int(r2[room]) # people decrease
                # check where those people went
                for adj_room in floor_graph[room]: 
                    r1_adj = r1[adj_room] # values of each room
                    r2_adj = r2[adj_room]
                    people_change_adj = int(r2_adj) - int(r1_adj)
                    # people probably moved to these rooms
                    # if the change is big, assign all to one room
                    if people_change_adj >= people_change:
                        room_index = room_cols.index(adj_room)
                        trans[room_index] += people_change
                        break
                    # if change is small assign to each room, decrement from total change
                    elif people_change_adj > 0:
                        room_index = room_cols.index(adj_room)
                        trans[room_index] += people_change_adj
                        people_change -= people_change_adj
        
        room_total_count = room_data[room].to_numpy().sum()
        if room_total_count != 0:
            trans[:] = [x/room_total_count for x in trans]
        # check for stationary prob of staying in room
        if movement_check(room_data, room) >= 1:
            trans[room_cols.index(room)] = 0.1
        else:
            trans[room_cols.index(room)] = 1 - movement_check(room_data, room)
        transition[room] = trans
    pd.DataFrame(transition).to_csv("transition_" + str(split_count)+ ".csv")

# load data
data = pd.read_csv("data.csv", index_col = 0)
# split the data into 3 chunks, morning, noon, after noon
room_true_data = data[list(floor_graph.keys())]
splits = [0, 19, 2279, 2401]

for i in range(0,3):
    # print(splits[i], splits[i+1])
    trans_data = room_true_data[splits[i]: splits[i+1]]
    create_transition_matrix(floor_graph, trans_data, i)
    print("Matrix ", i, " created")