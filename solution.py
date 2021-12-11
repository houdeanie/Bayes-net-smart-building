'''
COMP9418 Assignment 2
This file is the example code to show how the assignment will be tested.

Name:  Dean Hou   zID: z5163159

Name: Tony Lu    zID: z5204814
'''

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

###################################
# Code stub
# 
# The only requirement of this file is that is must contain a function called get_action,
# and that function must take sensor_data as an argument, and return an actions_dict
# 


# thess global states variables demonstrates how to keep track of information over multiple 
# calls to get_action 
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
# makes lists fo sensors
all_sensors = {**reliable_sensors,**unreliable_sensors,**door_sensors}
nondoor_sensors = {**reliable_sensors,**unreliable_sensors}

# initialising start state by using the ground truth data row 1
start_state = np.zeros((1,41),dtype='float')
start_state[0][11] = 18
start_state[0][21] = 2
start_state[0][40] = 1
current_state = start_state.copy()
previous_state = start_state.copy()

rooms = ['r1', 'r2', 'r3', 'r4', 'r5', 'r6', 'r7', 'r8',
       'r9', 'r10', 'r11', 'r12', 'r13', 'r14', 'r15', 'r16', 'r17', 'r18',
       'r19', 'r20', 'r21', 'r22', 'r23', 'r24', 'r25', 'r26', 'r27', 'r28',
       'r29', 'r30', 'r31', 'r32', 'r33', 'r34', 'r35', 'c1', 'c2', 'c3', 'c4',
       'o1', 'outside']

threshold_on = 0.15

# import transition matrices
transition_matrix0 = pd.read_csv("transition_0.csv", index_col = 0)
transition_matrix1 = pd.read_csv("transition_1.csv", index_col = 0)
transition_matrix2 = pd.read_csv("transition_2.csv", index_col = 0)
transition_matrix_np0 = np.array(transition_matrix0)
transition_matrix_np1 = np.array(transition_matrix1)
transition_matrix_np2 = np.array(transition_matrix2)

# import reliability matrix
reliability_matrix = pd.read_csv("reliability.csv", index_col = 0)

# take sensor input, determine which lights to turn on or off
# first makes a transition/passage in time using the tranisition matix based on time
# then uses evidence/sensor reliability to adjust values based on observation
# then use thresholding to determine which lights to keep on or off
def get_action(sensor_data):
    # global params, declare state as a global variable so it can be read and modified within this function
    global rooms
    global current_state
    global previous_state
    global all_sensors
    global nondoor_sensors
    global door_sensors
    global threshold_on
    global transition_matrix_np0
    global transition_matrix_np1
    global transition_matrix_np2
    global reliability_matrix
    # print(current_state)
    # define dict to cahnge later to return
    actions_dict = {'lights1': 'on', 'lights2': 'on', 'lights3': 'on', 'lights4': 'on', 'lights5': 'on', 'lights6': 'on', 'lights7': 'on', 
                    'lights8': 'on', 'lights9': 'on', 'lights10': 'on', 'lights11': 'on', 'lights12': 'on', 'lights13': 'on', 'lights14': 'on', 
                    'lights15': 'on', 'lights16': 'on', 'lights17': 'on', 'lights18': 'on', 'lights19': 'on', 'lights20': 'on', 'lights21': 'on', 
                    'lights22': 'on', 'lights23': 'on', 'lights24': 'on', 'lights25': 'on', 'lights26': 'on', 'lights27': 'on', 'lights28': 'on', 
                    'lights29': 'on', 'lights30': 'on', 'lights31': 'on', 'lights32': 'on', 'lights33': 'on', 'lights34': 'on', 'lights35':'on'}

    # convert sensor data values for motion/no motion to 1 and 0
    for key, val in sensor_data.items():
        if key in nondoor_sensors.keys():
            if val == "motion":
                sensor_data[key] = 1
            else:
                sensor_data[key] = 0

    ##### perform passage of time step/ one time step forward #####
    # use specially designed transition matrices for specific time periods
    if int(sensor_data['time'].hour) == 8 and int(sensor_data['time'].minute) <= 10:
        current_state = previous_state @ transition_matrix_np0
    elif int(sensor_data['time'].hour) < 17:
        current_state = previous_state @ transition_matrix_np1
    elif int(sensor_data['time'].hour) == 17 and int(sensor_data['time'].minute) < 30:
        current_state = previous_state @ transition_matrix_np1
    else:
        current_state = previous_state @ transition_matrix_np2

    ###### observation step/ given evidence of sensors #####
    # adjust for sensor readings, + dealing with None
    # if unreliable motion sensor detected motion, check the sensors is reliable and replace with value
    # if reliable sensor detected moition, check the sensors is reliable and replace with value
    for sensor in nondoor_sensors.keys():
        # print("sensor_data",sensor_data[i], i)
        room = nondoor_sensors[sensor]
        room_index = rooms.index(room)
        # say we see a motion in the sensor, should we trust it and change the value?
        # sensor says there is motion in the room, but there doesnt seem ot be many people in the room from state
        # change the value to increase
        if sensor_data[sensor] != 0 and current_state[0][room_index] < threshold_on:
            # we note from the reliability matrix, that unreliable sensor 4 is very bad at predicting true positives
            if sensor != 'unreliable_sensor4':
                current_state[0][room_index] = reliability_matrix[sensor][0]
        # sensor says no motion in the room, but the room value is high
        # we change the room value based on how much we trust the sensor, usually to decrease or probability of a false negative
        elif sensor_data[sensor] == 0 and current_state[0][room_index] > threshold_on:
            current_state[0][room_index] = 1 - reliability_matrix[sensor][1]
      
    # setting value for door sensors, none = 0, if people in there then give it the value, if no people set to 0
    # check if the door values on either side match the evidence, if it does not, change it
    for sensor in door_sensors.keys():
        rooml = door_sensors[sensor][0]
        roomr = door_sensors[sensor][1]
        room_indexl = rooms.index(rooml)
        room_indexr = rooms.index(roomr)
        # change room value on true positive or false negative
        # replace only if we have gigh confience in our sensors
        if sensor_data[sensor] != None:
            # door sensor detected something
            if sensor_data[sensor] > 0:
                # check for left room and replace with probability of true positive
                if current_state[0][room_indexl] < threshold_on:
                    current_state[0][room_indexl] = eval(reliability_matrix[sensor][0])[0]
                # check for right room and replace with probability of true positive
                if current_state[0][room_indexr] < threshold_on:
                    current_state[0][room_indexr] = eval(reliability_matrix[sensor][1])[0]
            # no door value
            else:
                # check if left room is has probabiity of false negative
                if current_state[0][room_indexl]:
                    current_state[0][room_indexl] = 1 - eval(reliability_matrix[sensor][0])[1]
                # check for right room probability of false negative
                if current_state[0][room_indexr]:
                    current_state[0][room_indexr] = 1 - eval(reliability_matrix[sensor][1])[1]
    
    # setting the robot values in the state
    robot_sensors = ['robot1','robot2']
    for r in robot_sensors:
        if sensor_data[r] != None:
            room = eval(sensor_data[r])[0]
            people_count = eval(sensor_data[r])[1]
            sensor_room_index = rooms.index(room)
            # print(sensor_room_index)
            # if room.startswith('r'):
            current_state[0][sensor_room_index] = people_count
    
    # normalise all the values
    # if current_state.sum() > 0:
    #     current_state = current_state/current_state.sum()
    # print(current_state)
    # set the lights on/ lights off based on values   
    count = 0
    for i in actions_dict.keys():
        if current_state[0][count] > threshold_on:
            actions_dict[i] = "on"
        else:
            actions_dict[i] = "off"
        # print(current_state[0][count]*10, i, actions_dict[i])
        count = count + 1
    previous_state = current_state
    # print(list(current_state[0]))
    return actions_dict