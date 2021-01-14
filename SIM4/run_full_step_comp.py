#Import packages
from flatmap_functions import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import os, random
import itertools, time
import multiprocessing, concurrent.futures

#Import models
from RL_map import run_map
from HRL_map import run_HRL_map
from RL_sync_map import run_RL_sync
from HRL_sync_map import run_HRL_sync
### PARAMETERS ###

n_trials = 300
alpha = 0.15   #Default
beta = 0.2     #Default
tau = 5        #Default
gamma =0.9     #Default
max_steps = 1000    #Default
reward_size = 100   #Default
mapname = '/home/isaiah/Documents/Thesis/Code/Models/Maps/map4.txt'

##### RUN EVERYTHING #####

list_of_models = [run_map, run_RL_sync, run_HRL_map, run_HRL_sync]
model_names = ["run_map", "run_RL_sync", "run_HRL_map", "run_HRL_sync"]

n_sim = 20

RL_steps = np.zeros((n_sim,n_trials))
RL_sync_steps= np.zeros((n_sim,n_trials))
HRL_steps = np.zeros((n_sim,n_trials))
HRL_sync_steps = np.zeros((n_sim,n_trials))
array_list = [RL_steps,RL_sync_steps,HRL_steps ,HRL_sync_steps]




for x in range(len(list_of_models)):

    """Run models n times for comparison"""


    """Create Multiprocessing pool that alocates loop iterations across cores according to hardware"""
    with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
        #Results are logged as objects into a list
        results = [executor.submit(list_of_models[x],mapname,max_steps = 2000, n_trials=n_trials,seed=i) for i in range(n_sim)]
        #Retrieve 'actual' results from futures object
        results = [f.result() for f in results]

        sim=0
        for result in results:
            array_list[x][sim,:] = result[0]
            sim+=1



# #Save trial length
np.save('RL_steps_{}.2'.format(n_sim),array_list[0])
np.save('RL_sync_steps_{}.2'.format(n_sim),array_list[1])
np.save('HRL_steps_{}.2'.format(n_sim),array_list[2])
np.save('HRL_sync_steps_{}.2'.format(n_sim),array_list[3])


#plt.savefig('Step_Comparison_{}.png'.format(i))


