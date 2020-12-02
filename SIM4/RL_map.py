
import matplotlib.pyplot as plt
import numpy as np
import random, time
from flatmap_functions import *
import pandas as pd
import itertools
import multiprocessing, concurrent.futures



""" F L A T M A P """
def run_map(mapname,n_trials = int, alpha = 0.15, beta = 0.2, tau = 5, gamma = 0.9, max_steps = 1000, reward_size = 100):
    """ The agent begins in a walled grid and has to find 
        the goal to obtain a reward."""
    time0 = time.perf_counter()
    print("Running first model: big map with regular ol' RL")

    # Grid #
    sub_reward_size = 0   #flat learning

    # # # # # # # # # # # # # #
    # # Setting up the map  # #
    # # # # # # # # # # # # # #

    states = create_grid_from_file(map_file=mapname,reward_size=reward_size,sub_reward_size=sub_reward_size)
    #set of actions
    move_name=["UP", "R-UP", "RIGHT","R-DOWN","DOWN","L-DOWN", "LEFT" ,"LEFT-UP"] 
    moves = [[-1, 0],[-1, 1], [0, 1], [1, 1], [1, 0],[1, -1], [0, -1], [-1, -1]]
    action_set = list(range(len(moves))) #index list
    n_actions = int(len(action_set))


    # Logging dependent variables
    trial_length = np.zeros((n_trials))

    #Insert input tuple into 'readable' dictionary
    parameters = {"alpha": alpha
            ,"beta": beta
            ,"gamma": gamma
            ,"tau": tau}


    # # # # # # # # # # # # # #
    # #      Simulation     # #
    # # # # # # # # # # # # # #

    #Resetting weights
    W = np.zeros((states.shape[0],states.shape[1],n_actions))  #initial state-action weights
    V = np.zeros((states.shape[0],states.shape[1]))            #initial state weight

    # Learning Parameters


    trial_length = np.zeros((n_trials))
    for trial in range(n_trials):

        """A trial is considered as each journey the actor makes until the goal
            or until it runs out of steps"""
        
        start_loc = [1,int(states.shape[1]-2)]   #start in the top left
        n_steps = 0
        at_goal = False
        while not at_goal:
            
            #starting location at first trial
            if n_steps == 0:
                current_loc = start_loc
            """""""""""""""
            Learning
            """""""""""""""
            # select action
            
            action_index = softmax_action(action_weights=W[current_loc[0],current_loc[1],:],tau= parameters["tau"])

            
            #update location
            new_loc = update_location(grid = states, loc=current_loc,move = moves[action_index])
            #log coordinates for weight matrices
            y,x,y_p,x_p,a = current_loc[0], current_loc[1], new_loc[0], new_loc[1], action_index #location coordinates


            
            if states[y_p][x_p] == reward_size:
                at_goal = True
                reward = reward_size
            else:
                at_goal = False #for now assume not at goal
                reward = 0
            #update weights according to TD-learning

            delta = reward + gamma*V[y_p][x_p] - V[y][x]   #prediction error
            V[y][x] = V[y][x] + alpha*delta                #state value update -> "Critic"
            W[y][x][a] = W[y][x][a] + beta*delta           #state-action value update -> "Actor"

            current_loc = new_loc
            n_steps+=1
            if n_steps == max_steps:
                #print("I didnt make it to goal")
                break
            
        trial_length[trial] = n_steps
    time1 = time.perf_counter()
    print("For the first model I took {} seconds".format(time1-time0))
    return trial_length, V





