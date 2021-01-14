#%%
from flatmap_functions import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import os, random
import itertools, time
import multiprocessing, concurrent.futures
""" SYNC ROOMS """

# # PARAMETERS # #
"""Learning"""
reward_size = 100     # reward at goal
sub_reward_size = 99  # pseudo reward at subgoal
alpha = 0.01          # actor learning parameter
beta = 0.01            # critic learning parameter
tau = 20            # exploration parameter
gamma = 0.9           # discount parameter

"""Structural"""
n_trials = 300


#Putting them in a dictionary
parameters = {"alpha": alpha
            ,"beta": beta 
            ,"gamma": gamma
            ,"tau": tau
            ,"reward size" : reward_size
            ,"sub reward size" : sub_reward_size}

# # # # # # # # # # # # # #
# # Setting up the map  # #
# # # # # # # # # # # # # #
""" The agent begins in a walled map with rooms and has to find 
    the goal to obtain a reward."""
# Get map from file
states = create_grid_from_file(map_file='Models/Maps/map4.txt',reward_size=reward_size,sub_reward_size=sub_reward_size)

#get initiation states
initiation_set = get_intitation_states(grid = states, reward_size=reward_size, sub_reward_size=sub_reward_size)

#get termination states 
termination_set = get_termination_states(grid = states, sub_reward_size=sub_reward_size)

#%%
#set of primary actions
move_name=["UP", "R-UP", "RIGHT","R-DOWN","DOWN","L-DOWN", "LEFT" ,"LEFT-UP"] 
moves = [[-1, 0],[-1, 1], [0, 1], [1, 1], [1, 0],[1, -1], [0, -1], [-1, -1]]
move_set = list(range(len(moves))) #index list

#Total number of doors
n_doors = np.count_nonzero(states == sub_reward_size)
#Total number of action choices at the root level
action_set = range(len(move_set)+2)
print(action_set)
#Total number of higher level weights
option_set = range(int(n_doors*2))



# # # # # # # # # # # # # #
# # Setting up Weights  # #
# # # # # # # # # # # # # #

# Root level weights
W = np.zeros((states.shape[0],states.shape[1],len(action_set)))
V = np.zeros((states.shape[0],states.shape[1]))

# Higher level weights
W_option = np.zeros((states.shape[0],states.shape[1],len(move_set),len(option_set)))
V_option = np.zeros((states.shape[0],states.shape[1],len(option_set)))





#%% 
"""  Option Training """
n_steps_training = 5000000



#Step counter
o_step_list = [[] for _ in range(len(option_set))]



#intializing variables
n_steps = 0        #step counter
n_o_steps = 0      #steps in option counter
option_choice = 0
at_goal = False
""" Start training"""
current_loc = [2,int(states.shape[1]-2)] #start in top left
while n_steps < n_steps_training:
    n_o_steps=0
    #action choice on based on root level weights
    action_index = softmax_action(action_weights = W[current_loc[0],current_loc[1],:],tau = tau)

    chose_option = action_index>max(move_set) #check if the agent chose a move or an option
    
    """ If the agent chose an option it goes into a higher order weightspace"""
    ###############################################################################################################
    if chose_option:
        n_o_steps = 0
        option_choice+=1
        init_loc = current_loc #log initial location
        #Select the correct set of higher order weights
        option_index = initiation_set[init_loc[0],init_loc[1],int(action_index-len(move_set))]

        # Higher order Weight Matrices
        W_o = W_option[:,:,:,int(option_index)]
        V_o = V_option[:,:,int(option_index)]

        #Get termination state
        termination_state = termination_set[int(option_index)]
        """Higher order action sequence"""
        while current_loc != termination_state:
            #Choose a step to make 
            action_index = softmax_action(action_weights= W_o[current_loc[0],current_loc[1],:,], tau = parameters["tau"])
            #Make that step
            new_loc = update_location(grid =states, loc =current_loc, move=moves[action_index])
            #Log coordinates for weight updating function
            y, x, y_p, x_p, a = current_loc[0], current_loc[1], new_loc[0], new_loc[1], action_index #location coordinates
            
            
            if new_loc == termination_state:
                sub_reward = states[y_p][x_p]
            else: 
                sub_reward = 0
            
            #Update Weights
            delta = sub_reward +gamma*V_o[y_p][x_p] - V_o[y][x]

            V_o[y][x] = V_o[y][x] + alpha*delta
            W_o[y][x][a] = W_o[y][x][a] + beta*delta 

            n_steps+=1 #took a step!
            n_o_steps+=1 #took a step in an option
            current_loc = new_loc
            if n_steps == n_steps_training:
                break

        #print("made it out of option with {} steps".format(n_o_steps))
        W_option[:,:,:,int(option_index)] = W_o 
        V_option[:,:,int(option_index)] = V_o

    ###############################################################################################################
        """ If the agent chose an option it goes into a higher order weightspace"""
    else:
        #get new location
        new_loc = update_location(grid = states, loc=current_loc,move = moves[action_index])
        n_steps+=1 #took a step!
        current_loc = new_loc #update location and move on ur just training

    ################################################################################################################



print(option_choice)
np.save("Option_State_Weights.npy",V_option)
np.save("Option_Action_Weights.npy",W_option)
for i in range(len(option_set)):
    plt.figure("Value maps of learned options")
    plt.subplot(4,4,i+1)
    plt.imshow(V_option[:,:,i], cmap = 'hot', interpolation = 'nearest')
plt.show()



plt.figure("Value maps of root-level")
plt.imshow(V, cmap = 'hot', interpolation = 'nearest')
plt.show()





