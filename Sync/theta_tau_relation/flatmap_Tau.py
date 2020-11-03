import matplotlib.pyplot as plt
import numpy as np
import random
from flatmap_functions import *
import pandas as pd
""" F L A T M A P    S Y N C """
### MAIN PARAMETERS ###

reward_size = 20
map_size = 10

n_trials = 60
n_steps = 100





# # # # # # # # # # # # # #
# # Setting up the map  # #
# # # # # # # # # # # # # #
""" The agent begins in a walled grid and has to find 
    the goal to obtain a reward."""
# Grid #
states = create_square_grid(size = map_size, goal_location=[int(map_size-2),1], reward_size=reward_size)
print(states)
state_set = list(range(int(states.shape[0]*states.shape[1]))) #index of states

#set of actions
move_name=["UP", "R-UP", "RIGHT","R-DOWN","DOWN","L-DOWN", "LEFT" ,"LEFT-UP"] 
moves = [[-1, 0],[-1, 1], [0, 1], [1, 1], [1, 0],[1, -1], [0, -1], [-1, -1]]
action_set = list(range(len(moves))) #index list
n_actions = int(len(action_set))

#State-Action Weight Matrix
W = np.zeros((states.shape[0],states.shape[1],n_actions))  #initial state-action weights
V = np.zeros((states.shape[0],states.shape[1]))            #initial state weights


# # # # # # # # # # # # # #
# #      Simulation     # #
# # # # # # # # # # # # # #

# Logging dependent variables

Goal_reach  = np.zeros((n_steps,n_trials))     #record if goal is reached 
Move = np.zeros((n_steps,n_trials))            #record move
pred_err = np.zeros((n_steps,n_trials)) #logging the prediction error
trial_length = np.zeros((n_trials))


exploration_rate = np.zeros((n_steps,n_trials))

""" L O O P """
i=0


tau_values = np.arange(1,21)
alpha_values = np.linspace(0.25,0.3,2)
beta_values = np.linspace(0.25,0.3,2)
measures = np.array(["Exploration Steps", "Total Steps"])
Datatable = np.zeros((len(alpha_values),len(beta_values),len(tau_values)))




a=0
b=0
t=0
for alpha in alpha_values:
    
    b=0
    for beta in beta_values:
        t=0
        for tau in tau_values:

            #Resetting weights
            W = np.zeros((states.shape[0],states.shape[1],n_actions))  #initial state-action weights
            V = np.zeros((states.shape[0],states.shape[1]))            #initial state weight
            
            # Learning Parameters
            parameters = {"alpha": alpha
                        ,"beta": beta
                        ,"gamma": 0.9
                        ,"tau": tau}
            exploration = 0
            trial_length = np.zeros((n_trials))
            for trial in range(n_trials):
                """A trial is considered as each journey the actor makes until the goal
                    or until it runs out of steps"""
                
                start_loc = [1,int(states.shape[1]-2)]   #start in the top left

                for step in range(n_steps):
                    
                    #starting location at first trial
                    if step == 0:
                        current_loc = start_loc
                    """""""""""""""
                    Learning
                    """""""""""""""
                    # select action
                    

                    action_index = softmax_action(action_rate=W[current_loc[0],current_loc[1],:],tau= parameters["tau"])

                    if action_index != int(np.argmax(W[current_loc[0],current_loc[1],:])):
                        exploration +=1
                    
                    #update location
                    new_loc = update_location(grid = states, loc=current_loc,move = moves[action_index])

                    #log coordinates for weight matrices
                    coordinates = [current_loc[0], current_loc[1], new_loc[0], new_loc[1], action_index] #location coordinates

                    #update weights according to TD-learning
                    V, W, delta, at_goal = update_weights(param=parameters, index=coordinates, V=V, W=W, states=states)

                    
                    if at_goal:
                    
                        break
                    
                    current_loc = new_loc
            

                    #logging some stuff
                    #Move[step,trial] = action_index
                    #pred_err[step,trial] = delta
                
                trial_length[trial] = step
    #alpha
        #beta
            #tau
            Datatable[a,b,t] = exploration
            
            t+=1

        b+=1
    a+=1    







midx = pd.MultiIndex.from_product([alpha_values, beta_values, tau_values])
long_data = np.zeros((len(midx)))
i=0
for a in range(len(alpha_values)):
    for b in range(len(beta_values)):
        for t in range(len(tau_values)):
                
                long_data[i] = Datatable[a,b,t]
                i+=1

print(Datatable)
export = pd.DataFrame(long_data, index =midx)
export.to_csv("Tau_Values")
 


