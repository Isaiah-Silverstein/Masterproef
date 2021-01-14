import matplotlib.pyplot as plt
import numpy as np
import random, time
from flatmap_functions import *
import pandas as pd
import itertools
import multiprocessing, concurrent.futures
start = time.perf_counter()

""" F L A T M A P    S Y N C """


### PARAMETER VALUES ACROSS SIMULATIONs ###

tau_values = np.linspace(0.1,2,20)
alpha_values = np.linspace(0.1,0.9,10)
beta_values = np.linspace(0.1,0.9,10)

paramlist = list(itertools.product(alpha_values,beta_values,tau_values))


### SIMULATION PARAMETERS ###

reward_size = 100
map_size = 10

n_trials = 100
n_steps = 400


# # # # # # # # # # # # # #
# # Setting up the map  # #
# # # # # # # # # # # # # #
""" The agent begins in a walled grid and has to find 
    the goal to obtain a reward."""
# Grid #
states = create_square_grid(size = map_size, goal_location=[int(map_size-2),1], reward_size=reward_size)
state_set = list(range(int(states.shape[0]*states.shape[1]))) #index of states

#set of actions
move_name=["UP", "R-UP", "RIGHT","R-DOWN","DOWN","L-DOWN", "LEFT" ,"LEFT-UP"] 
moves = [[-1, 0],[-1, 1], [0, 1], [1, 1], [1, 0],[1, -1], [0, -1], [-1, -1]]
action_set = list(range(len(moves))) #index list
n_actions = int(len(action_set))




def run_tau_sim(param_tuple = tuple):
    """ Main simulation function. It accepts a tuple of relevant parameter values and spits out relevant results"""


    #Insert input tuple into 'readable' dictionary
    parameters = {"alpha": param_tuple[0]
            ,"beta": param_tuple[1]
            ,"gamma": 0.9
            ,"tau": param_tuple[2]}

    print(parameters)


    # # # # # # # # # # # # # #
    # #      Simulation     # #
    # # # # # # # # # # # # # #

    # Logging dependent variables

    Goal_reach  = np.zeros((n_steps,n_trials))     #record if goal is reached 
    Move = np.zeros((n_steps,n_trials))            #record move
    pred_err = np.zeros((n_steps,n_trials)) #logging the prediction error
    trial_length = np.zeros((n_trials))


    exploration_rate = np.zeros((n_steps,n_trials))
    
    #Resetting weights
    W = np.zeros((states.shape[0],states.shape[1],n_actions))  #initial state-action weights
    V = np.zeros((states.shape[0],states.shape[1]))            #initial state weight
    
    # Learning Parameters

    greedy, exploration = 0,0
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
            action_index = softmax_action(action_weights=W[current_loc[0],current_loc[1],:],tau= parameters["tau"])

            if action_index in np.where(W[current_loc[0],current_loc[1],:] == max(W[current_loc[0],current_loc[1],:]))[0]:
                greedy+=1
            else:
                exploration+=1
            
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
    
    # plt.figure("Value maps of root-level")
    # plt.imshow(V, cmap = 'hot', interpolation = 'nearest')
    # plt.show()

    # plt.plot(range(n_trials),trial_length,"--")
    # plt.title("Number of steps per trial")
    # plt.show()

    return [parameters['alpha'],parameters['beta'],parameters['tau'],exploration]


if __name__ == '__main__':
    """Create Multiprocessing pool that alocates loop iterations across cores according to hardware"""
    with concurrent.futures.ProcessPoolExecutor(max_workers= 10) as executor:
        #Results are logged as objects into a lists
        results = [executor.submit(run_tau_sim,param_tuple) for param_tuple in paramlist]
        #Retrieve 'actual' results from futures object
        results = [f.result() for f in results]


data_export = pd.DataFrame(results,columns = ['Alpha','Beta','Tau','Exploration'])
data_export.to_csv("Tau_Valuesl")
 
finish = time.perf_counter()
print("Script took {} sec to finish".format(finish-start))