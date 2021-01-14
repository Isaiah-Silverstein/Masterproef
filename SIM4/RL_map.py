
import matplotlib.pyplot as plt
import numpy as np
import time, random
from flatmap_functions import *
import pandas as pd
import itertools
import multiprocessing, concurrent.futures



""" F L A T M A P """
def run_map(mapname,n_trials = int,seed=int, alpha = 0.15, beta = 0.2, tau = 5, gamma = 0.9, max_steps = 1000, reward_size = 100):

    np.random.seed(seed)
    def softmax_action(action_weights = [], tau = int):
        action_indices = list(range(len(action_weights)))
        f = np.exp((action_weights - np.max(action_weights))/tau)  # shift values
        action_prob = f / f.sum(axis=0)
        action_index = np.random.choice(action_indices, 1, p=action_prob)
        return action_index[0]
    """ The agent begins in a walled grid and has to find 
        the goal to obtain a reward."""
    time0 = time.perf_counter()
    print("Running first model: big map with regular ol' RL")

    # Grid #
    sub_reward_size = 0   #flat learning

    # # # # # # # # # # # # # #
    # # Setting up the map  # #
    # # # # # # # # # # # # # #

    states = create_grid_from_file(map_file=mapname,goal_location=[10,3],reward_size=reward_size,sub_reward_size=sub_reward_size)
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
    greedy=0
    exploration=0
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
    

            action_index = softmax_action(action_weights=W[current_loc[0],current_loc[1],:], tau = parameters["tau"])
            if action_index in np.where(W[current_loc[0],current_loc[1],:]== max(W[current_loc[0],current_loc[1],:]))[0]:
                greedy+=1
            else:
                exploration+=1
            
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
    
    print("I took {0} exploratory steps and {1} greedy steps this simulation".format(exploration,greedy))
    print("In this sim I took   a total {} steps".format(np.sum(trial_length)))
    time1 = time.perf_counter()
    print("For the first model I took {} seconds".format(time1-time0))
    return trial_length, V

if __name__ == "__main__":
    """Create Multiprocessing pool that alocates loop iterations across cores according to hardware"""
    n_sim = 100
    n_trials = 300
    mapname = '/home/isaiah/Documents/Thesis/Code/Models/Maps/map4.txt'


    seed_list = [random.random() for _ in range(n_sim)]
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=11) as executor:
        #Results are logged as objects into a list            
        results = [executor.submit(run_map,mapname, n_trials=n_trials,seed=seed) for seed in range(n_sim)]
        #results = [executor.submit(get_random_number,i) for i in range(n_sim)]
        #Retrieve 'actual' results from futures object
        results = [f.result() for f in results]

    step_results = np.zeros((n_sim,n_trials))
    sim =0
    for result in results:
        step_results[sim,:] = result[0]
        sim+=1
    plt.plot(range(n_trials),step_results.mean(axis=0),"b-")
    plt.title("Number of steps per trial")
    plt.show()

    #np.save('RL_step_results',step_results)



# if __name__ == '__main__':
#     # Compare map sizes
#     #number of trials
#     n_trials = 300
#     #list of map files
#     filelist = ['Models/Maps/map0.txt','Models/Maps/map1.txt','Models/Maps/map3.txt','Models/Maps/map4.txt']

#     trial_length_list = []
#     V_list = []
#     for name in filelist:
#         trial_length, V = run_map(name)
#         trial_length_list.append(trial_length)
#         V_list.append(V)



#     for i in range(len(filelist)):
#         plt.figure("State Value maps for different maps")
#         plt.subplot(2,2,i+1)
#         plt.imshow(V_list[i], cmap = 'hot', interpolation = 'nearest')
#     plt.show()



#     plt.plot(range(n_trials),trial_length_list[0],"r--")
#     plt.plot(range(n_trials),trial_length_list[1],"b--")
#     plt.plot(range(n_trials),trial_length_list[2],"r-")
#     plt.plot(range(n_trials),trial_length_list[3],"b-")
#     plt.title("Number of steps per trial")
#     plt.show()

