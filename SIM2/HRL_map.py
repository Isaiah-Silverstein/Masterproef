
from flatmap_functions import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import os, random
import itertools, time
import multiprocessing, concurrent.futures
## Training by trial and Error





""" SYNC ROOMS """
def run_HRL_map(mapname,n_trials = int, W_option = [], V_option= [], seed = int, alpha = 0.15, beta = 0.2, tau =3, gamma = 0.9, max_steps = 1000, reward_size = 100,goal =[],start = []):
    """HRL algorithm to solve map problem faster"""

    #softmax function initialized within function for random seed issues during multiprocessing
    np.random.seed(seed)
    def softmax_action(action_weights = [], tau = int):
        action_indices = list(range(len(action_weights)))
        f = np.exp((action_weights - np.max(action_weights))/tau)  # shift values
        action_prob = f / f.sum(axis=0)
        action_index = np.random.choice(action_indices, 1, p=action_prob)
        return action_index[0]
    time0 = time.perf_counter()
    print("Running that good HRL stuff")
    # # PARAMETERS # #
    """Learning"""
    sub_reward_size = 99  # pseudo reward at subgoal
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
    states = create_grid_from_file(map_file=mapname,goal_location = goal ,reward_size=reward_size,sub_reward_size=sub_reward_size)

    #get initiation states
    initiation_set = get_intitation_states(grid = states, reward_size=reward_size, sub_reward_size=sub_reward_size)

    #get termination states 
    termination_set = get_termination_states(grid = states, sub_reward_size=sub_reward_size)

    #set of primary actions
    move_name=["UP", "R-UP", "RIGHT","R-DOWN","DOWN","L-DOWN", "LEFT" ,"LEFT-UP"] 
    moves = [[-1, 0],[-1, 1], [0, 1], [1, 1], [1, 0],[1, -1], [0, -1], [-1, -1]]
    move_set = list(range(len(moves))) #index list

    #Total number of doors
    n_doors = np.count_nonzero(states == sub_reward_size)
    #Total number of action choices at the root level
    action_set = range(len(move_set)+2)
    #Total number of higher level weights
    option_set = range(int(n_doors*2))



    # # # # # # # # # # # # # #
    # # Setting up Weights  # #
    # # # # # # # # # # # # # #

    # Root level weights
    W = np.zeros((states.shape[0],states.shape[1],len(action_set)))
    V = np.zeros((states.shape[0],states.shape[1]))

    #Top Level weights
    W_option = W_option
    V_option = V_option




    """ Goal Training """
    start_loc = start



    #logging arrays
    #Step counter
    
    trial_steps = np.zeros((n_trials))
    options_chosen = np.zeros((n_trials))
    for trial in range(n_trials):
        option_choice=0
        n_steps = 0
        n_o_steps = 0
        at_goal = False
        current_loc = start_loc

        while not at_goal:


            action_index = softmax_action(action_weights = W[current_loc[0],current_loc[1],:],tau = tau)  


            chose_option = action_index>max(move_set)
            ##########################################################################
            if chose_option:
                init_action = action_index
                n_o_steps=0            #reset step counter
                option_choice+=1       #add an option choice counter
                init_loc = current_loc #log initial location
                #Select the correct set of higher order weights
                o = int(initiation_set[init_loc[0],init_loc[1],int(action_index-len(move_set))])

 
                r_cum_list = []
                #Get termination state
                termination_state = termination_set[o]
                """Higher order action sequence"""
                while current_loc != termination_state:

                    #make a move!
                    action_index = softmax_action(action_weights= W_option[current_loc[0],current_loc[1],:,o], tau = parameters["tau"])
                    new_loc = update_location(grid=states,loc=current_loc,move=moves[action_index])


                    y, x, y_p, x_p, a = current_loc[0], current_loc[1], new_loc[0], new_loc[1], action_index #location coordinates


                    """Rewards at top and root level"""
                    sub_reward = 0

                    if states[new_loc[0],new_loc[1]] == reward_size:
                        at_goal = True
                        reward = reward_size
                        
                    else:
                        reward = 0
                        at_goal = False
                    ## Create the list of rewards found for the option
                    r_cum_list.append((gamma**(n_o_steps - 1)*reward))
                    #calculate r_cum (hihi)
                    r_cum = np.sum(r_cum_list)

                    if new_loc == termination_state:
                        sub_reward = sub_reward_size

                    
                    #Update Weights
                    delta_o = sub_reward + gamma*V_option[y_p][x_p][o] - V_option[y][x][o]

                    V_option[y][x][o] = V_option[y][x][o] + alpha*delta_o
                    W_option[y][x][a][o] = W_option[y][x][a][o] + beta*delta_o               
                    
                    n_steps+=1
                    n_o_steps+=1
                    current_loc = new_loc #update current location




                    #End of option when at goal

                    if n_steps>=max_steps:
                        break
                    elif at_goal:
                        break
                


                
                y,x,y_p,x_p,a = init_loc[0],init_loc[1],new_loc[0],new_loc[1], init_action

                
                delta = r_cum + (gamma**n_o_steps)*V[y_p][x_p] - V[y][x]
            
                V[y][x] = V[y][x] + alpha*delta

                W[y][x][a] = W[y][x][a] + beta*delta
            


                ##########################################################################
            else:
                new_loc = update_location(grid = states, loc=current_loc,move = moves[action_index])

                y ,x ,y_p, x_p, a = current_loc[0],current_loc[1], new_loc[0],new_loc[1],action_index
                
                if states[new_loc[0],new_loc[1]] == reward_size:
                    at_goal = True
                    reward = reward_size
                else:
                    reward = 0
                
                delta = reward + gamma*V[y_p][x_p] - V[y][x]
                
                V[y][x] = V[y][x] + alpha*delta

                W[y][x][a] = W[y][x][a] +beta*delta

                current_loc = new_loc
                n_steps+=1
            if n_steps>max_steps:
                #print("Didn't make it :(")
                break
        ##############################################################################################
        """After the episode"""
        #print(current_loc)
        #print("I am at goal location!")
        trial_steps[trial] = n_steps
        options_chosen[trial] = option_choice
        #print("I took {} options this episode".format(option_choice))
    time1 = time.perf_counter()
    print("For the third model I took {} seconds".format(time1-time0))
    return trial_steps, V, V_option, W_option
    


if __name__ == "__main__":
          
    n_trials= 300
    mapname = '/home/isaiah/Documents/Thesis/Code/Models/Maps/map4.txt'
    states = create_grid_from_file(map_file=mapname,reward_size=100,sub_reward_size=99)
    n_comparisons = 50
    n_sim = 6

    comparisons = np.zeros((n_comparisons,n_sim,n_trials))
    for n in range(n_comparisons):
        print(states)
        W_option_list =[]
        V_option_list=[]
        
        #Initial start and goal location

        # Higher level weights
        n_moves = 8
        n_options = 8
        W_option = np.zeros((states.shape[0],states.shape[1],n_moves,n_options)) #initiales state action values
        V_option = np.zeros((states.shape[0],states.shape[1],n_options))               #initialize the option state values
        for sim in range(n_sim):


            diff_y = 0
            diff_x = 0
            while int(diff_y+diff_x) < 12:
                    start = random_loc(states)
                    goal = random_loc(states)
                    diff_y = abs(start[0]-goal[0])
                    diff_x = abs(start[1]-goal[1])

            print(start,goal)

            trial_length, V, V_option, W_option =run_HRL_map(mapname,n_trials = n_trials, tau = 5,V_option = V_option,W_option =W_option,seed = sim,  max_steps = 1000, reward_size = 100,goal= goal, start=start)
            
           
            W_option_list.append(W_option)
            V_option_list.append(V_option)

            comparisons[n,sim,:] = trial_length


        


    plt.figure("Value maps of root-level")
    plt.imshow(V, cmap = 'hot', interpolation = 'nearest')
    plt.show()


    for i in range(n_options):
        plt.figure("Value maps of learned options")
        plt.subplot(4,4,i+1)
        plt.imshow(V_option[:,:,i], cmap = 'hot', interpolation = 'nearest')
    plt.show()



    
    print(comparisons.shape)
    sim_steps = np.mean(comparisons,axis = 0)
    np.save("HRL_learning",sim_steps)
    print(sim_steps.shape)
    colorlist = ['b','m','k','g','c','r']
    index_list = [0,n_sim-1]
    c=0
    for i in index_list:
        plt.plot(range(n_trials),sim_steps[i,:],'-',color=colorlist[c],alpha = 0.8, label = "Simulation: {}".format(i+1))
        c+=1
    plt.title("Generalizing options")
    plt.ylabel("Steps")
    plt.xlabel("Episode")
    plt.legend(loc ='upper right')
    plt.show()



