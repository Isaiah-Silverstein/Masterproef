
from flatmap_functions import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import os, random
import itertools, time
import multiprocessing, concurrent.futures



""" SYNC ROOMS """
def run_HRL_map(mapname,n_trials = int, seed = int, alpha = 0.15, beta = 0.2, tau = 5, gamma = 0.9, max_steps = 1000, reward_size = 100,HRL=True):
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
    sub_reward_size =  99  # pseudo reward at subgoal
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
    states = create_grid_from_file(map_file=mapname,goal_location = [10,3],reward_size=reward_size,sub_reward_size=sub_reward_size)

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
    # Higher level weights
    W_option = np.zeros((states.shape[0],states.shape[1],len(move_set),len(option_set)))
    V_option = np.zeros((states.shape[0],states.shape[1],len(option_set)))





    #%% 
    """  Option Training """
    n_steps_training = 40000






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
            o = int(initiation_set[init_loc[0],init_loc[1],int(action_index-len(move_set))])

            #Get termination state
            termination_state = termination_set[o]
            """Higher order action sequence"""
            while current_loc != termination_state:
                #Choose a step to make 
                action_index = softmax_action(action_weights= W_option[current_loc[0],current_loc[1],:,o], tau = parameters["tau"])
                #Make that step
                new_loc = update_location(grid =states, loc =current_loc, move=moves[action_index])
                #Log coordinates for weight updating function
                y, x, y_p, x_p, a = current_loc[0], current_loc[1], new_loc[0], new_loc[1], action_index #location coordinates
                
                
                if new_loc == termination_state:
                    sub_reward = states[y_p][x_p]
                else: 
                    sub_reward = 0
                
                #Update Weights
                delta_o = sub_reward + gamma*V_option[y_p][x_p][o] - V_option[y][x][o]

                V_option[y][x][o] = V_option[y][x][o] + alpha*delta_o
                W_option[y][x][a][o] = W_option[y][x][a][o] + beta*delta_o     

                n_steps+=1 #took a step!
                n_o_steps+=1 #took a step in an option
                current_loc = new_loc
                if n_steps == n_steps_training:
                    break
            o_step_list[o].append(n_o_steps)



        ###############################################################################################################
            """ If the agent chose an option it goes into a higher order weightspace"""
        else:
            #get new location
            new_loc = update_location(grid = states, loc=current_loc,move = moves[action_index])
            n_steps+=1 #took a step!
            current_loc = new_loc #update location and move on ur just training

        ################################################################################################################

    # for i in range(len(option_set)):
    #     plt.figure("Value maps of learned options")
    #     plt.subplot(4,4,i+1)
    #     plt.imshow(V_option[:,:,i], cmap = 'hot', interpolation = 'nearest')
    # plt.show()



    """ Goal Training """
    start_loc = [2,int(states.shape[1]-2)] #start in top left



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

            #first action choice on based on root level
            if not HRL:
                action_index=8
                while action_index>max(move_set):
                    action_index = softmax_action(action_weights = W[current_loc[0],current_loc[1],:],tau = tau)
            else:
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

 
                
                #Get termination state
                termination_state = termination_set[o]
                """Higher order action sequence"""
                while current_loc != termination_state:

                    #make a move!
                    action_index = softmax_action(action_weights= W_option[current_loc[0],current_loc[1],:,o], tau = parameters["tau"])
                    new_loc = update_location(grid=states,loc=current_loc,move=moves[action_index])


                    y, x, y_p, x_p, a = current_loc[0], current_loc[1], new_loc[0], new_loc[1], action_index #location coordinates



                    if new_loc == termination_state:
                        sub_reward = sub_reward_size
                    else: 
                        sub_reward = 0
                    
                    #Update Weights
                    delta_o = sub_reward + gamma*V_option[y_p][x_p][o] - V_option[y][x][o]

                    V_option[y][x][o] = V_option[y][x][o] + alpha*delta_o
                    W_option[y][x][a][o] = W_option[y][x][a][o] + beta*delta_o               
                    
                    n_steps+=1
                    n_o_steps+=1
                    current_loc = new_loc #update current location

                    """Root-level stuff"""
                    if states[new_loc[0],new_loc[1]] == reward_size:
                        at_goal = True
                        reward = reward_size
                        break
                    else:
                        reward = 0
                        at_goal = False


                    #End of option when at goal

                    if n_steps>=max_steps:
                        print("didnt make it to subgoal")
                        print(init_loc)
                        print(new_loc)
                        break
                
                if at_goal:

                    print("made it")
                y,x,y_p,x_p,a = init_loc[0],init_loc[1],new_loc[0],new_loc[1], init_action

            
                delta = (gamma**(n_o_steps-1))*reward + (gamma**n_o_steps)*V[y_p][x_p] - V[y][x]
            
                V[y][x] = V[y][x] + alpha*delta

                W[y][x][a] = W[y][x][a] + beta*delta
            
                #Log additional things
                o_step_list[o].append(n_o_steps)
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
    return trial_steps, V #, V_option, o_step_list, options_chosen
    

# if __name__ == "__main__":
#     """Create Multiprocessing pool that alocates loop iterations across cores according to hardware"""
#     n_sim = 60
#     n_trials = 500
#     mapname = '/home/isaiah/Documents/Thesis/Code/Models/Maps/map4.txt'
    
#     with concurrent.futures.ProcessPoolExecutor(max_workers=11) as executor:
#         #Results are logged as objects into a list            
#         results = [executor.submit(run_HRL_map,mapname, n_trials=n_trials, seed= seed) for seed in range(n_sim)]
#         #results = [executor.submit(get_random_number,i) for i in range(n_sim)]
#         #Retrieve 'actual' results from futures object
#         results = [f.result() for f in results]

#     step_results = np.zeros((n_sim,n_trials))
#     sim =0
#     for result in results:
#         step_results[sim,:] = result[0]
#         sim+=1
    
#     np.save('HRL_step_results',step_results)

# def worker(procnum, return_dict):
#     """worker function"""
#     print(str(procnum) + " represent!")
#     return_value = procnum*2
#     #return_dict[procnum] = return_value
#     return return_value


# if __name__ == "__main__":
#     manager = multiprocessing.Manager()
#     return_dict = manager.dict()
#     jobs = []
#     for i in range(5):
#         p = multiprocessing.Process(target=worker, args=(i, return_dict))
#         jobs.append(p)
#         p.start()


#     for proc in jobs:
#         proc.join()

#     print(return_dict)


if __name__ == "__main__":
          
    n_trials= 300
    mapname = '/home/isaiah/Documents/Thesis/Code/Models/Maps/map4.txt'



    n_sim = 20
    sims_steps = np.zeros((n_sim,n_trials))
    sims_steps_RL = np.zeros((n_sim,n_trials))
    i=0
    for sim in range(n_sim):
        i+=1
        trial_length, V, V_option,n_o_steps,options_chosen =run_HRL_map(mapname,n_trials = n_trials, seed = i, alpha = 0.12, beta = 0.15, tau = 5, gamma = 0.9, max_steps = 2000, reward_size = 100)
        trial_length1, V1, V_option1,n_o_steps1,options_chosen1 =run_HRL_map(mapname,n_trials = n_trials, HRL=False, seed = i+i, alpha = 0.12, beta = 0.15, tau = 5, gamma = 0.9, max_steps = 2000, reward_size = 100)
        
        sims_steps[sim,:] = trial_length
        sims_steps_RL[sim,:] = trial_length1

        


    plt.figure("Value maps of root-level")
    plt.imshow(V, cmap = 'hot', interpolation = 'nearest')
    plt.show()


    for i in range(8):
        plt.figure("Value maps of learned options")
        plt.subplot(4,4,i+1)
        plt.imshow(V_option[:,:,i], cmap = 'hot', interpolation = 'nearest')
    plt.show()

    for i in range(8):
        plt.figure("Steps made within each option")
        plt.subplot(4,4,1+i)
        plt.plot(range(len(n_o_steps[i])),n_o_steps[i])
    plt.show()


    #trial_length1, V1, V1_option, n_osteps = run_HRL_map(mapname, n_trials=n_trials, seed=2,HRL=False)

    #plt.plot(range(n_trials), trial_length1,"c-")
    
    plt.plot(range(n_trials),sims_steps.mean(axis=0),"r-")
    plt.plot(range(n_trials),sims_steps_RL.mean(axis=0),"g-")
    plt.plot(range(n_trials),options_chosen,'b-')
    plt.title("Number of steps per trial")
    plt.show()



