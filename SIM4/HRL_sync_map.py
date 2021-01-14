from flatmap_functions import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import os, random
import itertools
import multiprocessing, concurrent.futures
import time
""" SYNC ROOMS """
def run_HRL_sync(mapname,n_trials = int,seed = int, alpha = 0.15, beta = 0.2, tau = 5, gamma = 0.9, max_steps = 1000, reward_size = 100):
    
    np.random.seed(seed)
    def softmax_action(action_weights = [], tau = int):
        action_indices = list(range(len(action_weights)))
        f = np.exp((action_weights - np.max(action_weights))/tau)  # shift values
        action_prob = f / f.sum(axis=0)
        action_index = np.random.choice(action_indices, 1, p=action_prob)
        return action_index[0]




    time0 = time.perf_counter()
    print("Running the HRL sync big boi")
    # # PARAMETERS # #
    """Learning"""
    sub_reward_size = 99  # pseudo reward at subgoal


    """Structural"""
    n_trials = n_trials

    n_steps_training = 40000
    srate =500
    total_time = int(1.5*srate)  #total timesteps or "time the agent gets to think about moving"

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

    #Total number of doors
    n_doors = np.count_nonzero(states == sub_reward_size)
    assert n_doors ==4, "nope can't do anything else than 4 doors sorry"

    #get initiation states
    initiation_set = get_intitation_states(grid = states, reward_size=reward_size, sub_reward_size=sub_reward_size)

    #get termination states 
    termination_set = get_termination_states(grid = states, sub_reward_size=sub_reward_size)

    #%%
    #set of primary actions
    move_name=["UP", "R-UP", "RIGHT","R-DOWN","DOWN","L-DOWN", "LEFT" ,"LEFT-UP"] 
    moves = [[-1, 0],[-1, 1], [0, 1], [1, 1], [1, 0],[1, -1], [0, -1], [-1, -1]]
    move_set = list(range(len(moves))) #index list
    n_moves = len(move_set)
    #Total number of action choices at the root level
    action_set = range(len(move_set)+2)
    n_actions = len(action_set)
    #Total number of higher level weights
    option_set = range(int(n_doors*2))
    n_options = len(option_set)


    # # # # # # # # # # # # # #
    # # Setting up Weights  # #
    # # # # # # # # # # # # # #

    # Root level weights
    W = np.zeros((states.shape[0],states.shape[1],len(action_set)))
    V = np.zeros((states.shape[0],states.shape[1]))

    # Higher level weights
    W_option = np.zeros((states.shape[0],states.shape[1],len(move_set),len(option_set)))
    V_option = np.zeros((states.shape[0],states.shape[1],len(option_set)))


    ############################################################################################################################################
    ############################################################################################################################################
    """  Option Training """

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


    """ Synchronization Matrices """

    # # # # # # # # # # # # # # # # # # # # # # # #
    # # Setting up the synchronization modules  # #
    # # # # # # # # # # # # # # # # # # # # # # # #

    """""""""""""""""""""
    Processing module
    """""""""""""""""""""

    # Initial variables #

    r2_max = 1    #maximum amplitude of nodes
    drift = .8      #rate of drift between coupling parameters

    cg_1 = (30/srate)*np.pi             #gamma band coupling parameter for input information
    cg_2 = cg_1 + (drift/srate)*2*np.pi #gamma band coupling parameter for actions
    
    damp = 0.3     #damping parameter
    decay = 0.9    #decay parameter
    noise = 0.5    #noise parameter

    # Initial matrices #

    #Setting up phase code neurons matrices
    S_Phase = np.zeros((2,states.shape[0],states.shape[1],total_time)) #State phase code units
    A_Phase_root = np.zeros((2,n_actions,total_time)) #Action phase code units for root level
    A_Phase_top  = np.zeros((2,n_moves,total_time))   #Action phase code units for top level

    #Setting up rate code neurons across entire task
    S_Rate = np.zeros((states.shape[0],states.shape[1],total_time)) #State rate code units
    A_Rate_root = np.zeros((n_actions,total_time)) #Action rate code units for root level
    A_Rate_top = np.zeros((n_moves,total_time))    #Action rate code units for top level

    """""""""""""""""""""
    Control   module
    """""""""""""""""""""

    # MFC #
    # Initial variables 
    r2_MFC = .7           #maximum amplitude MFC node
    damp_MFC = 0.03       # damping parameter MFC
    acc_slope = 10        # MFC slope parameter ---> steepness of burst probability distribution
    ct = (5/srate)*2*np.pi #theta band coupling parameter for MFC

    #Setting up phase code nodes for the MFC
    MFC = np.zeros((2,total_time))
    #Setting up phase code neuron for MFC -> Bernoulli rate code
    Be = 0

    """When the be value as the rate code of MFC
    reaches certain threshold the MFC will send a burst to coupled neurons"""

    # LFC #
    #Module indicating which states should be initiate action-state synchronization
    LFC = np.zeros((states.shape[0],states.shape[1],max_steps))

    #Module that gives the right indices to synchronize
    LFC_sync = 0

    # Recording sync
    #sync = np.zeros((states.shape[0],states.shape[0],n_actions,max_steps,n_trials)) 



    #####################################################################################################################################
    ########################################################################################################################################
    """ Goal Training """

    start_loc = [2,int(states.shape[1]-2)] #start in top left

    trial_steps = np.zeros((n_trials))
    for trial in range(n_trials):
        option_choice=0
        n_steps = 0
        at_goal = False
        current_loc = start_loc
        S_Phase[:,:,:,0] = (2*np.random.random_sample((2,states.shape[0],states.shape[1])))-1   # random starting points processing module
        A_Phase_root[:,:,0] = (2*np.random.random_sample((2,n_actions)))-1  # idem
        while not at_goal:
            #first action choice on based on root level
            if n_steps >0:
                S_Phase[:,:,:,0] = S_Phase[:,:,:,total_time-1]   # random starting points processing module
                A_Phase_root[:,:,0] = A_Phase_root[:,:,total_time-1]  # idem
            #phase reset
            MFC[:,0]=np.ones((2))*r2_MFC 

            # LFC setting instruction per step: each state is an input
            LFC[current_loc[0],current_loc[1],n_steps] = 1

            action_to_sync = softmax_action(action_weights = W[current_loc[0],current_loc[1],:],tau = tau)
            LFC_sync = int(action_to_sync)
            LFC_desync = list(range(len(action_set)))
            LFC_desync.pop(LFC_sync)     # List of actions the LFC desynchronizes
            # The actor makes the move #
            for t in range(total_time-1):

                
                #Update phase code neurons for actions and states in processing module
                #State phase code neurons   
                S_Phase[:,:,:,t+1] = update_phase(nodes=S_Phase[:,:,:,t], grid = True, radius=r2_max, damp = damp, coupling = cg_1,multiple=True )
                
                #Action phase code neurons
                A_Phase_root[:,:,t+1] = update_phase(nodes=A_Phase_root[:,:,t], grid = False, radius=r2_max, damp = damp, coupling = cg_2,multiple=True )

                #Update phase code untis of MFC
                MFC[:,t+1] = update_phase(nodes=MFC[:,t], grid = False, radius=r2_MFC, damp=damp_MFC, coupling=ct,multiple=False)

                #MFC rate code neuron-> Bernoulli process

                Be = 1/(1 + np.exp(-acc_slope*(MFC[0,t]-1.33)))    # Bernoulli process 
                #Bernoulli[tijd,step,trial] = Be                                 # logging Be value

                p = random.random()

                if p < Be:

                    Gaussian = np.random.normal(size = [1,2])  #noise factor as normal distribution
                    #Hit[tijd,step,trial] = 1
                    
                        
                    x, y = current_loc[1], current_loc[0]

                    #the LFC decides which state is paired with which actions

                    if LFC[y,x,n_steps]:
                        #The state the actor is in receives a burst because it is the only input
                        S_Phase[:,y,x,t+1] = decay*S_Phase[:,y,x,t] + Gaussian

                        # and all the actions that are to be synchronized to that state receive a burst
                        
                        if type(LFC_sync) is int:
                            A_Phase_root[:,LFC_sync,t+1] = decay*A_Phase_root[:,LFC_sync,t] + Gaussian
                        # Desynchronize all other actions !
                        for node in LFC_desync:
                            A_Phase_root[:,int(node),t+1] = decay*A_Phase_root[:,int(node),t] - Gaussian*noise

                S_Rate[current_loc[0],current_loc[1],t]= (1/(1+np.exp(-5*S_Phase[0,current_loc[0],current_loc[1],t]-0.6)))
                A_Rate_root[:,t]=(S_Rate[current_loc[0],current_loc[1],t]*(W[current_loc[0],current_loc[1],:]+1))*(1/(1+np.exp(-5*A_Phase_root[0,:,t]-0.6)))
            
            # select action
            action_index = int(np.argmax(np.sum(A_Rate_root[:,:],axis=1)))
            chose_option = action_index>max(move_set)
            ##########################################################################
            if chose_option:
                init_action = action_index
                n_o_steps=0            #reset step counter
                option_choice+=1       #add an option choice counter
                init_loc = current_loc #log initial location
                #Select the correct set of higher order weights
                option_index = initiation_set[init_loc[0],init_loc[1],int(action_index-len(move_set))]

                # Higher order Weight Matrices
                W_o = W_option[:,:,:,int(option_index)]
                V_o = V_option[:,:,int(option_index)]
                S_Phase[:,:,:,0] = (2*np.random.random_sample((2,states.shape[0],states.shape[1])))-1   # random starting points processing module
                A_Phase_top[:,:,0] = (2*np.random.random_sample((2,n_moves)))-1  # idem
                #Reset cumulative reward list
                r_cum_list = []
                #Get termination state
                termination_state = termination_set[int(option_index)]
                """Higher order action sequence"""
                while current_loc != termination_state:
                    if n_o_steps>0:
                        #first action choice on based on root level
                        S_Phase[:,:,:,0] = S_Phase[:,:,:,total_time-1]  # random starting points processing module
                        A_Phase_top[:,:,0] = A_Phase_top[:,:,total_time-1] # idem
                    #phase reset
                    MFC[:,0]=np.ones((2))*r2_MFC 

                    # LFC setting instruction per step: each state is an input
                    LFC[current_loc[0],current_loc[1],n_steps] = 1

                    action_to_sync = softmax_action(action_weights = W_o[current_loc[0],current_loc[1],:],tau = tau)
                
                    LFC_sync = int(action_to_sync) #Which action does LFC sync to current state

                    LFC_desync = list(range(len(moves)))
                    LFC_desync.pop(LFC_sync)     # List of actions the LFC desynchronizes


                    # The actor makes the move #
                    for t in range(total_time-1):

                        
                        #Update phase code neurons for actions and states in processing module
                        #State phase code neurons   
                        S_Phase[:,:,:,t+1] = update_phase(nodes=S_Phase[:,:,:,t], grid = True, radius=r2_max, damp = damp, coupling = cg_1,multiple=True )
                        
                        #Action phase code neurons
                        A_Phase_top[:,:,t+1] = update_phase(nodes=A_Phase_top[:,:,t], grid = False, radius=r2_max, damp = damp, coupling = cg_2,multiple=True )

                        #Update phase code untis of MFC
                        MFC[:,t+1] = update_phase(nodes=MFC[:,t], grid = False, radius=r2_MFC, damp=damp_MFC, coupling=ct,multiple=False)

                        #MFC rate code neuron-> Bernoulli process

                        Be = 1/(1 + np.exp(-acc_slope*(MFC[0,t]-1.33)))    # Bernoulli process 
                        #Bernoulli[tijd,step,trial] = Be                                 # logging Be value

                        p = random.random()

                        if p < Be:

                            Gaussian = np.random.normal(size = [1,2])  #noise factor as normal distribution
                            #Hit[tijd,step,trial] = 1
                            
                                
                            x, y = current_loc[1], current_loc[0]

                            #the LFC decides which state is paired with which actions

                            if LFC[y,x,n_steps]:
                                #The state the actor is in receives a burst because it is the only input
                                S_Phase[:,y,x,t+1] = decay*S_Phase[:,y,x,t] + Gaussian

                                # and all the actions that are to be synchronized to that state receive a burst
                                
                                if type(LFC_sync) is int:
                                    A_Phase_top[:,LFC_sync,t+1] = decay*A_Phase_top[:,LFC_sync,t] + Gaussian
                                # Desynchronize all other actions !
                                for node in LFC_desync:
                                    A_Phase_top[:,int(node),t+1] = decay*A_Phase_top[:,int(node),t] - Gaussian*noise

                        S_Rate[current_loc[0],current_loc[1],t]= (1/(1+np.exp(-5*S_Phase[0,current_loc[0],current_loc[1],t]-0.6)))
                        A_Rate_top[:,t]=(S_Rate[current_loc[0],current_loc[1],t]*(W_o[current_loc[0],current_loc[1],:]+1))*(1/(1+np.exp(-5*A_Phase_top[0,:,t]-0.6)))
                    
                    # select action
                    action_index = int(np.argmax(np.sum(A_Rate_top[:,:],axis=1)))

                    """""""""""""""
                        Learning
                    """""""""""""""
                    new_loc = update_location(grid=states,loc=current_loc,move=moves[action_index])


                    y, x, y_p, x_p, a = current_loc[0], current_loc[1], new_loc[0], new_loc[1], action_index #location coordinates


                    if new_loc == termination_state:
                        sub_reward = states[y_p][x_p]
                    else: 
                        sub_reward = 0
                    
                    #Update Weights
                    delta = sub_reward + gamma*V_o[y_p][x_p] - V_o[y][x]

                    V_o[y][x] = V_o[y][x] + alpha*delta
                    W_o[y][x][a] = W_o[y][x][a] + beta*delta                
                    
                    
                    n_steps+=1
                    n_o_steps+=1
                    current_loc = new_loc #update current location

                    """Root-level stuff"""
                    if states[new_loc[0],new_loc[1]] == reward_size:
                        at_goal = True
                        reward = reward_size
                    else:
                        reward = 0
                    


                    # Log the reward for r_cum (hihi)
                    r_cum_list.append((gamma**(n_o_steps-1)*reward))
                    r_cum = np.sum(r_cum_list)
                    #End of option when at goal
                    
                    if at_goal:
                        break
                    if n_steps == max_steps:
                        break

                
                # Root Level learning
                y,x,y_p,x_p,a = init_loc[0],init_loc[1],new_loc[0],new_loc[1], init_action

                delta = r_cum + (gamma**n_o_steps)*V[y_p][x_p] - V[y][x]
            
                V[y][x] = V[y][x] + alpha*delta

                W[y][x][a] = W[y][x][a] + beta*delta
            
                #Log option Weights back in
                W_option[:,:,:,int(option_index)] = W_o
                V_option[:,:,int(option_index)] = V_o    

                #Log additional things
                o_step_list[int(option_index)].append(n_o_steps)
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
            if n_steps == max_steps:
                #print("didn't make it to goal :(")
                break
        ##############################################################################################
        """After the episode"""
        #print(current_loc)
        #if at_goal:
            #print("I am at goal location!")
        trial_steps[trial] = n_steps
        #print("I took {0} options and {1} steps this episode".format(option_choice,n_steps))

    time1 = time.perf_counter()
    print("For the fourth model I took {} minutes".format((time1-time0)/60))
    return trial_steps, V #, V_option,n_o_steps



if __name__ == "__main__":
          
    n_trials= 300
    mapname = '/home/isaiah/Documents/Thesis/Code/Models/Maps/map4.txt'



    n_sim = 11

    with concurrent.futures.ProcessPoolExecutor(max_workers=11) as executor:
        #Results are logged as objects into a list            
        results = [executor.submit(run_HRL_sync,mapname, n_trials=n_trials,tau = 5, seed= seed) for seed in range(n_sim)]
        #results = [executor.submit(get_random_number,i) for i in range(n_sim)]
        #Retrieve 'actual' results from futures object
        results = [f.result() for f in results]

    sims_steps = np.zeros((n_sim,n_trials))
    sim =0
    for result in results:
        sims_steps[sim,:] = result[0]
        sim+=1


    # plt.figure("Value maps of root-level")
    # plt.imshow(V, cmap = 'hot', interpolation = 'nearest')
    # plt.show()


    # for i in range(8):
    #     plt.figure("Value maps of learned options")
    #     plt.subplot(4,4,i+1)
    #     plt.imshow(V_option[:,:,i], cmap = 'hot', interpolation = 'nearest')
    # plt.show()

    # for i in range(8):
    #     plt.figure("Steps made within each option")
    #     plt.subplot(4,4,1+i)
    #     plt.plot(range(len(n_o_steps[i])),n_o_steps[i])
    # plt.show()


    #trial_length1, V1, V1_option, n_osteps = run_HRL_map(mapname, n_trials=n_trials, seed=2,HRL=False)

    #plt.plot(range(n_trials), trial_length1,"c-")
    
    plt.plot(range(n_trials),sims_steps.mean(axis=0),"r-")
    plt.title("Number of steps per trial")
    plt.show()


# for i in range(len(option_set)):
#     plt.figure("Value maps of learned options")
#     plt.subplot(4,4,i+1)
#     plt.imshow(V_option[:,:,i], cmap = 'hot', interpolation = 'nearest')
# plt.show()


# plt.figure("Value maps of root-level")
# plt.imshow(V, cmap = 'hot', interpolation = 'nearest')
# plt.show()

# plt.plot(range(n_trials),trial_steps,"--")
# plt.title("Number of steps per trial")
# plt.show()




