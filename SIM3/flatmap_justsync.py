from flatmap_functions import *
import numpy as np
import random, itertools,time
import pandas as pd
import multiprocessing, concurrent.futures

start = time.perf_counter()

""" F L A T M A P    S Y N C """

### PARAMETER VALUES ACROSS SIMULATIONs ###

radius_values = np.linspace(0.01,0.99,20)
alpha_values = np.linspace(0.1,0.9,10)
beta_values = np.linspace(0.1,0.9,10)


paramlist = list(itertools.product(alpha_values,beta_values,radius_values))


### MAIN SIMULATION PARAMETERS ###

reward_size = 100
map_size = 10

n_trials = 100
n_steps = 400

srate = 500  #sample rate

total_time = int(1.5*srate)  #total timesteps or "time the agent gets to think about moving"


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

n_states = len(state_set)
n_actions= len(action_set)

"""""""""""""""""""""
Control   module
"""""""""""""""""""""

# MFC #
# Initial variables 

damp_MFC = 0.03       # damping parameter MFC
acc_slope = 10        # MFC slope parameter ---> steepness of burst probability distribution
ct = (5/srate)*2*np.pi #theta band coupling parameter for MFC



def run_radius_sim(param_tuple = tuple):
    """ Main simulation function. It accepts a tuple of relevant parameter values and spits out relevant results"""

    #Insert input tuple into 'readable' dictionary
    parameters = {"alpha": param_tuple[0]
            ,"beta": param_tuple[1]
            ,"gamma": 0.9
            ,"r2_acc": param_tuple[2]}

    print(parameters)

    

    """Processing Module"""
    #Setting up phase code neurons across entire task
    S_Phase = np.zeros((2,states.shape[0],states.shape[1],total_time)) #State phase code units
    A_Phase = np.zeros((2,n_actions,total_time)) #Action phase code units

    #Setting up rate code neurons across entire task
    S_Rate = np.zeros((states.shape[0],states.shape[1],total_time)) #State rate code units
    A_Rate = np.zeros((n_actions,total_time)) #Action rate code units
    #State-Action Weight Matrix
    W = np.zeros((states.shape[0],states.shape[1],n_actions))  #initial state-action weights
    V = np.zeros((states.shape[0],states.shape[1]))             #initial state weights

    """Control Module"""

    # MFC # 
    r2_MFC = parameters["r2_acc"]         #maximum amplitude MFC node

    #Setting up phase code nodes for the MFC
    MFC = np.zeros((2,total_time))
    #Setting up phase code neuron for MFC -> Bernoulli rate code
    Be = 0       


    # LFC #

    #Module indicating which actions should be coupled to which state
    LFC = np.zeros((states.shape[0],states.shape[1],n_steps))

    LFC_sync = 0   # For now we're just going to leave the action to sync as 0

    """ Logging dependent variables """ 
    Hit = np.zeros((total_time))  #log when there is a burst from the MFC
    #Goal_reach  = np.zeros((n_steps,n_trials))     #record if goal is reached 
    #Move = np.zeros((n_steps,n_trials))            #record move
    Bernoulli = np.zeros((total_time)) #Logging the bernoulli process variables (should be in between -.8 and .8)
    #pred_err = np.zeros((states.shape[0],states.shape[1],n_steps,n_trials)) #logging the prediction error
    trial_length = []

    # Recording sync
    #sync = np.zeros((n_states,n_actions,n_steps,n_trials)) 


    # # # # # # # # # # # # # #
    # #      Simulation     # #
    # # # # # # # # # # # # # #

    exploration = 0
    greedy =0
    sync_fail =0
    for trial in range(n_trials):
        """A trial is considered as each journey the actor makes until the goal
            or until it runs out of steps"""
        
        start_loc = [1,int(states.shape[1]-2)]   #start in the top left
        
        for step in range(n_steps):

            #starting location at first trial
            if step == 0:
                current_loc = start_loc
            """""""""""""""""""""
            Synchronization
            """""""""""""""""""""
            S_Phase[:,:,:,0] = np.random.random_sample((2,states.shape[0],states.shape[1]))   # random starting points processing module
            A_Phase[:,:,0] = np.random.random_sample((2,n_actions))  # idem
                            
            #phase reset
            MFC[:,0]=np.ones((2))*r2_MFC 


            # LFC setting instruction
            LFC[current_loc[0],current_loc[1],step] = 1

            # What we want is the lfc to indicate the state and then have the LFC sync pro actively select an action based on state action value maps
            action_to_sync = greedy_action(action_weights=W[current_loc[0],current_loc[1],:])
            LFC_sync = int(action_to_sync)
            LFC_desync = list(range(len(moves)))
            LFC_desync.pop(LFC_sync)    

            # The actor makes the move #
            for time in range(total_time-1):
            
                #Update phase code neurons for actions and states in processing module
                #State phase code neurons
                S_Phase[:,:,:,time+1] = update_phase(nodes=S_Phase[:,:,:,time], grid = True, radius=r2_max, damp = damp, coupling = cg_1,multiple=True )
                
                #Action phase code neurons
                A_Phase[:,:,time+1] = update_phase(nodes=A_Phase[:,:,time], grid = False, radius=r2_max, damp = damp, coupling = cg_2,multiple=True )

                #Update phase code untis of MFC
                MFC[:,time+1] = update_phase(nodes=MFC[:,time], grid = False, radius=r2_MFC, damp=damp_MFC, coupling=ct,multiple=False)

                #MFC rate code neuron-> Bernoulli process

                Be = 1/(1 + np.exp(-acc_slope*(MFC[0,time]-1.33)))    # Bernoulli process 
                Bernoulli[time] = Be                                 # logging Be value

                p = random.random()

                if p < Be:

                    Gaussian = np.random.normal(size = [1,2])  #noise factor as normal distribution
                    Hit[time] = 1
                    
                    x, y = current_loc[1], current_loc[0]
                        
                    #If the states are coded in the LFC (for now this is just the current state)
                    if LFC[y,x,step]:
                        #The states receive a burst
                        S_Phase[:,y,x,time+1] = decay*S_Phase[:,y,x,time] + Gaussian
                        

                        # and all the actions that are to be synchronized to that state receive a burst
                        if type(LFC_sync) is int:
                            A_Phase[:,LFC_sync,time+1] = decay*A_Phase[:,LFC_sync,time] + Gaussian
                        # Desynchronize all other actions !
                        for node in LFC_desync:
                            A_Phase[:,int(node),time+1] = decay*A_Phase[:,node,time] - Gaussian*noise
                        # elif type(LFC_sync) is list:
                        #     for node in LFC_sync:
                        #         A_Phase[:,int(node),time+1] = decay*A_phase[:,int(node),time] + Gaussian #for when multiple actions need to be synced
            
                #Updating rate code units
                #Only the rate code neuron of a single state is updated because the actor can only be in one place at the same time
                S_Rate[current_loc[0],current_loc[1],time]= (1/(1+np.exp(-5*S_Phase[0,current_loc[0],current_loc[1],time]-0.6)))
                #A_Rate[:,time]=(S_Rate[current_loc[0],current_loc[1],time]*(W[current_loc[0],current_loc[1],:]+1))*(1/(1+np.exp(-5*A_Phase[0,:,time]-0.6)))
                A_Rate[:,time]=(S_Rate[current_loc[0],current_loc[1],time])*(1/(1+np.exp(-5*A_Phase[0,:,time]-0.6)))

            """""""""""""""
            Learning
            """""""""""""""
            # select action
            action_index = int(np.argmax(np.sum(A_Rate[:,:],axis=1)))

            if action_index in np.where(W[current_loc[0],current_loc[1],:] == max(W[current_loc[0],current_loc[1],:]))[0]:
                greedy+=1
            else:
                exploration+=1
            if action_index != LFC_sync:
                sync_fail+=1

            #update location
            new_loc= update_location(grid = states, loc=current_loc,move = moves[action_index])

            #log coordinates for weight matrices
            coordinates = [current_loc[0], current_loc[1], new_loc[0], new_loc[1], action_index] #location coordinates

            #update weights according to TD-learniaction_mismatchng
            V, W, delta, at_goal = update_weights(param=parameters, index=coordinates, V=V, W=W, states=states)


            if at_goal:
                trial_length.append(step)
                break
            
            current_loc = new_loc
            if (step+1) == n_steps:
                trial_length.append(step)
    return [parameters["alpha"],parameters["beta"],parameters["r2_acc"],exploration, sync_fail], V, trial_length


# """Create Multiprocessing pool that alocates loop iterations across cores according to hardware"""
# # for limiting the amount of cores, insert key argument ProcessPoolExecutor(max_workers= "number of cores") 
# with concurrent.futures.ProcessPoolExecutor() as executor:
#     #Results are logged as objects into a list
#     results = [executor.submit(run_radius_sim,param_tuple) for param_tuple in paramlist]
#     #Retrieve 'actual' results from futures object
#     results = [f.result() for f in results]

# """Data exports"""
# data_export = pd.DataFrame(results, columns = ['Alpha','Beta','Radius','Exploration','Sync Rate'])
# data_export.to_csv("Radius_Values")

import matplotlib.pyplot as plt

paratuple = (0.99,0.99,0.1)

results, V, trial_length = run_radius_sim(paratuple)

print(results)
print(np.sum(trial_length))

plt.imshow(V, cmap = 'hot',interpolation = 'nearest')
plt.show()

plt.plot(range(len(trial_length)),trial_length)
plt.show()


finish = time.perf_counter()
print("Script took {} sec to finish".format(finish-start))