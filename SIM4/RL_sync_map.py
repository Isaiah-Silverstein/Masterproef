import matplotlib.pyplot as plt
import numpy as np
import random,os
import time
from RL_map import run_map
from flatmap_functions import *

""" F L A T M A P    S Y N C """
### MAIN PARAMETERS ###

def run_RL_sync(mapname,n_trials = int, seed = int,alpha = 0.15, beta = 0.2, tau = 5, gamma = 0.9, max_steps = 1000, reward_size = 100):
    """ Sync model completing the map"""

    # Softmax can't be from external file, because multiprocessing messes up the seed values
    np.random.seed(seed)
    def softmax_action(action_weights = [], tau = int):
        action_indices = list(range(len(action_weights)))
        f = np.exp((action_weights - np.max(action_weights))/tau)  # shift values
        action_prob = f / f.sum(axis=0)
        action_index = np.random.choice(action_indices, 1, p=action_prob)
        return action_index[0]

    srate = 500  #sample rate
        
    total_time = int(1.5*srate)  #total timesteps or "time the agent gets to think about moving"

    time0 = time.perf_counter()

    print("Running the RL model but with sync !")
    srate = 500  #sample rate
        
    total_time = int(1.5*srate)  #total timesteps or "time the agent gets to think about moving"

    # Learning Parameters
    parameters = {"alpha": alpha
                ,"beta": beta
                ,"gamma": gamma
                ,"tau": tau}
    n_steps = max_steps
    n_trials = n_trials
    
    sub_reward_size = 0 # no subgoals!
    # # # # # # # # # # # # # #
    # # Setting up the map  # #
    # # # # # # # # # # # # # #
    """ The agent begins in a walled grid and has to find 
        the goal to obtain a reward."""
    # Grid #
    states = create_grid_from_file(map_file=mapname,goal_location = [10,3],reward_size=reward_size,sub_reward_size=sub_reward_size)
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

    #Setting up phase code neurons across entire task
    S_Phase = np.zeros((2,states.shape[0],states.shape[1],total_time)) #State phase code units
    A_Phase = np.zeros((2,n_actions,total_time)) #Action phase code units

    #Setting up rate code neurons across entire task
    S_Rate = np.zeros((states.shape[0],states.shape[1],total_time)) #State rate code units
    A_Rate = np.zeros((n_actions,total_time)) #Action rate code units
    #State-Action Weight Matrix
    W = np.zeros((states.shape[0],states.shape[1],n_actions))#*0.001   #initial state-action weights
    V = np.zeros((states.shape[0],states.shape[1]))#*0.001             #initial state weights

    """""""""""""""""""""
    Control   module
    """""""""""""""""""""

    # MFC #
    # Initial variables 
    r2_MFC = 0.7         #maximum amplitude MFC node
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
    LFC = np.zeros((states.shape[0],states.shape[1],n_steps))

    #Module that gives the right indices to synchronize
    LFC_sync = 0



    # # # # # # # # # # # # # #
    # #      Simulation     # #
    # # # # # # # # # # # # # #

    # Logging dependent variables
    Hit = np.zeros((total_time,n_steps,n_trials))  #log when there is a burst from the MFC
    # Goal_reach  = np.zeros((n_steps,n_trials))     #record if goal is reached 
    # Move = np.zeros((n_steps,n_trials))            #record move
    # Bernoulli = np.zeros((total_time,n_steps,n_trials)) #Logging the bernoulli process variables (should be in between -.8 and .8)
    # pred_err = np.zeros((states.shape[0],states.shape[1],n_steps,n_trials)) #logging the prediction error
    trial_length = np.zeros((n_trials))

    # Recording sync
    sync = np.zeros((n_states,n_actions,n_steps,n_trials)) 

    """ L O O P """

    exploration = 0
    exploration_intent =0
    sync_fail=0
    greedy=0
    for trial in range(n_trials):
        """A trial is considered as each journey the actor makes until the goal
            or until it runs out of steps"""
        at_goal = False
        start_loc = [1,int(states.shape[1]-2)]   #start in the top left
        step = 0 
        S_Phase[:,:,:,0] = (2*np.random.random_sample((2,states.shape[0],states.shape[1])))-1   # random starting points processing module
        A_Phase[:,:,0] = (2*np.random.random_sample((2,n_actions)))-1  # idem
        while not at_goal:
            #starting location at first trial
            if step == 0:
                current_loc = start_loc
            else:
                S_Phase[:,:,:,0] = S_Phase[:,:,:,total_time-1]   # random starting points processing module
                A_Phase[:,:,0] = A_Phase[:,:,total_time-1] # idem
            """""""""""""""""""""
            Synchronization
            """""""""""""""""""""

            
            #phase reset
            MFC[:,0]=np.ones((2))*r2_MFC 


            # LFC setting instruction per step: each state is an input
            LFC[current_loc[0],current_loc[1],step] = 1

            # What we want is the lfc to indicate the state and then have the LFC sync pro actively select an action based on state action value maps
            
            action_to_sync = softmax_action(action_weights=W[current_loc[0],current_loc[1],:],tau=10)
            if action_to_sync in np.where(W[current_loc[0],current_loc[1],:]== max(W[current_loc[0],current_loc[1],:]))[0]:
                greedy+=0
            else:
                exploration_intent+=1
                              
    
            #Which action does LFC sync to current state
            LFC_sync = int(action_to_sync)
            LFC_desync = list(range(len(moves)))
            LFC_desync.pop(LFC_sync)     

            # The actor makes the move #
            for t in range(total_time-1):

                
                #Update phase code neurons for actions and states in processing module
                #State phase code neurons   
                S_Phase[:,:,:,t+1] = update_phase(nodes=S_Phase[:,:,:,t], grid = True, radius=r2_max, damp = damp, coupling = cg_1,multiple=True )
                
                #Action phase code neurons
                A_Phase[:,:,t+1] = update_phase(nodes=A_Phase[:,:,t], grid = False, radius=r2_max, damp = damp, coupling = cg_2,multiple=True )

                #Update phase code untis of MFC
                MFC[:,t+1] = update_phase(nodes=MFC[:,t], grid = False, radius=r2_MFC, damp=damp_MFC, coupling=ct,multiple=False)

                #MFC rate code neuron-> Bernoulli process

                Be = 1/(1 + np.exp(-acc_slope*(MFC[0,t]-1)))    # Bernoulli process 
                #Bernoulli[time,step,trial] = Be                                 # logging Be value

                p = random.random()

                if p < Be:

                    Gaussian = np.random.normal(size = [1,2])  #noise factor as normal distribution
                    #Hit[tijd,step,trial] = 1
                    
                        
                    x, y = current_loc[1], current_loc[0]

                    #the LFC decides which state is paired with which actions

                    if LFC[y,x,step]:
                        #The state the actor is in receives a burst because it is the only input
                        S_Phase[:,y,x,t+1] = decay*S_Phase[:,y,x,t] + Gaussian

                        # and all the actions that are to be synchronized to that state receive a burst
                        if type(LFC_sync) is int:
                            A_Phase[:,LFC_sync,t+1] = decay*A_Phase[:,LFC_sync,t] + Gaussian
                        
                        # Desynchronize all other actions !
                        for node in LFC_desync:
                            A_Phase[:,int(node),t+1] = decay*A_Phase[:,int(node),t] - Gaussian*noise

                #Updating rate code units
                #Only the rate code neuron of a single state is updated because the actor can only be in one place at the same time
                S_Rate[current_loc[0],current_loc[1],t]= (1/(1+np.exp(-5*S_Phase[0,current_loc[0],current_loc[1],t]-0.6)))
                A_Rate[:,t]=(S_Rate[current_loc[0],current_loc[1],t]*(W[current_loc[0],current_loc[1],:]+1))*(1/(1+np.exp(-5*A_Phase[0,:,t]-0.6)))
                #A_Rate[:,t]=(S_Rate[current_loc[0],current_loc[1],t])*(1/(1+np.exp(-5*A_Phase[0,:,t]-0.6)))
            
            # select action
            action_index = int(np.argmax(np.sum(A_Rate[:,:],axis=1)))
            if action_index in np.where(W[current_loc[0],current_loc[1],:] == max(W[current_loc[0],current_loc[1],:]))[0]:
                greedy+=1
            else:
                exploration+=1

            if action_index != LFC_sync:
                sync_fail+=1
        
            """""""""""""""
            Learning
            """""""""""""""

            #update location
            new_loc= update_location(grid = states, loc=current_loc,move = moves[action_index])

            #log coordinates for weight matrices
            coordinates = [current_loc[0], current_loc[1], new_loc[0], new_loc[1], action_index] #location coordinates

            #update weights according to TD-learning
            V, W, delta, at_goal = update_weights(param=parameters, index=coordinates, V=V, W=W, states=states, reward_size = reward_size)


            #update_location
            current_loc = new_loc
            step+=1
            if step ==n_steps:
                #print("Agent did not reach goal")
                break
        
        trial_length[trial] = step   
        
    print("I took {0} exploratory steps and {1} greedy steps this simulation".format(exploration,greedy))
    print("I intended to explore {} times".format(exploration_intent))
    print("Sync of correct action failed {} times".format(sync_fail))
    print("In this sim I took   a total {} steps".format(np.sum(trial_length)))
    
    time1 = time.perf_counter()
    print("For the second model I took {} minutes".format((time1-time0)/60))
    return trial_length, V





if __name__ == '__main__':
    mapname = '/home/isaiah/Documents/Thesis/Code/Models/Maps/map1.txt'

    trial_length, V  = run_RL_sync(mapname,n_trials =50, seed =1, alpha = 0.15, beta = 0.2, tau = 3, gamma = 0.9, max_steps = 200, reward_size = 100)
    trial_length0, V0  = run_RL_sync(mapname,n_trials =50 ,seed=2, alpha = 0.15, beta = 0.2, tau = 0.5, gamma = 0.9, max_steps = 200, reward_size = 100)
    trial_length1, V1=  run_map(mapname,n_trials = 50,seed =3 , alpha = 0.15, beta = 0.2, tau = 3, gamma = 0.9, max_steps = 200, reward_size = 100)

    fig, axs = plt.subplots(3)
    axs[0].imshow(V, cmap = 'hot', interpolation = 'nearest')
    axs[0].title.set_text("State Value heat map")
    axs[1].plot(np.arange(len(trial_length)),trial_length, "r-")
    axs[1].plot(np.arange(len(trial_length0)),trial_length0, "g-")
    axs[1].plot(np.arange(len(trial_length1)),trial_length1,"m-")
    axs[1].title.set_text("Number of steps per trial")
    axs[2].imshow(V1, cmap = 'hot', interpolation = 'nearest')
    axs[2].title.set_text("State Value heat RL map")
    plt.tight_layout()
    plt.show()


# for i in range(n_actions):
#     #action
#     fig, axs = plt.subplots(2)
#     axs[0].title.set_text('Phase code units of starting state and action node number {}'.format(move_name[i]))
#     axs[0].plot(np.arange(int(total_time)),S_Phase[0,current_loc[0],current_loc[1],:])
#     axs[0].plot(np.arange(int(total_time)),A_Phase[0,i,:])
#     axs[0].plot(np.arange(int(total_time)),Hit[:,step,trial])
#     axs[1].plot(np.arange(int(total_time)), S_Rate[current_loc[0],current_loc[1],:],"r-")
#     axs[1].plot(np.arange(int(total_time)), A_Rate[i,:],"b-")
#     plt.tight_layout()
#     plt.show()



    
# fig, axs = plt.subplots(2)
# axs[0].plot(np.arange(int(total_time)),Bernoulli[:,step,trial])
# axs[0].title.set_text("Bernoulli values")
# axs[1].plot(np.arange(int(total_time)),-acc_slope*(MFC[0,:,step,trial]-1.5))
# axs[1].title.set_text("MFC amplitude")
# plt.tight_layout()
# plt.show()
    



