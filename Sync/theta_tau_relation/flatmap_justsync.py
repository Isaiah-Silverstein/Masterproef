import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd
from flatmap_functions import *

""" F L A T M A P    S Y N C """
### MAIN PARAMETERS ###

reward_size = 20
map_size = 10

n_trials = 60
n_steps = 100

srate = 500  #sample rate

total_time = int(1.5*srate)  #total timesteps or "time the agent gets to think about moving"

r2_values = np.linspace(0,1,20)
alpha_values = np.linspace(0.25,0.3,2)
beta_values = np.linspace(0.25,0.3,2)
measures = np.array(["Exploration Steps", "Total Steps"])
Datatable = np.zeros((len(alpha_values),len(beta_values),len(r2_values)))



a=0
b=0
r=0
for alpha in alpha_values:
    
    b=0
    for beta in beta_values:
        r=0
        for r2_acc in r2_values:

            # Learning Parameters
            parameters = {"alpha": alpha
                        ,"beta": beta
                        ,"gamma": 0.9}

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
            S_Phase = np.zeros((2,states.shape[0],states.shape[1],total_time,n_steps,n_trials)) #State phase code units
            A_Phase = np.zeros((2,n_actions,total_time,n_steps,n_trials)) #Action phase code units

            #Setting up rate code neurons across entire task
            S_Rate = np.zeros((states.shape[0],states.shape[1],total_time,n_steps,n_trials)) #State rate code units
            A_Rate = np.zeros((n_actions,total_time,n_steps,n_trials)) #Action rate code units
            #State-Action Weight Matrix
            W = np.ones((states.shape[0],states.shape[1],n_actions))*0.5   #initial state-action weights
            V = np.ones((states.shape[0],states.shape[1]))*0.5             #initial state weights

            """""""""""""""""""""
            Control   module
            """""""""""""""""""""

            # MFC #
            # Initial variables 
            r2_MFC = r2_acc          #maximum amplitude MFC node
            damp_MFC = 0.03       # damping parameter MFC
            acc_slope = 10        # MFC slope parameter ---> steepness of burst probability distribution
            ct = (5/srate)*2*np.pi #theta band coupling parameter for MFC

            #Setting up phase code nodes for the MFC
            MFC = np.zeros((2,total_time,n_steps,n_trials))
            #Setting up phase code neuron for MFC -> Bernoulli rate code
            Be = 0       
            """When the be value as the rate code of MFC
            reaches certain threshold the MFC will send a burst to coupled neurons"""

            # LFC #

            #Module indicating which actions should be coupled to which state
            LFC = np.zeros((states.shape[0],states.shape[1],n_steps))

            #Module that gives the right indices to synchronize
            LFC_sync = np.zeros((states.shape[0],states.shape[1],n_actions))

            LFC_sync[:,:,:] = action_set   # For now we're just going to synchronize all actions


            # # # # # # # # # # # # # #
            # #      Simulation     # #
            # # # # # # # # # # # # # #

            # Logging dependent variables
            Hit = np.zeros((total_time,n_steps,n_trials))  #log when there is a burst from the MFC
            Goal_reach  = np.zeros((n_steps,n_trials))     #record if goal is reached 
            Move = np.zeros((n_steps,n_trials))            #record move
            Bernoulli = np.zeros((total_time,n_steps,n_trials)) #Logging the bernoulli process variables (should be in between -.8 and .8)
            pred_err = np.zeros((states.shape[0],states.shape[1],n_steps,n_trials)) #logging the prediction error
            trial_length = []

            # Recording sync
            sync = np.zeros((n_states,n_actions,n_steps,n_trials)) 




            """ L O O P """

            greedy, exploration = 0 ,0 
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
                    S_Phase[:,:,:,0,step,trial] = np.random.random_sample((2,states.shape[0],states.shape[1]))   # random starting points processing module
                    A_Phase[:,:,0,step,trial] = np.random.random_sample((2,n_actions))  # idem
                                    
                    #phase reset
                    MFC[:,0,step,trial]=np.ones((2))*r2_MFC 
                    # The actor makes the move #
                    for time in range(total_time-1):
                        

                        # LFC setting instruction
                        LFC[current_loc[0],current_loc[1],step] = 1

                        #Update phase code neurons for actions and states in processing module
                        #State phase code neurons
                        S_Phase[:,:,:,time+1,step,trial] = update_phase(nodes=S_Phase[:,:,:,time,step,trial], grid = True, radius=r2_max, damp = damp, coupling = cg_1,multiple=True )
                        
                        #Action phase code neurons
                        A_Phase[:,:,time+1,step,trial] = update_phase(nodes=A_Phase[:,:,time,step,trial], grid = False, radius=r2_max, damp = damp, coupling = cg_2,multiple=True )

                        #Update phase code untis of MFC
                        MFC[:,time+1,step,trial] = update_phase(nodes=MFC[:,time,step,trial], grid = False, radius=r2_MFC, damp=damp_MFC, coupling=ct,multiple=False)

                        #MFC rate code neuron-> Bernoulli process

                        Be = 1/(1 + np.exp(-acc_slope*(MFC[0,time,step,trial]-1.33)))    # Bernoulli process 
                        Bernoulli[time,step,trial] = Be                                 # logging Be value

                        p = random.random()

                        if p < Be:

                            Gaussian = np.random.normal(size = [1,2])  #noise factor as normal distribution
                            Hit[time,step,trial] = 1
                            
                            #For all states in grid map
                            for y in range(states.shape[0]):
                                for x in range(states.shape[1]):
                                
                                    #If the states are coded in the LFC (for now this is just the current state)
                                    if LFC[y,x,step]:
                                        #The states receive a burst
                                        S_Phase[:,y,x,time+1,step,trial] = decay*S_Phase[:,y,x,time,step,trial] + Gaussian

                                        # and all the actions that are to be synchronized to that state receive a burst
                                        for nodes in LFC_sync[current_loc[0],current_loc[1],:]:
                                            A_Phase[:,int(nodes),time+1,step,trial] = decay*A_Phase[:,int(nodes),time,step,trial] + Gaussian
                                    
                                    # For desynchronization it would probably look something like this
                                    #elif LFC[y,x,trial] == -1:
                                        # #The irrelevant states receive a negative burst
                                        # S_Phase[:,y,x,time+1,trial] = decay*S_Phase[:,y,x,time,trial] - Gaussian

                                        # # and all the irrelevant actions that are to be desynchronized to that state receive a negative burst
                                        # for nodes in LFC_sync[current_loc[0],current_loc[1],:]
                                        #     A_Phase[:,int(nodes),time+1,trial] = decay*A_Phase[:,int(nodes),time,trial] - Gaussian
                    
                        #Updating rate code units
                        #Only the rate code neuron of a single state is updated because the actor can only be in one place at the same time
                        S_Rate[current_loc[0],current_loc[1],time,step,trial]= (1/(1+np.exp(-5*S_Phase[0,current_loc[0],current_loc[1],time,step,trial]-0.6)))
                        A_Rate[:,time,step,trial]=(S_Rate[current_loc[0],current_loc[1],time,step,trial]*W[current_loc[0],current_loc[1],:])*(1/(1+np.exp(-5*A_Phase[0,:,time,step,trial]-0.6)))


                    t = time
                    """""""""""""""
                    Learning
                    """""""""""""""
                    # select action
                    action_index = int(np.argmax(np.sum(A_Rate[:,:,step,trial],axis=1)))
                    if action_index != int(np.argmax(W[current_loc[0],current_loc[1],:])):
                        exploration += 1
                        print("yes")
                    else:
                        greedy += 1
                    #update location
                    new_loc= update_location(grid = states, loc=current_loc,move = moves[action_index])

                    #log coordinates for weight matrices
                    coordinates = [current_loc[0], current_loc[1], new_loc[0], new_loc[1], action_index] #location coordinates

                    #update weights according to TD-learning
                    V, W, delta, at_goal = update_weights(param=parameters, index=coordinates, V=V, W=W, states=states)


                    if at_goal:
                        trial_length.append(step)
                        steps_last_trial = step
                        break
                    
                    current_loc = new_loc

                    if (step+1) == n_steps:
                        trial_length.append(step)
                        steps_last_trial = step
                        
                
                print(greedy, exploration)
    #alpha
        #beta
            #tau
            Datatable[a,b,r] = exploration
            
            r+=1

        b+=1
    a+=1    





midx = pd.MultiIndex.from_product([alpha_values, beta_values, r2_values])
long_data = np.zeros((len(midx)))
i=0
for a in range(len(alpha_values)):
    for b in range(len(beta_values)):
        for r in range(len(r2_values)):
                
                long_data[i] = Datatable[a,b,r]
                i+=1


export = pd.DataFrame(long_data, index =midx)
export.to_csv("Radius_Values")
 



