import matplotlib.pyplot as plt
import random 
import numpy as np

""" F L A T M A P    S Y N C """

# # # # # # # # # # # # # #
# #      Functions      # #
# # # # # # # # # # # # # #

def update_phase(nodes = [], grid = True, radius = 1, damp = 0.3, coupling = 0.3, multiple = True):
    """ updating the phase per step"""
    if multiple:
        if grid:
            Phase = np.zeros((2,nodes.shape[1],nodes.shape[2]))
            r2 = np.sum(nodes*nodes,axis = 0)
            E = nodes[0,:,:]
            I = nodes[1,:,:]
            #excitatory update formula
            """E_i(t + dt) = E_i(t) - C * I_i(t) - D * J(r > r_min) * E_i(t)""" 
            Phase[0,:,:] = E - coupling*I - damp*((r2>radius).astype(int))*E 
            """I_i(t + dt) = I_i(t) + C * E_i(t) - D * J(r > r_min) * I_i(t)"""
            Phase[1,:,:] = I + coupling*E - damp*((r2>radius).astype(int))*I

        else:
            Phase = np.zeros((2,len(nodes[0,:])))
            r2 = np.sum(nodes*nodes,axis = 0)
            E = nodes[0,:]
            I = nodes[1,:]
            #excitatory update formula
            """E_i(t + dt) = E_i(t) - C * I_i(t) - D * J(r > r_min) * E_i(t)""" 
            Phase[0,:] = E - coupling*I - damp*((r2>radius).astype(int))*E 
            """I_i(t + dt) = I_i(t) + C * E_i(t) - D * J(r > r_min) * I_i(t)"""
            Phase[1,:] = I + coupling*E - damp*((r2>radius).astype(int))*I

    else:
        Phase = np.zeros((2))
        r2 = np.sum(nodes[0]*nodes[1],axis = 0)
        E = nodes[0]
        I = nodes[1]
        #excitatory update formula
        """E_i(t + dt) = E_i(t) - C * I_i(t) - D * J(r > r_min) * E_i(t)""" #Burst??
        Phase[0] = E - coupling*I- damp*((r2>radius).astype(int))*E 
        """I_i(t + dt) = I_i(t) + C * E_i(t) - D * J(r > r_min) * I_i(t)"""
        Phase[1] = I + coupling*E - damp*((r2>radius).astype(int))*I
    return Phase

        


# # # # # # # # # # # # # #
# # Setting up the map  # #
# # # # # # # # # # # # # #
""" The agent begins at the center of a 3x3 grid and has to find 
    the goal to obtain a reward."""
# Grid #
states = np.zeros((3,3))    # Array of states

state_set = list(range(int(states.shape[0]*states.shape[1]))) #index of states

# Reward map #
goal_map = states
for row in range(goal_map.shape[0]):
    for column in range(goal_map.shape[1]):
        """ Setting up the (un)desired locations for the actor"""
        # location of goal
        if row == 2 and column == 0:
            goal_map[row][column] = 100
        # all other locations
        else: 
            goal_map[row][column] = -100
print(goal_map)


#set of actions
#        left      up      right   down     L-down   L-up    R-down    R-up  #
moves = [[-1, 0], [0, 1], [1, 0], [0, -1],[-1, -1], [-1, 1], [1, -1], [1, 1]]

action_set = list(range(len(moves))) #index list



# # Setting up the time constraints # # 

n_trials = 12

srate = 500  #sample rate

# total time in relation to sample rate to simulate time per move
pre_move_t = int(.2*srate)     #just you know, being in the moment
move_t = int(1.3*srate)        #thinking about where to go
total_time = pre_move_t+move_t #total timesteps
all_steps = total_time*n_trials
# # # # # # # # # # # # # # # # # # # # # # # #
# # Setting up the synchronization modules  # #
# # # # # # # # # # # # # # # # # # # # # # # #

"""""""""""""""""""""
  Processing module
"""""""""""""""""""""

# Initial variables #

r2_max = 1     #maximum amplitude of nodes
drift = 2      #rate of drift between coupling parameters

cg_1 = (30/srate)*np.pi             #gamma band coupling parameter for input information
cg_2 = cg_1 + (drift/srate)*2*np.pi #gamma band coupling parameter for actions
 
damp = 0.3     #damping parameter
decay = 0.9    #decay parameter
noise = 0.5    #noise parameter

# Initial matrices #

n_states = len(state_set)
n_actions= len(action_set)

#Setting up phase code neurons across entire task
S_Phase = np.zeros((2,states.shape[0],states.shape[1],total_time,n_trials)) #State phase code units
A_Phase = np.zeros((2,n_actions,total_time,n_trials)) #Action phase code units

#Setting up rate code neurons across entire task
S_Rate = np.zeros((states.shape[0],states.shape[1],total_time,n_trials)) #State rate code units
A_Rate = np.zeros((n_actions,total_time,n_trials)) #Action rate code units
#State-Action Weight Matrix
W = np.ones((states.shape[0],states.shape[1],n_actions))*0.5


"""""""""""""""""""""
  Control   module
"""""""""""""""""""""

# MFC #
# Initial variables 
r2_MFC = .8            #maximum amplitude MFC node
damp_MFC = 0.003       # damping parameter MFC
acc_slope = 10         # MFC slope parameter ---> steepness of burst probability distribution
ct = (5/srate)*2*np.pi #theta band coupling parameter for MFC

#Setting up phase code nodes for the MFC
MFC = np.zeros((2,total_time,n_trials))
#Setting up phase code neuron for MFC -> Bernoulli rate code
Be = 0       
"""When the be value as the rate code of MFC
 reaches certain threshold the MFC will send a burst to coupled neurons"""

# LFC #

#Module indicating which actions should be coupled to which state
LFC = np.zeros((states.shape[0],states.shape[1],n_trials))

#Module that gives the right indices to synchronize
LFC_sync = np.zeros((states.shape[0],states.shape[1],n_actions))

LFC_sync[:,:,:] = action_set   # For now we're just going to synchronize all actions


print(LFC_sync)
# # # # # # # # # # # # # #
# #      Simulation     # #
# # # # # # # # # # # # # #

# Logging dependent variables
Hit = np.zeros((total_time,n_trials))  #log when there is a burst from the MFC
Goal_reach  = np.zeros((n_trials))     #record if goal is reached 
Move = np.zeros((n_trials))            #record move

# Recording sync
sync = np.zeros((n_states,n_actions,n_trials)) 




""" L O O P """

#starting points of phase code oscillations
S_Phase[:,:,:,0,0] = np.random.random((2,states.shape[0],states.shape[1]))   # random starting points processing module
A_Phase[:,:,0,0] = np.random.random((2,n_actions))  # idem
MFC[:,0,0] = np.random.random((2))                  # random starting points MFC

time = 0

start_loc = [1,1]   #start in the center
for trial in range(n_trials):
    #after every trial step the agent gets relocated to the middle
    current_loc = start_loc
    
    #Starting points for the trial
    if trial>0:
        #Continuity from one trial to the next
        S_Phase[:,:,:,0,trial] = S_Phase[:,:,:,time,trial-1]
        A_Phase[:,:,0,trial] = A_Phase[:,:,time,trial-1]
        MFC[:,0,trial]       = MFC[:,time,trial-1]
    
    t = time
    
    """Sync without considering action"""
    # Pre decision time, just some initial synchronization #
    for time in range(pre_move_t):
        
        #Set the LFC what to syncronize (which is all actions)
        LFC[current_loc[0],current_loc[1],trial] = 1

        #Update phase code neurons for actions and states in processing module
        #State phase code neurons
        S_Phase[:,:,:,time+1,trial] = update_phase(nodes=S_Phase[:,:,:,time,trial], grid = True, radius=r2_max, damp = damp, coupling = cg_1,multiple=True )
        #Action phase code neurons
        A_Phase[:,:,time+1,trial] = update_phase(nodes=A_Phase[:,:,time,trial], grid = False, radius=r2_max, damp = damp, coupling = cg_2,multiple=True )

        #Update phase code untis of MFC
        MFC[:,time+1,trial] = update_phase(nodes=MFC[:,time,trial], grid = False, radius=r2_MFC, damp=damp_MFC, coupling=ct,multiple=False)

        #MFC rate code neuron-> Bernoulli process

        Be = 1/(1 + np.exp(-acc_slope*(MFC[0,time,trial]-1)))
        p = random.random()

        if p < Be:

            Gaussian = np.random.normal(size = [1,2])  #noise factor as normal distribution
            
            Hit[time,trial] = 1
            
            #For all states in grid map
            for y in range(states.shape[0]):
                for x in range(states.shape[1]):
                
                #If the states are coded in the LFC (for now this is just the current state)
                    if LFC[y,x,trial]:
                            #The states receive a burst
                            S_Phase[:,y,x,time+1,trial] = decay*S_Phase[:,y,x,time,trial] + Gaussian

                            # and all the actions that are to be synchronized to that state receive a burst
                            for nodes in LFC_sync[current_loc[0],current_loc[1],:]:
                                A_Phase[:,int(nodes),time+1,trial] = decay*A_Phase[:,int(nodes),time,trial] + Gaussian
                
                # For desynchronization it would probably look something like this
                #elif LFC[y,x,trial] == -1:
                        # #The irrelevant states receive a negative burst
                        # S_Phase[:,y,x,time+1,trial] = decay*S_Phase[:,y,x,time,trial] - Gaussian

                        # # and all the irrelevant actions that are to be desynchronized to that state receive a negative burst
                        # for nodes in LFC_sync[current_loc[0],current_loc[1],:]
                        #     A_Phase[:,int(nodes),time+1,trial] = decay*A_Phase[:,int(nodes),time,trial] - Gaussian

    t = time


    """Sync and action"""
    # The actor makes the move #
    for time in range(move_t):
        
        #Update phase code neurons for actions and states in processing module
        #State phase code neurons
        S_Phase[:,:,:,time+1,trial] = update_phase(nodes=S_Phase[:,:,:,time,trial], grid = True, radius=r2_max, damp = damp, coupling = cg_1,multiple=True )
        #Action phase code neurons
        A_Phase[:,:,time+1,trial] = update_phase(nodes=A_Phase[:,:,time,trial], grid = False, radius=r2_max, damp = damp, coupling = cg_2,multiple=True )

        #Update phase code untis of MFC
        MFC[:,time+1,trial] = update_phase(nodes=MFC[:,time,trial], grid = False, radius=r2_MFC, damp=damp_MFC, coupling=ct,multiple=False)

        #MFC rate code neuron-> Bernoulli process

        Be = 1/(1 + np.exp(-acc_slope*(MFC[0,time,trial]-1)))
        p = random.random()

        if p < Be:

            Gaussian = np.random.normal(size = [1,2])  #noise factor as normal distribution
            Hit[time,trial] = 1
            
            #For all states in grid map
            for y in range(states.shape[0]):
                for x in range(states.shape[1]):
                
                    #If the states are coded in the LFC (for now this is just the current state)
                    if LFC[y,x,trial]:
                            #The states receive a burst
                            S_Phase[:,y,x,time+1,trial] = decay*S_Phase[:,y,x,time,trial] + Gaussian

                            # and all the actions that are to be synchronized to that state receive a burst
                            for nodes in LFC_sync[current_loc[0],current_loc[1],:]:
                                A_Phase[:,int(nodes),time+1,trial] = decay*A_Phase[:,int(nodes),time,trial] + Gaussian
       
        #Updating rate code units
        #Only the rate code neuron of a single state is updated because the actor can only be in one place at the same time
        S_Rate[current_loc[0],current_loc[1],time,trial]= (1/(1+np.exp(-5*S_Phase[0,current_loc[0],current_loc[1],time,trial]-0.6)))
        A_Rate[:,time,trial]=(S_Rate[current_loc[0],current_loc[1],time,trial]*W[current_loc[0],current_loc[1],:])*(1/(1+np.exp(-5*A_Phase[0,:,time,trial]-0.6)))


    # After some time the actor has made up its mind and makes a move

    action_index = list(np.sum(A_Rate[:,:,trial],axis=1)).index(max(np.sum(A_Rate[:,:,trial],axis=1)))
    #Selected action is
    move = np.array((moves[action_index]))
    #update location
    new_loc=np.array((current_loc))+move
    
    """""""""""""""
      Learning
    """""""""""""""
    alpha = 0.2   #learning rate


    #logging some coordinates
    y = current_loc[0]    # y coordinate on grid
    x = current_loc[1]    # x coordinate on grid
    y_p = new_loc[0]      # y prime coordinate
    x_p = new_loc[1]      # x prime coordinate

    reward = goal_map[y_p][x_p] # reward 
    W[y][x][action_index] = W[y][x][action_index] + alpha*(reward - W[y][x][action_index])  #weight update


trial=11

print(W[y][x])
print(A_Rate[:,time,trial])


fig, axs = plt.subplots(4)
#good action
axs[0].title.set_text('Phase code units of center state and the correct action node')
axs[0].plot(np.arange(int(total_time)),S_Phase[0,y,x,:,trial])
axs[0].plot(np.arange(int(total_time)),A_Phase[0,6,:,trial])
axs[0].plot(np.arange(int(total_time)),Hit[:,trial])
axs[1].plot(np.arange(int(total_time)), S_Rate[y,x,:,trial])
axs[1].plot(np.arange(int(total_time)), A_Rate[6,:,trial])
#bad action
axs[2].title.set_text('Phase code units of center state and the wrong action node')
axs[2].plot(np.arange(int(total_time)),S_Phase[0,y,x,:,trial])
axs[2].plot(np.arange(int(total_time)),A_Phase[0,2,:,trial])
axs[2].plot(np.arange(int(total_time)),Hit[:,trial])
axs[3].plot(np.arange(int(total_time)), S_Rate[y,x,:,trial])
axs[3].plot(np.arange(int(total_time)), A_Rate[2,:,trial])
plt.tight_layout()
plt.show()


    


    





