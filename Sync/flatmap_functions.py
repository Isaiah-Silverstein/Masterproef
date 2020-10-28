import random 
import numpy as np

""" F L A T M A P    S Y N C """

# # # # # # # # # # # # # #
# #      Functions      # #
# # # # # # # # # # # # # #

"""Sync functions"""


def update_phase(nodes = [], grid = True, radius = 1, damp = 0.3, coupling = 0.3, multiple = True):
    """ updating the phase per step according to the following formulas"""
    """ E_i(t + dt) = E_i(t) - C * I_i(t) - D * J(r > r_min) * E_i(t) """
    """ I_i(t + dt) = I_i(t) + C * E_i(t) - D * J(r > r_min) * I_i(t) """
    # The E node is conventially in the first index and I in the second

    if multiple:
        r2 = np.sum(nodes*nodes,axis = 0)  #radius squared is sum of E and I nodes squared
        if grid:
            Phase = np.zeros((2,nodes.shape[1],nodes.shape[2])) #Setting up return array
            E = nodes[0,:,:]
            I = nodes[1,:,:]
            Phase[0,:,:] = E - coupling*I - damp*((r2>radius).astype(int))*E 
            Phase[1,:,:] = I + coupling*E - damp*((r2>radius).astype(int))*I

        else:
            Phase = np.zeros((2,len(nodes[0,:]))) #Setting up return array
            E = nodes[0,:]
            I = nodes[1,:]
            Phase[0,:] = E - coupling*I - damp*((r2>radius).astype(int))*E 
            Phase[1,:] = I + coupling*E - damp*((r2>radius).astype(int))*I

    else:
        Phase = np.zeros((2)) #Setting up return array
        r2 = np.sum(nodes[0]*nodes[1],axis = 0) #radius squared is sum of E and I nodes squared
        E = nodes[0]
        I = nodes[1]
        Phase[0] = E - coupling*I- damp*((r2>radius).astype(int))*E 
        Phase[1] = I + coupling*E - damp*((r2>radius).astype(int))*I

    return Phase


"""Mechanical funtions"""


def update_location(grid = [], loc = [], move = []):
    """updating the location of the agent, should it run into a wall it is just returned to the original location"""
    new_loc = np.array((loc))+np.array((move))
    if grid[new_loc[0],[new_loc[1]]] == 1:
        new_loc = loc
    return list(new_loc)


def create_square_grid(size = 5, goal_location = [3,1], reward_size = 100):
    """Create a grid of zeros that is surrounded by ones"""
    height, width = size, size                 #size of map
    borders = np.ones((height,width))                 #array of walls
    tiles = np.zeros((int(height-2),int(width-2)))    #array of tiles that the agent can move in
    borders[1:int(height-1),1:int(width-1)] = tiles   #plug the tiles in the map

    """create a goal map based on the goal location""" 
    grid = borders    #map based on grid map
    # loop over map indices
    for row in range(size):
        for column in range(size):
            """ Setting up the (un)desired locations for the actor"""
            # location of goal
            if row == goal_location[0] and column == goal_location[1]:
                grid[row][column] = reward_size

    return grid



""" Reinforcement Learning functions """


def softmax_action(action_rate = [], tau = 10):
    """Selecting an action by the Softmax equation"""
    # initializing a list of action indices
    action_indices = list(range(len(action_rate)))
    #Softmax action equation in paper
    action_prob = np.zeros((len(action_indices)))    
    for action in action_indices:
        #exponent weight of current action divided by exp weight of all other actions
        #logging the action probability
        action_prob[action] = np.exp((action_rate[action]/tau))/np.sum(np.exp(action_rate/tau))
    #softmax action
    action_index = np.random.choice(action_indices, 1, p=action_prob)
    return action_index[0]


def update_weights(param = dict, index= list, V = [], W = [], states = []):
    alpha = param["alpha"]    #state value learning parameter or  "critic"
    gamma = param["gamma"]    #reward discount parameter
    beta = param["beta"]      #state-action value parameter   or   "actor"
    at_goal = False           #for now assume agent not at goal
    y, x, y_p, x_p, a = index[0],index[1],index[2],index[3], index[4]  #relevant indices

    reward = states[y_p][x_p]                      #see if the action resulted in reward
    
    delta = reward + gamma*V[y_p][x_p] - V[y][x]   #prediction error
    V[y][x] = V[y][x] + alpha*delta                #state value update -> "Critic"
    W[y][x][a] = W[y][x][a] + beta*delta           #state-action value update -> "Actor"

    if reward>0:
        at_goal = True

    return V, W, delta, at_goal