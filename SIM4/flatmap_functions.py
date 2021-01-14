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


###MAP related things ###

def random_loc(grid = []):
    """Get a random location that isn't a wall"""
    y, x = 0, 0  #upper left corner is definitely a wall
    while grid[y,x] == 1:
        #get random coordinates and then see if it's a wall
        y, x = random.choice(range(int(grid.shape[0]))),random.choice(range(int(grid.shape[1])))
    return [y,x]



def update_location(grid = [], loc = [], move = []):
    """updating the location of the agent, should it run into a wall it is just returned to the original location"""
    new_loc = np.array((loc))+np.array((move))
    if grid[new_loc[0],[new_loc[1]]] == 1:
        new_loc = loc
    return list(new_loc)




def create_grid_from_file(map_file = "enter file name", goal_location = [], reward_size = int, sub_reward_size = int):
    """ Create a map for HRL sync model by reading a text file where the map
        can be "drawn". 

        The map has letters that are then translated to relevant numbers
        walls --> 1
        floor tiles --> 0
        doors --> size of door reward
        goal --> size of final reward

        The function then returns the map in a numpy array"""
    
    assert reward_size > 5, "Reward Size is not large enough (<6)"
    if sub_reward_size != 0:
        assert sub_reward_size > 5, "Psuedo reward size is not large enough (<6)"
    def filter(c):
        """ translates the letters to the desired number"""
        if c == 'F':
            return 0    # 'FLOOR' or empty space
        elif c == 'W':
            return 1    # 'WALL'
        elif c == 'G':
            return sub_reward_size  # 'GOAL'
        elif c == 'A':
            return 2
        elif c == 'B':
            return 3
        elif c == 'C':
            return 4
        elif c == 'D':
            return 5

    # opening up the text file and putting the elements into an array
    with open(map_file) as file:
        grid_list = []
        #read the map file line per line
        for line in file.readlines():
            line = line.split()
            grid_list.append(list(map(filter, line)))
            grid_array = np.array(grid_list)
    
    if len(goal_location) == 0:
        grid_array[int(grid_array.shape[0]-2),2] = reward_size
    else: 
        assert int(grid_array[goal_location[0],goal_location[1]]) != 1, "Don't put the goal in a wall!"
        assert int(grid_array[goal_location[0],goal_location[1]]) != sub_reward_size, "Don't put the goal in a door!"

        grid_array[goal_location[0],goal_location[1]] = reward_size

    return grid_array





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
# def softmax_action(action_weights = [], tau = int):
#     """Selecting an action by the Softmax equation"""
#     # initializing a list of action indices
#     action_indices = list(range(len(action_weights)))
#     #Softmax action equation in paper
#     action_prob = np.zeros((len(action_indices)))
    
#     for action in action_indices:
#         #exponent weight of current action divided by exp weight of all other actions
#         #logging the action probability
#         action_prob[action] = np.exp((action_weights[action]/tau))/np.sum(np.exp(action_weights/tau))
#     #softmax action

#     action_index = np.random.choice(action_indices, 1, p=action_prob)
#     return action_index[0]


def softmax_action(action_weights = [], tau = int,seed=float):
    action_indices = list(range(len(action_weights)))
    f = np.exp((action_weights - np.max(action_weights))/tau)  # shift values
    action_prob = f / f.sum(axis=0)
    action_index = np.random.choice(action_indices, 1, p=action_prob)
    return action_index[0]

def greedy_action(action_weights= []):

    # drawing up the action weights for the current location
    state_action_weights = action_weights
    # getting the index of the action with highest value
    max_action_index = np.where(state_action_weights == max(state_action_weights))[0]
    #print(np.where(state_action_weights == max(state_action_weights)))

    action_index = random.choice(max_action_index)  # selecting the move
    return action_index




def get_intitation_states(grid = [], reward_size = int, sub_reward_size = int):
    """Returns an array that includes the available options for
        each state in the grid"""
    n_doors = np.count_nonzero(grid == sub_reward_size)       #number of doors
    initiation_set = np.zeros((grid.shape[0],grid.shape[1],2)) #empty initiation state grid
    rooms_list = [i for i in range(2,(2+n_doors))]          #list of available floor indices
    
    # hard code for the options indices in each room 
    list_for_floors = [[0,1],[2,3],[4,5],[6,7]]
    # hard code the options indices for doors (draw it out)
    list_for_doors = [[1,3],[0,5],[2,7],[4,6]]

    
    # Go over each state and fill relevant values in initiation states
    i = 0 # door counter
    for y in range(int(grid.shape[0])):
        for x in range(int(grid.shape[1])):
            state_value = grid[y,x] #get state code value

            #for normal state values
            if state_value in rooms_list:
                #fill in the initation set
                initiation_set[y,x,:] = list_for_floors[int(state_value-2)]

            #for goal or subgoal states
            elif state_value in [reward_size,sub_reward_size]:
                #if its a subgoal state
                if state_value ==sub_reward_size:
                    initiation_set[y,x,:] = list_for_doors[i]
                    i+=1
                #if its a goal state
                else:
                    #check values left and right to see which room ur in
                    left_value = grid[y,int(x-1)]
                    right_value= grid[y,int(x+1)]
                    if left_value in rooms_list:
                        initiation_set[y,x,:] = list_for_floors[int(left_value-2)]
                    elif right_value in rooms_list:
                        initiation_set[y,x,:] = list_for_floors[int(right_value-2)]

    return initiation_set


def get_termination_states(grid = [], sub_reward_size = int):
    doorlist = np.where(grid == sub_reward_size)
    n_doors = np.count_nonzero(grid == sub_reward_size)
    doors = []
    for i in range(n_doors):
        y = doorlist[0]
        x = doorlist[1]
        doors.append([y[i],x[i]])

    termination_set = [doors[0],doors[1],doors[0],doors[2],doors[1],doors[3],doors[2],doors[3]]
    return termination_set

    



def update_weights(param = dict, index= list, V = [], W = [], states = [], reward_size = int):
    alpha = param["alpha"]    #state value learning parameter or  "critic"
    gamma = param["gamma"]    #reward discount parameter
    beta = param["beta"]      #state-action value parameter   or   "actor"
    at_goal = False           #for now assume agent not at goal
    y, x, y_p, x_p, a = index[0],index[1],index[2],index[3], index[4]  #relevant indices

    # if y == y_p and x == x_p:
    #     reward = -1
    # else:
    if states[y_p][x_p] == reward_size:
        reward = reward_size
        at_goal = True
    else:
        reward = 0

    delta = reward + gamma*V[y_p][x_p] - V[y][x]   #prediction error
    V[y][x] = V[y][x] + alpha*delta                #state value update -> "Critic"
    W[y][x][a] = W[y][x][a] + beta*delta           #state-action value update -> "Actor"



    return V, W, delta, at_goal

