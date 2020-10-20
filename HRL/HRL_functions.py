import numpy as np
import random, os, time
import matplotlib.pyplot as plt

print(os.getcwd())
# Defining the codes
floor = 0
wall = 1
agent = 2
goal = 3
door = 4


""" STRUCTURAL PARAMETERS """

# Choose the environment
env = 'HRL/Env_5.txt'
# If set to True then the agent can move diagonally
diagonal_movement = True

""""""""""""""""""""""""""""""

reward_size = 100
psuedo_reward_size = 100


# Possible moves
if diagonal_movement:
    #        left      up      right   down     L-down   L-up    R-down    R-up  #
    moves = [[-1, 0], [0, 1], [1, 0], [0, -1],[-1, -1], [-1, 1], [1, -1], [1, 1]]
else:
    #        left       up     right    down        #
    moves = [[-1, 0], [0, 1], [1, 0], [0, -1]]

# This encodes corresponding text file keys to numerical value

def filter(c):
    if c == 'F':
        return floor    # 'FLOOR' or empty space
    elif c == 'W':
        return wall    # 'WALL'
    elif c == 'S':
        return agent    # 'START'
    elif c == 'G':
        return goal    # 'GOAL'
    elif c == 'D':
        return door     # 'DOOR'



# opening up the text file and putting the elements into an array
with open(env) as file:
    grid_list = []

    for line in file.readlines():
        line = line.split()
        grid_list.append(list(map(filter, line)))
        grid_array = np.array(grid_list)

''' Location and structural functions '''
# returns the location of a given element in the array
def get_index(array, search):
    result = np.where(array == search)
    assert len(result[0]) == 1, "there is more than one of what you are looking far"
    index = np.array((int(result[0]), int(result[1])))

    return index


# Returns a list with all floor_locations for future use
def get_all_floor_locations(grid_array):
    all_floor_locations = []
    for y in range(grid_array.shape[0]):
        for x in range(grid_array.shape[1]):
            if not grid_array[y][x] == 1:
                all_floor_locations.append([y,x])
    return all_floor_locations

#generates a random starting location
def random_start_loc(locations):
    return random.choice(locations)

# A (way too bulky) function tagging the location of all the rooms
def tag_four_rooms(grid_array, env):
    ''' Dividing grid into rooms and tagging the room by lettername'''

    width_rooms = []
    for y in range(grid_array.shape[0]):
        is_it_a_wall = []
        for x in range(grid_array.shape[1]):
            # check if the grid element is a wall or a door
            if grid_array[y][x] == 1 or grid_array[y][x] == 4:
                is_it_a_wall.append(True)
            else:
                is_it_a_wall.append(False)

            if len(is_it_a_wall) > 1:
                # if there are two consecutive wall elements the the row is likely to be completely wall
                if is_it_a_wall[-1] == True and is_it_a_wall[-2] == True:
                    # Storing the information about the wall row
                    width_rooms.append([[y,True],[x-1,grid_array.shape[1]]])
                    wall_loc = y
                    break
                if y == wall_loc + 1: # only check the width of the rooms for the row after a wall
                    # Storing where in the row the floors begin and end
                    if is_it_a_wall[-1] == False and is_it_a_wall[-2] == True:
                        x_start = x-1   #begin
                    
                    elif is_it_a_wall[-1] == True and is_it_a_wall[-2] == False:
                        x_end = x # end
                        width_rooms.append([[y,False],[x_start,x_end]])

    # This method of division requires there to be four rooms
    room_A = np.copy(grid_array[width_rooms[0][0][0]:width_rooms[3][0][0]+1,width_rooms[1][1][0]:width_rooms[1][1][1]+1])
    room_B = np.copy(grid_array[width_rooms[0][0][0]:width_rooms[3][0][0]+1,width_rooms[2][1][0]:width_rooms[2][1][1]+1])
    room_C = np.copy(grid_array[width_rooms[3][0][0]:width_rooms[6][0][0]+1,width_rooms[4][1][0]:width_rooms[4][1][1]+1])
    room_D = np.copy(grid_array[width_rooms[3][0][0]:width_rooms[6][0][0]+1,width_rooms[5][1][0]:width_rooms[5][1][1]+1])

    # putting the room-arrays into a list
    room_list = [room_A,room_B,room_C, room_D]

    tag = 0
    for room in room_list:
        for y in range(room.shape[0]):
            for x in range(room.shape[1]):
                if room[y][x] == 1:
                    room[y][x] = 7
                elif room[y][x] == 4:
                    room[y][x] = 4
                else:
                    room[y][x] = tag
        tag+=1


    ### the new array

    #deleting the double column walls
    room_B = np.delete(room_B,0, axis = 1)
    room_D = np.delete(room_D,0, axis = 1)
    # putting it together
    which_room = np.vstack((np.hstack((room_A,room_B)),np.hstack((room_C,room_D))))

    #removing the double row
    if env == 'Env_3.txt':
        which_room = np.delete(which_room,5,axis = 0)
    elif env == 'Env_1.txt':
        which_room = np.delete(which_room,10,axis= 0)
    elif env == 'Env_5.txt':
        which_room = np.delete(which_room,5,axis = 0)
    return which_room

''' Movement functions '''

# Function for taking a primitive action in environment
def make_move(agent_loc, action):
    new_loc = agent_loc + np.array(action)  # adding up location vectors

    # checking if agent hit wall
    if grid_array[new_loc[0]][new_loc[1]] == wall:
        finished = False
        reward = 0
        return agent_loc, finished, reward  # return old location

    # checking if agent made it to goal
    elif grid_array[new_loc[0]][new_loc[1]] == goal:
        finished = True
        reward = reward_size
        return new_loc, finished, reward

    # normal step
    else:
        finished = False
        reward = 0
        return new_loc, finished, reward


''' Action Selection  '''

'''  " Whether softmax action selection or epsilon-greedy action selection is better
     is unclear and may depend on the task and on human factors. [...] ; setting Tau 
     requires knowledge of the likely action values and of powers of e. "
                - Sutton & Barto   '''

# Epsilon greedy action selectiol: best action selected with chance 1-epsilon
def e_greedy_action(Q, agent_loc, epsilon):

    # drawing up the action weights for the current location
    state_action_weights = Q[agent_loc[0]][agent_loc[1]]
    if np.random.rand() > epsilon:
        # getting the index of the action with highest value
        max_action_index = np.where(state_action_weights == max(state_action_weights))[0]
        action = moves[random.choice(max_action_index)]  # selecting the move
    else:
        # Exploration ! Selecting a random action
        action = random.choice(moves)

    return action


# Softmax action selection: weighted action selection with temperatur parameter tau
def softmax_action(Q, policy, agent_loc, tau):
    # initializing a list of action indices
    action_indices = list(range(len(moves)))

    y, x = agent_loc[0], agent_loc[1]


    sum_of_weights = np.sum( np.exp( ((Q[y][x][:]) - max(Q[y][x][:]))/tau) )
    # policy updating
    for i in action_indices:
        # according to softmax equation
        policy[y][x][i] = (np.exp( ((Q[y][x][i])-max(Q[y][x][:]))/tau) )/sum_of_weights

    # selecting action from policy
    action_index = np.random.choice(action_indices, 1, p=policy[y][x])
    action = moves[int(action_index)]

    return action



""" The following functions describe making a move when an option
is intitiated and how actions are selected before and during such
an option """

## Making a move while in an option is initiated  (higher order action structure)
def within_option_move(agent_loc, action, option, training = True):
    new_loc = agent_loc + np.array(action)
    
    #In the case of a regular move
    at_door = False
    at_goal = False
    reward = 0

    # checking if agent hit wall
    if grid_array[new_loc[0]][new_loc[1]] == wall:
        new_loc = agent_loc # return old location

    # checking if agent made it to goal
    elif grid_array[new_loc[0]][new_loc[1]] == goal:
        if list(new_loc) == option[2]:
            at_door = True
            reward = psuedo_reward_size
            if training:
                at_goal = True
            else:
                at_goal = True
                reward = reward_size

    # checking if agent reaches the option's terminal state
    elif list(new_loc) == option[2]:
        at_door = True
        reward = psuedo_reward_size

    return new_loc, reward, at_door, at_goal

# Softmax option selection: weighted option selection with temperatur parameter tau
# the options now entail the classical list of moves extended by an additional index
# that, when selected, indicates that the agent will select an option. 
# --> This function should be used at the root level.

def softmax_option(Q, policy, agent_loc, tau):
    # initializing a list of option indices
    action_indices = list(range(len(moves)+1))

    y, x = agent_loc[0], agent_loc[1]

    sum_of_weights = np.sum( np.exp( ((Q[y][x][:]) - max(Q[y][x][:]))/tau) )
    # policy updating
    for i in action_indices:
        # according to softmax equation
        policy[y][x][i] = (np.exp( ((Q[y][x][i])-max(Q[y][x][:]))/tau) )/sum_of_weights
    # selecting action from policy
    index = np.random.choice(action_indices, 1, p=policy[y][x])

    index = int(index)
    ## Check whether an option or an action has been chosen
    # option
    if index == action_indices[-1]:
        chose_option = True
        action = index
    # action
    else:
        chose_option = False
        action = moves[index]
    
    return chose_option, action

## Creating a choice between options, softmax-style for all options after the agent
## has made a choice to take an option instead of an option
def choose_any_option(Q, policy,agent_loc, list_of_options, tau):
    option_indices = list(range(len(list_of_options)))

    y, x = agent_loc[0], agent_loc[1]

    sum_of_weights = np.sum( np.exp( ((Q[y][x][:]) - max(Q[y][x][:]))/tau) )
    # policy updating
    for i in option_indices:
        # according to softmax equation
        policy[y][x][i] = (np.exp( ((Q[y][x][i])-max(Q[y][x][:]))/tau) )/sum_of_weights
    # selecting action from policy
    option_index = np.random.choice(option_indices, 1, p=policy[y][x])



    return list_of_options[int(option_index)], option_index






''' Plotting Stuff '''
#Function that gives a graphical representation of the learned path


def display_path(Q, start_loc):
    agent_loc = start_loc  # intial location
    finished = False

    while not finished:
        # greedy action selection
        action = e_greedy_action(Q, agent_loc, epsilon=0)

        # update agent location in grid
        grid_array[agent_loc[0]][agent_loc[1]] = agent

        # plot
        plt.imshow(grid_array, cmap='hot', interpolation='nearest')
        plt.show()

        new_loc, finished, reward = make_move(agent_loc, action)  # Take step

        # update space behind agent back to floor space
        grid_array[agent_loc[0]][agent_loc[1]] = floor
        agent_loc = new_loc  # update location


####################
# Option Framework #
####################

# Creating the list of all locations where options are terminated (in this case: doors)
termination_set = []
## termination set is the set of all doors
for y in range(grid_array.shape[0]):
    for x in range(grid_array.shape[1]):
        if grid_array[y][x] == 4:
            termination_set.append([y,x])


# A state/action value function is initialized for each option
# and the options are coded as a list with the following indices:
# 0: Option specific State-Value function
# 1: Option specific Action-Value function
# 2: location of termination state (door)
# 3: Option specific policy

## Option 1 ##
# Weights to "go to door inbetween AB"
Q_go_to_AB = np.zeros((grid_array.shape[0], grid_array.shape[1], len(moves)))
V_go_to_AB = np.zeros((grid_array.shape))
policy_AB = np.full((grid_array.shape[0], grid_array.shape[1], len(moves)), (1/len(moves)))
# option-relevant info data list
go_to_AB = [V_go_to_AB,Q_go_to_AB,termination_set[0],policy_AB,0]

## Option 2 ##
# Weights to "go to door inbetween AC"
Q_go_to_AC = np.zeros((grid_array.shape[0], grid_array.shape[1], len(moves)))
V_go_to_AC = np.zeros((grid_array.shape))
policy_AC = np.full((grid_array.shape[0], grid_array.shape[1], len(moves)), (1/len(moves)))
# option-relevant info data list
go_to_AC = [V_go_to_AC,Q_go_to_AC,termination_set[1],policy_AC,1]


## Option 3 ##
# Weights to "go to door inbetween BC"
Q_go_to_BD = np.zeros((grid_array.shape[0], grid_array.shape[1], len(moves)))
V_go_to_BD = np.zeros((grid_array.shape))
policy_BD = np.full((grid_array.shape[0], grid_array.shape[1], len(moves)), (1/len(moves)))
# option-relevant info data list
go_to_BD = [V_go_to_BD,Q_go_to_BD,termination_set[2],policy_BD,2]


## Option 4 ##
# Weights to "go to door inbetween CD"
Q_go_to_CD = np.zeros((grid_array.shape[0], grid_array.shape[1], len(moves)))
V_go_to_CD = np.zeros((grid_array.shape))
policy_CD = np.full((grid_array.shape[0], grid_array.shape[1], len(moves)), (1/len(moves)))
# option-relevant info data list
go_to_CD = [V_go_to_CD,Q_go_to_CD,termination_set[3],policy_CD,3]

## Option 5
# Weights to "go to the goal"
termination_set.append(list(get_index(grid_array,goal)))
Q_go_to_G = np.zeros((grid_array.shape[0], grid_array.shape[1], len(moves)))
V_go_to_G = np.zeros((grid_array.shape))
policy_G = np.full((grid_array.shape[0], grid_array.shape[1], len(moves)), (1/len(moves)))
# option-relevant info data list
go_to_G = [V_go_to_G,Q_go_to_G,termination_set[4],policy_G,4]

#list of all the option-objects
list_of_options = [go_to_AB,go_to_BD,go_to_CD,go_to_AC,go_to_G]


# A grid array with each room and it's corresponding tag
which_rooms = tag_four_rooms(grid_array, env)
# Each room is indexed according to the list below
room_labels = ["A","B","C","D"]
# The option location dictionary
# For each location there are now two options available, each directed at a door.
# For each door the options are directed at the other two closest doors.
which_options        = {"A": [go_to_AB,go_to_AC],"B":[go_to_AB,go_to_BD],
                      "C": [go_to_CD,go_to_AC], "D": [go_to_CD,go_to_BD],
                        tuple(termination_set[0]):[go_to_AB,go_to_BD],tuple(termination_set[1]):[go_to_AB,go_to_CD], 
                        tuple(termination_set[2]):[go_to_AB,go_to_CD], tuple(termination_set[3]):[go_to_AC,go_to_BD] }

def which_options_per_room(init_loc):
        # Calling up the object of the chosen option
        # if at a floor location then check which room you are in
    if not which_rooms[init_loc[0]][init_loc[1]] == 4:
        room = room_labels[which_rooms[init_loc[0]][init_loc[1]]]

        option_list = which_options[room]
        # if at a door location check the options
    else:
        option_list = which_options[tuple(init_loc)]
    if len(option_list) < 3:
        option_list.append(go_to_G)
    #returning the list of options available for room
    return option_list