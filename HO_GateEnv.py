#!/usr/bin/env python3

'''
Isaiah attempts an environment
'''
import numpy as np
import random
import time
import matplotlib.pyplot as plt
from HRL_functions import *

''' Key Parameters '''

"""Exploration parameters """
# Exploration parameter for e-greedy action selection
epsilon = 0.1    # greedy action selection if epsilon -> 0
# 'Temperature' paramter for softmax action selection
tau = 5   # greedy action selection if tau -> 0
"""Learning parameters """
alpha = 0.1 # Learning parameter for action-values
alpha_c = 0.15 # Learning parameter for state-values
gamma = 0.9      # Discount parameter
room_based_options = False
###################################################################################
#                               Option Framework                                  #
###################################################################################

''' Root Level Value Arrays '''

# Empty State-Value function
V = np.zeros((grid_array.shape))
# Empty Action-Value function
Q = np.zeros((grid_array.shape[0], grid_array.shape[1], (len(moves)+1)))
# Empty Policy (with equiprobable actions)
policy = np.full((grid_array.shape[0], grid_array.shape[1], (len(moves)+1)), (1/(len(moves)+1)))


## Creating a choice between options, softmax-style for all options after the agent
## has made a choice to take an option instead of an option
Q_options = np.zeros((grid_array.shape[0], grid_array.shape[1], len(list_of_options)))
policy_options = np.full((grid_array.shape[0], grid_array.shape[1], len(list_of_options)), (1/len(list_of_options)))


## Other than that, the option could be selected from a smaller set of options, those relevant to the room
Q_rooms = np.zeros((grid_array.shape[0], grid_array.shape[1], 3))
policy_rooms = np.full((grid_array.shape[0], grid_array.shape[1], 3), (1/3))


""" Now the agent can put what it has learned to good use. Roaming around
the environtment-space, potentially making use of the options, and discovering
the reward cause by the goal-state. It should then make use of the option-framework 
to get to the goal-state in a faster manner."""



##########################################################################
#                        Goal - Training                                 #
##########################################################################
''' Root Level Value Arrays '''

# Empty State-Value function
V = np.zeros((grid_array.shape))

# Empty Action-Value function
Q = np.zeros((grid_array.shape[0], grid_array.shape[1], (len(moves)+1)))

# Empty Policy (with equiprobable actions)
policy = np.full((grid_array.shape[0], grid_array.shape[1], (len(moves)+1)), (1/(len(moves)+1)))

## Creating a choice between options, softmax-style for all options after the agent
## has made a choice to take an option instead of an option
Q_options = np.zeros((grid_array.shape[0], grid_array.shape[1], len(list_of_options)))
policy_options = np.full((grid_array.shape[0], grid_array.shape[1], len(list_of_options)), (1/len(list_of_options)))


## Other than that, the option could be selected from a smaller set of options, those relevant to the room
Q_rooms = np.zeros((grid_array.shape[0], grid_array.shape[1], 3))
policy_rooms = np.full((grid_array.shape[0], grid_array.shape[1], 3), (1/3))


''' Parameters '''
n_episodes = 200
max_episode_step = 2000
max_o_step = 2000



alpha =  0.1 # state value learning rate
alpha_c = 0.15# action value learning rate
tau = tau # exploration parameter


# Plotting variables
o_step_list = []
step_list_bis = [[] for i in range(len(list_of_options))]
ep_step_list = []


""""
DEbug stuff

# Empty State-Value function
V_test = np.zeros((grid_array.shape))

# Empty Action-Value function
Q_test = np.zeros((grid_array.shape[0], grid_array.shape[1], (len(moves))))

# Empty Policy (with equiprobable actions)
policy_test = np.full((grid_array.shape[0], grid_array.shape[1], (len(moves))), (1/(len(moves))))

Debug stuff
"""

#####################
## Getting Started ##
#####################
n_episodes =200
chose_option = False
at_goal = False
at_door = False
n_steps = 0
n_o_steps = 0
reward = 0
how_many_options = 0

start_loc = get_index(grid_array,agent)

for episode in range(n_episodes):
    at_goal = False
    n_steps = 0
    reward = 0
    ##    # Getting Started #    ##

    # Starting location
    agent_loc = start_loc

    # intial action choice

    chose_option, action = softmax_option(Q,policy,agent_loc,tau)

    while not at_goal: 

        if n_steps > max_episode_step:
            print("didn't make it to goal")
            break
        if chose_option:


            # After the agent choose to take a higher order action, it needs to choose an option
            option, option_index = choose_any_option(Q_options,policy_options,agent_loc,list_of_options,tau)

            # The option is selected now the learning can begin
            r_cum_list = []          # clearing the r_cum list
            init_loc = agent_loc     # intial location
            init_action = action        # intial action

            # Getting started in the option
            action = softmax_action(option[1],option[3],init_loc,tau)    # first within-option action

            while not at_door:
                # Making a move in the environment
                new_loc, psuedo_r, at_door, at_goal = within_option_move(agent_loc,action,option,training = False)
                n_steps += 1 #logging steps
                n_o_steps+=1 #logging option-steps

                if at_goal:
                    reward = reward_size
                
                ## Create the list of rewards found for the option
                r_cum_list.append((gamma**(n_o_steps - 1)*reward))
                #calculate r_cum
                r_cum = np.sum(r_cum_list)

                # selecting an action prime
                action_prime = softmax_action(option[1],option[3],new_loc,tau)

                ''' Updating Functions '''

                y, x, a = agent_loc[0], agent_loc[1], moves.index(action)
                y_prime, x_prime, a_prime = new_loc[0], new_loc[1], moves.index(action_prime)
                V_o, Q_o = option[0], option[1]

                # Prediction error
                if at_door or at_goal:
                    delta = psuedo_r - V_o[y][x]
                else:
                    delta = psuedo_r + gamma*V_o[y_prime][x_prime] - V_o[y][x]

                #State-Value function
                V_o[y][x] = V_o[y][x] + alpha_c*delta

                #Action-Value function
                Q_o[y][x][a] = Q_o[y][x][a] + alpha*delta

                #Updating for next step
                option[0], option[1] = V_o, Q_o
                agent_loc, action = new_loc, action_prime

                ''' Tracking some variables '''
                
                if n_o_steps > max_o_step:
                    print("Option is taking too long")
                    break
                    

            
            # After option
            if at_door == True:
                print("sucessful in {} steps".format(n_o_steps))
                print(init_loc)
                print(new_loc)
                print(option[2])
            if at_goal:
                print("at goal")

            step_list_bis[option[-1]].append(n_o_steps)

            ##the link between root-value functions and higher order reward
            y, x, a = init_loc[0], init_loc[1], init_action
            y_prime, x_prime = new_loc[0], new_loc[1]

            print(r_cum)
            #pred error
            delta = r_cum + (gamma**n_o_steps)*V[y_prime][x_prime] - V[y][x]

            # State-Value Update
            V[y][x] = V[y][x] + alpha_c*delta

            # SARSA Update rule
            Q[y][x][a] = Q[y][x][a] + alpha*delta

            ## The link between option choice and selection
            a = option_index
            # SARSA Update rule
            Q_options[y][x][a] = Q_options[y][x][a] + alpha*delta

            how_many_options += 1
            o_step_list.append(n_o_steps)


            #resetting option settings
            at_door = False
            psuedo_r = 0
            n_o_steps = 0

            # Selecting a new (primitive) option
            agent_loc = new_loc 
            chose_option, action = softmax_option(Q, policy, agent_loc, tau)

        else:

            new_loc, at_goal, reward = make_move(agent_loc, action)
            n_steps +=1

            if at_goal:
                print("Ejo I'm at the goal")

            
            chose_option, action_prime = softmax_option(Q,policy,agent_loc,tau)

            if chose_option:
                action_prime_index = action_prime
            else:
                action_prime_index = moves.index(action_prime)
            # Necessary indices for value functions
            # For the option framework future steps are compared
            # to the initial state in which the option began
            y, x, a = agent_loc[0], agent_loc[1], moves.index(action)
            y_prime, x_prime, a_prime = new_loc[0], new_loc[1], action_prime_index

            # Prediction Error
            if at_goal:
                delta = reward - V[y][x]
            else:
                delta = reward + gamma*V[y_prime][x_prime] - V[y][x]

            # State-Value Update
            V[y][x] = V[y][x] + alpha_c*delta

            # SARSA Update rule
            Q[y][x][a] = Q[y][x][a] + alpha*delta
            
            action = action_prime
            agent_loc = new_loc

    print("End of episode:")
    print(at_goal)
    print(new_loc)
    print(reward)

    ep_step_list.append(n_steps)


print("During goal-training, a total of {} options were selected".format(how_many_options))



x=1
plt.figure(4)
for option in list_of_options:
    plt.subplot(3,2,x)
    plt.imshow(option[0], cmap = 'hot', interpolation = 'nearest')
    x+=1

plt.show()

plt.figure(5)
plt.plot(list(range(len(o_step_list))), o_step_list)
plt.show()

plt.figure(6)
plt.imshow(V, cmap = 'inferno', interpolation = 'nearest')
plt.show()

plt.figure(7)
for o in range(len(list_of_options)):
    plt.subplot(3,2,o+1)
    plt.plot(list(range(len(step_list_bis[o]))), step_list_bis[o])
plt.show()

plt.figure(8)
plt.plot(list(range(n_episodes)),ep_step_list)
plt.show()
