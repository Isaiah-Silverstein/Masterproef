import numpy as np 
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
print(os.getcwd())


def create_grid_from_file(map_file = "enter file name"):
    def filter(c):
        if c == 'W':
            return 1    # 'WALL'
        else:
            return 0
    # opening up the text file and putting the elements into an array
    with open(map_file) as file:
        grid_list = []
        #read the map file line per line
        for line in file.readlines():
            line = line.split()
            grid_list.append(list(map(filter, line)))
            grid_array = np.array(grid_list)

    return grid_array
states = create_grid_from_file('/home/isaiah/Documents/Thesis/Code/Models/Maps/map4.txt')


os.chdir("/home/isaiah/Documents/Thesis/Code/Models/SIM5/S5Data/RL/")
converged = True
Accuracy = np.load("Accuracy.npy")
if converged:
    A_Plog = np.load("A_Plog220.npy")
    A_log = np.load("A_log220.npy")
    loc_log = np.load("loc_log220.npy")
    S_log = np.load("S_log220.npy")
    S_Plog = np.load("S_Plog220.npy")
    MFC_log = np.load("MFC_log220.npy")
    Hit = np.load("Hit220.npy")
    Be = np.load("Bernoulli220.npy")

else:
    A_Plog = np.load("A_Plog6.npy")
    A_log = np.load("A_log6.npy")
    loc_log = np.load("loc_log6.npy")
    S_log = np.load("S_log6.npy")
    S_Plog = np.load("S_Plog6.npy")
    MFC_log = np.load("MFC_log6.npy")
    Hit = np.load("Hit6.npy")
    Be = np.load("Bernoulli6.npy")
move_name=["UP", "R-UP", "RIGHT","R-DOWN","DOWN","L-DOWN", "LEFT" ,"LEFT-UP","Go To door 2","Go to door 3"] 
n_actions =  A_log.shape[0]
print( A_Plog.shape, S_log.shape)

total_time = A_log.shape[1]
n_steps =int(np.max(loc_log))-1

sync = np.zeros((loc_log.shape[0],loc_log.shape[1],n_actions,n_steps))

sync_map = np.zeros((loc_log.shape[0],loc_log.shape[1]))

# for y in range(loc_log.shape[0]):
#     for x in range(loc_log.shape[1]):
#         if states[y,x] == 0:
#             for a in range(n_actions):
#                 for step in range(n_steps):

#                         sync[y,x,a,step] = np.corrcoef(S_Plog[y,x,:,step]**2,A_Plog[a,:,step]**2)[0,1]
        
#             sync_map[y,x] = np.nanmean(np.nanmean(np.absolute(sync[y,x,:,:]**2),axis = 0))

# # print(sync_map_top)
# print(sync_map)
# plt.imshow(sync_map, cmap = 'jet',vmin =0,vmax=0.1,interpolation = 'nearest')
# plt.colorbar()
# plt.show()








theta_gamma = np.zeros((states.shape[0],states.shape[1]))

sync_list = []

# #Theta en gamma power
# for y in range(loc_log.shape[0]):
#     for x in range(loc_log.shape[1]):
#         if states[y,x] == 0:
#             sync_list = []
#             for step in range(n_steps):
               
#                 sync_list.append(np.corrcoef(S_Plog[y,x,:,step]**2,Be[:,step]**2)[0,1])

#             theta_gamma[y,x] = np.nanmean(sync_list)

# plt.imshow(theta_gamma,cmap ='jet',interpolation = "nearest",vmin =0,vmax=0.1)
# plt.colorbar()

# plt.show()
# print(theta_gamma)





# for i in range(n_steps):
#     print(sync[:,:,0,i])
# 
mpl.style.use('seaborn-dark')






theta_gamma_list  = []
sync_list = np.zeros((n_actions,n_steps))
for step in range(n_steps):
    current_loc = np.where(loc_log[:,:] == step+1)
    theta_gamma_list.append(np.corrcoef(Be[:,step]**2,np.mean(A_Plog[:,:,step]**2,axis=0))[0,1])

    for a in range(n_actions):
        sync_list[a,step] = np.corrcoef(S_Plog[current_loc[0],current_loc[1],:,step],A_Plog[a,:,step])[0,1]




# fig,axs=plt.subplots(2)
# axs[0].plot(np.arange(n_steps),theta_gamma_list)
# axs[0].set_ylim([-0.1,0.8])

# for a in range(n_actions):
#     axs[1].plot(np.arange(n_steps),sync_list[a,:])
# axs[1].set_ylim([-1,1])
# plt.show()

#print(S_log[current_loc[0], current_loc[1],:,step],A_Plog[i,:,step])
plot = True


for a in range(n_actions):
    print(np.corrcoef(S_log[current_loc[0], current_loc[1],:,step],A_Plog[a,:,step])[0,1])

if plot:
    
        #action
    fig, axs = plt.subplots(2)
    axs[0].plot(np.arange(int(total_time)),S_Plog[current_loc[0],current_loc[1],:,step][0], color = "k", label = "E: current state")
    axs[0].set_ylabel("Amplitude")
    
    axs[1].plot(np.arange(int(total_time)),Hit[:,step],color = "black",linewidth = 0.7, alpha = 0.8)
    axs[1].plot(np.arange(int(total_time)),MFC_log[:,step],color = "blue",label = "E: pMFC",linewidth = 1, alpha = 0.8)
    axs[1].set_ylabel("Amplitude")
    axs[1].legend(loc = "upper right")
    #axs[1].plot(np.arange(int(total_time)),Be[:,step])
    for i in range(2):
        axs[0].plot(np.arange(int(total_time)),A_Plog[4+i,:,step],linewidth = 1, alpha = 0.8, color = ["b","g"][i], label = ["E: relevant action","E: irrelevant action"][i])
    axs[0].legend(loc = "upper right")
    plt.tight_layout()
    plt.show()


