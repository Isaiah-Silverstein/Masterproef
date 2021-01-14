import numpy as np 
import matplotlib.pyplot as plt
import matplotlib as mpl
import os


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
states = create_grid_from_file('Change to where maps are /map4.txt')


os.chdir(" Change to where the data is")
A_root_log = np.load("A_root_HRL.npy")
A_top_log = np.load("A_top_log_HRL.npy")
loc_log = np.load("loc_log_HRL.npy")
S_log = np.load("S_log_HRL.npy")
S_root_log = np.load("S_root_HRL.npy")
MFC_root_log = np.load("MFC_root_HRL.npy")
MFC_top_log = np.load("MFC_top_HRL.npy")
Hit = np.load("Hit_HRL.npy")
Be = np.load("Be_HRL.npy")




move_name=["UP", "R-UP", "RIGHT","R-DOWN","DOWN","L-DOWN", "LEFT" ,"LEFT-UP","Go To door 2","Go to door 3"] 
n_actions =  A_root_log.shape[0]
n_moves = A_top_log.shape[0]
total_time = A_root_log.shape[1]


n_steps = 20 
n_options = 8
sync_root = np.zeros((loc_log.shape[0],loc_log.shape[1],n_actions,20))
sync_top = np.zeros((n_options,loc_log.shape[0],loc_log.shape[1],n_moves,20))
sync_map = np.zeros((loc_log.shape[0],loc_log.shape[1]))
sync_map_top = np.zeros((n_options,loc_log.shape[0],loc_log.shape[1]))
for y in range(loc_log.shape[0]):
    for x in range(loc_log.shape[1]):
        if states[y,x] == 0:
            for a in range(n_actions):
                for step in range(n_steps):
                        if a < n_moves:
                            sync_root[y,x,a,step] = np.corrcoef(S_root_log[y,x,:,step]**2,A_root_log[a,:,step]**2)[0,1]
                            for o in range(n_options):
                                sync_top[o,y,x,a,step] = np.corrcoef(S_log[o,y,x,:,step]**2,A_top_log[a,:,step]**2)[0,1]
                        else:
                            sync_root[y,x,a,step] = np.corrcoef(S_root_log[y,x,:,step]**2,A_root_log[a,:,step]**2)[0,1]

            

            sync_map[y,x] = np.nanmean(np.nanmean(np.absolute(sync_root[y,x,:,:]),axis = 0))
            for o in range(n_options):
                sync_map_top[o,y,x] = np.nanmean(np.nanmean(np.absolute(sync_top[o,y,x,:,:]),axis = 0))
# print(sync_map_top)
plt.imshow(sync_map, cmap = 'hot',vmin =0,vmax=0.3,interpolation = 'nearest')
plt.show()
for i in range(n_options):
    plt.imshow(sync_map_top[i], cmap = 'hot',vmin =0,vmax=0.3,interpolation = 'nearest')
    plt.show()




theta_gamma_r = np.zeros((states.shape[0],states.shape[1]))
theta_gamma_top = np.zeros((n_options,states.shape[0],states.shape[1]))

#Theta en gamma power
for y in range(loc_log.shape[0]):
    for x in range(loc_log.shape[1]):
        if states[y,x] == 0:
            sync_r_list = []
            sync_t_list = [[] for _ in range(n_options)]
            for step in range(n_steps):
                #Root level
                sync_r_list.append(np.absolute(np.corrcoef(S_root_log[y,x,:,step]**2, MFC_root_log[:,step]**2)[0,1]))
                #Top Level
                for o in range(n_options):
                    
                    sync_t_list[o].append(np.absolute(np.corrcoef(S_log[o,y,x,:,step]**2,MFC_top_log[:,step]**2)[0,1]))
            theta_gamma_r[y,x] = np.nanmean(sync_r_list)
            for o in range(n_options):
                theta_gamma_top[o,y,x] = np.nanmean(sync_t_list[o])
plt.imshow(theta_gamma_r,cmap ='jet',interpolation = "nearest",vmin =0,vmax=0.1)
plt.colorbar()
plt.show()


for i in range(n_options):
    plt.imshow(theta_gamma_top[i], cmap = 'jet',vmin =0,vmax=0.1,interpolation = 'nearest')
    plt.colorbar()
    plt.show()





# for i in range(n_steps):
#     print(sync[:,:,0,i])
# 
mpl.style.use('seaborn-dark')

step =  0
current_loc = np.where(loc_log[:,:,0] == step+1)
print(current_loc)
for a in range(10):
    print(np.corrcoef(S_root_log[current_loc[0],current_loc[1],:,step],A_root_log[a,:,step])[0,1])
#print(S_log[current_loc[0], current_loc[1],:,step])
plot = True
if plot:
    
        #action
    fig, axs = plt.subplots(2)
    axs[0].plot(np.arange(int(total_time)),S_root_log[current_loc[0],current_loc[1],:,step][0], color = "k", label = "E: current state")
    axs[0].set_ylabel("Amplitude")
    
    axs[1].plot(np.arange(int(total_time)),Hit[:,step],color = "black",linewidth = 0.7, alpha = 0.8)
    axs[1].plot(np.arange(int(total_time)),MFC_root_log[:,step],color = "blue",label = "E: pMFC",linewidth = 1, alpha = 0.8)
    axs[1].set_ylabel("Amplitude")
    axs[1].legend(loc = "upper right")

    for i in range(2):
        axs[0].plot(np.arange(int(total_time)),A_root_log[8+i,:,step],linewidth = 1, alpha = 0.8, color = ["b","g"][i], label = ["E: relevant option","E: irrelevant option"][i])
    axs[0].legend(loc = "upper right")
    plt.tight_layout()
    plt.show()



print(loc_log[:,:,0])
print(loc_log[:,:,1])
