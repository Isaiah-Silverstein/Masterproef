import numpy as np
import random
import pandas as pd
import seaborn as sb 
import matplotlib.pyplot as plt 
import scipy
import os
import seaborn as sb 
import matplotlib as mpl



#Change to correct directory
os.chdir("/home/isaiah/Documents/Thesis/Code/Analysis/SIM3")
tau_values = pd.read_csv("Tau_Values")     #insert the observations for the different Tau values
radius_values = pd.read_csv("Radius_Values")      #insert the observations for the different Radius values


tau_values.columns = ["N_sim","Alpha", "Beta", "Tau", "Exploration Rate"]      
radius_values.columns = ["N_sim","Alpha", "Beta", "Radius", "Exploration Rate", "Sync Rate"]



alpha_list = tau_values.Alpha.unique()       #list of unique alpha values
beta_list = tau_values.Beta.unique()         #list of unique alpha values
tau_list = tau_values.Tau.unique()           #list of unique tau values
radius_list = radius_values.Radius.unique()  #list of unique radius values

print(radius_list)
print(tau_list)


explore =0
#sanity check
assert len(radius_list) == len(tau_list), "Parameter sets don't contain equal observations, n1 = {0} and n2 = {1}".format(len(radius_list),len(tau_list))
assert explore ==0,"no plots this time"
#logging table
data_table = np.zeros((len(alpha_list), len(beta_list), len(tau_list),2))
stat_table = np.zeros((len(alpha_list), len(beta_list),2)) 

a = 0
for alpha in alpha_list:
    b = 0
    for beta in beta_list:
        
        alpha_taus = tau_values[tau_values["Alpha"] == alpha]
        relevant_taus = alpha_taus[alpha_taus["Beta"] == beta]

        alpha_radius = radius_values[radius_values["Alpha"] == alpha]
        relevant_radius = alpha_radius[alpha_radius["Beta"] == beta]

        stat_table[a,b,:] = scipy.stats.spearmanr(relevant_taus["Exploration Rate"],relevant_radius["Exploration Rate"])
        data_table[a,b,:,0] = relevant_taus["Exploration Rate"]
        data_table[a,b,:,1] = relevant_radius["Exploration Rate"]
        b+=1
    a+=1

x = [round(a,2) for a in alpha_list[0:9]]
y = [round(b,2) for b in beta_list[0:9]]

z = stat_table[0:9,0:9,0]


#print(stat_table)


ax = sb.heatmap(z , vmin=-1,vmax=1, cmap='jet', xticklabels=x, yticklabels=y )
ax.invert_yaxis()
ax.set_ylabel("Beta Values")
ax.set_xlabel("Alpha Values")
ax.set_title("Different Learning Rate Correlations")
plt.show()


k = 12
assert k == 12, "stop"




mpl.style.use('seaborn-dark')

"""Commented for when I only had 4 plots"""
fig, axs = plt.subplots(2,2)

axs[0,0].title.set_text("Alpha: {0}; Beta: {1} | r={2}, p={3}".format(round(alpha_list[1],2), round(beta_list[1],2),round(stat_table[0,0,0],3),round(stat_table[0,0,1],3)))
axs[0,0].plot(np.arange(len(tau_list)),data_table[1,1,:,0], "-o", color = "blue", label = "Tau values", linewidth= 0.8, alpha = .7)
axs[0,0].plot(np.arange(len(radius_list)), data_table[1,1,:,1], "--o", color = "red", label= "Radius values",linewidth= 0.8, alpha = 0.6)
axs[0,0].legend(loc = 'upper left')
axs[0,0].set_ylabel("Exploration rate")
axs[0,0].set_xlabel("Parameter Size")

axs[0,1].title.set_text("Alpha: {0}; Beta: {1} | r={2}, p={3}".format(round(alpha_list[2],2),round(beta_list[1],2),round(stat_table[0,1,0],3),round(stat_table[0,1,1],3)))
axs[0,1].plot(np.arange(len(tau_list)),data_table[2,1,:,0], "-o", color = "blue", label = "Tau values", linewidth= 0.8, alpha = .6)
axs[0,1].plot(np.arange(len(radius_list)), data_table[2,1,:,1], "--o", color = "red", label= "Radius values", linewidth= 0.8, alpha = .6)
axs[0,1].legend(loc = 'upper left')
axs[0,1].set_ylabel("Exploration rate")
axs[0,1].set_xlabel("Parameter Size")

axs[1,0].title.set_text("Alpha: {0}; Beta: {1} | r={2}, p={3}".format(round(alpha_list[1],2), round(beta_list[2],2),round(stat_table[1,0,0],3),round(stat_table[1,0,1],3)))
axs[1,0].plot(np.arange(len(tau_list)),data_table[1,2,:,0], "-o", color = "blue", label = "Tau values", linewidth= 0.8, alpha = .6)
axs[1,0].plot(np.arange(len(radius_list)), data_table[1,2,:,1], "--o", color = "red", label= "Radius values",linewidth= 0.8, alpha = .6)
axs[1,0].legend(loc = 'upper left')
axs[1,0].set_ylabel("Exploration rate")
axs[1,0].set_xlabel("Parameter Size")

axs[1,1].title.set_text("Alpha: {0}; Beta: {1} | r={2}, p={3}".format(round(alpha_list[2],2), round(beta_list[2],2),round(stat_table[1,1,0],3),round(stat_table[1,1,1],3)))
axs[1,1].plot(np.arange(len(tau_list)),data_table[2,2,:,0], "-o", color = "blue", label = "Tau values", linewidth= 0.8, alpha = .6)
axs[1,1].plot(np.arange(len(radius_list)), data_table[2,2,:,1], "--o", color = "red", label= "Radius values",linewidth= 0.8, alpha = .6)
axs[1,1].legend(loc = 'upper left')
axs[1,1].set_ylabel("Exploration rate")
axs[1,1].set_xlabel("Parameter Size")

plt.tight_layout()
plt.show()




        
