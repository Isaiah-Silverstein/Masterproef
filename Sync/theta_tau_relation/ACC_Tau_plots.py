import numpy as np
import random
import pandas as pd
import seaborn as sb 
import matplotlib.pyplot as plt 
import scipy



tau_values = pd.read_csv("Tau_Values")      #insert the observations for the different Tau values
radius_values = pd.read_csv("Radius_Values")#insert the observations for the different Radius values

tau_values.columns = ["Alpha", "Beta", "Tau", "Exploration Rate"]      
radius_values.columns = ["Alpha", "Beta", "Radius", "Exploration Rate"]


alpha_list = tau_values.Alpha.unique()       #list of unique alpha values
beta_list = tau_values.Beta.unique()         #list of unique alpha values
tau_list = tau_values.Tau.unique()           #list of unique tau values
radius_list = radius_values.Radius.unique()  #list of unique radius values


#sanity check
assert len(radius_list) == len(tau_list), "Parameter sets don't contain equal observations, n1 = {0} and n2 = {1}".format(len(radius_list),len(tau_list))


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

print(stat_table)


fig, axs = plt.subplots(2,2)

axs[0,0].title.set_text("Alpha: {0}; Beta: {1} | r={2}, p={3}".format(alpha_list[0], beta_list[0],round(stat_table[0,0,0],3),round(stat_table[0,0,1],3)))
axs[0,0].plot(np.arange(len(tau_list)),data_table[0,0,:,0], "-", color = "blue", label = "Tau values")
axs[0,0].plot(np.arange(len(radius_list)), data_table[0,0,:,1], "--", color = "red", label= "Radius values")

axs[0,1].title.set_text("Alpha: {0}; Beta: {1} | r={2}, p={3}".format(alpha_list[0], beta_list[1],round(stat_table[0,1,0],3),round(stat_table[0,1,1],3)))
axs[0,1].plot(np.arange(len(tau_list)),data_table[0,1,:,0], "-", color = "blue", label = "Tau values")
axs[0,1].plot(np.arange(len(radius_list)), data_table[0,1,:,1], "--", color = "red", label= "Radius values")

axs[1,0].title.set_text("Alpha: {0}; Beta: {1} | r={2}, p={3}".format(alpha_list[1], beta_list[0],round(stat_table[1,0,0],3),round(stat_table[1,0,1],3)))
axs[1,0].plot(np.arange(len(tau_list)),data_table[1,0,:,0], "-", color = "blue", label = "Tau values")
axs[1,0].plot(np.arange(len(radius_list)), data_table[1,0,:,1], "--", color = "red", label= "Radius values")


axs[1,1].title.set_text("Alpha: {0}; Beta: {1} | r={2}, p={3}".format(alpha_list[1], beta_list[1],round(stat_table[1,1,0],3),round(stat_table[1,1,1],3)))
axs[1,1].plot(np.arange(len(tau_list)),data_table[1,1,:,0], "-", color = "blue", label = "Tau values")
axs[1,1].plot(np.arange(len(radius_list)), data_table[1,1,:,1], "--", color = "red", label= "Radius values")

plt.tight_layout()
plt.show()




        
