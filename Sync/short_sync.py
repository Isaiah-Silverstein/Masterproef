# shorter version of the figureitout.py version of mehdi's task


import numpy as np
import random
import matplotlib.pyplot as plt


def update_phase(node = [], radius = 1, damp = 0.3, coupling = 0.3, multiple = True):
    """ updating the phase per step"""
    if multiple:
        Phase = np.zeros((len(node[:,0]),2))
        r2 = np.sum(node*node,axis = 1)
        E = node[:,0]
        I = node[:,1]
        #excitatory update formula
        """E_i(t + dt) = E_i(t) - C * I_i(t) - D * J(r > r_min) * E_i(t)""" #Burst??
        Phase[:,0] = E - coupling*I - damp*((r2>radius).astype(int))*E
        """I_i(t + dt) = I_i(t) + C * E_i(t) - D * J(r > r_min) * I_i(t)"""
        Phase[:,1] = I + coupling*E - damp*((r2>radius).astype(int))*I

    else:
        Phase = np.zeros((2))
        r2 = np.sum(node[0]*node[1],axis = 0)
        E = node[0]
        I = node[1]
        #excitatory update formula
        """E_i(t + dt) = E_i(t) - C * I_i(t) - D * J(r > r_min) * E_i(t)""" #Burst??
        Phase[0] = E - coupling*I- damp*((r2>radius).astype(int))*E
        """I_i(t + dt) = I_i(t) + C * E_i(t) - D * J(r > r_min) * I_i(t)"""
        Phase[1] = I + coupling*E - damp*((r2>radius).astype(int))*I

    return Phase




# Initial Variables
n_instr = 2 # if circle select circle if square select square
nStim = 2  #A circle or a square
nResp = 2  #A circle or a square

n_trials = n_instr*nStim  # total number of trials

#Thresholding
Threshold = 5
#drift rate
drift = 2

# Time and sample rate
srate=500                                               #sampling rate
Instr_time=int(.2 * srate)                              #Instruction presentation (200 ms)
Stim_time=int(.05 * srate)                              #Stimulus presentation of 50 ms
Resp_time=1 * srate                                     #max response time of 1s
Response_deadline=1 * srate                             #Response deadline

# Design
Design=np.zeros((4,2))

Design[0:2,:] = [0,0]
Design[2:4,:] = [1,1]
Design = Design.astype(int)
print(Design)

k = 1
assert k == 1, "stop here"


#max trial time
total_steps=(Instr_time + Stim_time  + Resp_time)


# # # # # # # # #
#  Processing   #
#    Module     #
# # # # # # # # #

#initial variables
r2max = 1                           # max amplitude
cg_1 = (30/srate)*2*np.pi           #couploing parameter for gamma waves
cg_2 = cg_1+(drift/srate)*2*np.pi   #coupling parameter for gamma waves with
                                    # with a frequency difference of 2 Hz

damp = 0.3                          # damping parameter
decay = 0.9                         # decay parameter
noise = 0.05                        # noise parameter

nUnits = nStim + nResp              #2 input nodes + 2 output nodes

#Initializing matrices

Phase      = np.zeros((nUnits,2,total_steps,n_trials))  #phase code neurons
Rate       = np.zeros((nUnits,total_steps,n_trials))       #rate code neurons


#weights initialization
W = np.ones((nStim,nResp))*0.5
W[0,1]=0.1
W[1,0]=0.1
print(W)


#integrator layer
Integr = np.zeros((nResp))
inh = np.ones((nResp,nResp))*-0.01
for i in range(nResp):
    inh[i,i]=0

cumulative = 1

# # # # # # # # #
#  Control      #
#    Module     #
# # # # # # # # #


# MFC #

r2_MFC = 1                   # radius MFC
ct = (5/srate)*2*np.pi       # coupling parameter for theta waves
damp_MFC = .003              # damping parameter
acc_slope = 10               # MFC slope parameter ---> steepness of burst

MFC = np.zeros((2,total_steps,n_trials)) # MFC phase units
Be = 0                                   # Bernoulli rate code MFC

# LFC #
LFC = np.zeros((n_instr,n_trials))

LFC_sync = np.zeros((n_instr,nStim))

# What response set does the LFC sync with

# Either  Square [0] or Circle [1]
LFC_sync[0,:] = [0,2]                 # Square with square response
LFC_sync[1,:] = [1,3]                 # circle with circle response


Instr_activation=np.diag(np.ones((2)))     # Instruction activation matrix
Stim_activation= np.zeros((nStim,nResp))
Stim_activation[0,:] = [1,0]    #square
Stim_activation[1,:] = [0,1]    #circle



# _________ simulation ___________ #



#starting points of oscillations
start = np.random.random((nUnits,2))       # random starting points
start_MFC = np.random.random((2))          # acc starting points
#assign to relevant matrices
Phase[:,:,0,0]=start
MFC[:,0,0]=start_MFC


#basic dependent variable
Hit = np.zeros((total_steps,n_trials)) # record when there is a hit
RT  = np.zeros((n_trials))             # record the reaction times per trial
accuracy  = np.zeros((n_trials))       # accuracy record
response = np.zeros((n_trials))        # record response
resp = np.ones((n_trials))*-1

#recording trial onset details
stim_lock   = np.zeros((n_trials))     # record stimulus
inst_lock   = np.zeros((n_trials))     # record instruction

#recording sync
sync = np.zeros((nStim,nResp, n_trials))  # record sync


## Trial Loop ##
time = 0
for trial in range(n_trials):



    #setting up the starting points for the trial
    if trial>0:

        Phase[:,:,0,trial]=Phase[:,:,time,trial-1]
        MFC[:,0,trial]=MFC[:,time,trial-1]

    t = time
    #phase reset
    MFC[:,t,trial]=np.ones((2))*r2_MFC

    # 1 #
    """ Instruction presentation and preperation period
        -> during this phase synchronisation begins because of the instruction but there's no 'visual' stimulation """
    for time in range(Instr_time):

        # LFC setting instruction
        LFC[Design[trial,0],trial] = 1

        # Updating phase code units of the processing module
        Phase[0:nStim,:,time+1,trial]=update_phase(node=Phase[0:nStim,:,time,trial], radius=r2max, damp=damp, coupling=cg_1, multiple=True)
        Phase[nStim:nUnits,:,time+1,trial]=update_phase(node=Phase[nStim:nUnits,:,time,trial], radius=r2max, damp=damp, coupling=cg_2, multiple=True)
        #Updating phase code units of the MFC
        MFC[:,time+1,trial]=update_phase(node=MFC[:,time,trial], radius=r2_MFC, damp=damp_MFC, coupling=ct, multiple=False)

        #Bernoulli process in MFC rate
        Be = 1/(1+np.exp(-acc_slope*(MFC[0,time,trial]-1)))
        p = random.random()

        #Burst
        if p < Be:

            Hit[time,trial] = 1 #logging hit

            Gaussian = np.random.normal(size=[1,2])

            for Instr in range(n_instr):

                if LFC[Instr,trial]:
                    for nodes in LFC_sync[Instr,:]:

                        Phase[int(nodes),:,time+1,trial] = decay*Phase[int(nodes),:,time,trial]+Gaussian


    t = time
    stim_lock[trial] = t
    # 2 #
    """ Response period where the stimulus is given and the model must provide an answer """
    for time in range(t, int(t + Stim_time+Resp_time)):


        # Updating phase code units of the processing module
        Phase[0:nStim,:,time+1,trial]=update_phase(node=Phase[0:nStim,:,time,trial], radius=r2max, damp=damp, coupling=cg_1, multiple=True)
        Phase[nStim:nUnits,:,time+1,trial]=update_phase(node=Phase[nStim:nUnits,:,time,trial], radius=r2max, damp=damp, coupling=cg_2, multiple=True)
        #Updating phase code units of the MFC
        MFC[:,time+1,trial]=update_phase(node=MFC[:,time,trial], radius=r2_MFC, damp=damp_MFC, coupling=ct, multiple=False)

        #Bernoulli process in MFC rate
        Be = 1/(1+np.exp(-acc_slope*(MFC[0,time,trial]-1)))

        p = np.random.random()

        #Burst
        if p < Be:
            Hit[time,trial] = 1 #logging hit

            Gaussian = np.random.normal(size=[1,2])
            for Instr in range(n_instr):
                if LFC[Instr,trial]:
                    for nodes in LFC_sync[Instr,:]:
                        Phase[int(nodes),:,time+1,trial] = decay*Phase[int(nodes),:,time,trial]+Gaussian


        #Updating rate code units
        Rate[0:nStim,time,trial]=Stim_activation[Design[trial,1],:]*(1/(1+np.exp(-5*Phase[0:nStim,0,time,trial]-0.6)))
        Rate[nStim:nUnits,time,trial]=np.matmul(Rate[0:nStim,time,trial],W)*(1/(1+np.exp(-5*Phase[nStim:nUnits,0,time,trial]-0.6)))

    resp[trial] = np.argmax(np.max(Rate[nStim:nUnits,:,trial],axis=1))#list(Rate[nStim:nUnits]).index(max(Rate[nStim:nUnits]))
    RT[trial]=(time-t)*(1000/srate)
    t = time
    response[trial]=t

    # Accuracy check after response
    if Design[trial,0]==0:
        if resp[trial]==0:
            accuracy[trial]=1
        else:
            accuracy[trial]=0
    if Design[trial,0]==1:
        if resp[trial]==1:
            accuracy[trial]=1
        else:
            accuracy[trial]=0

    for st in range(nStim):
        for rs in range(nResp):
            sync[st,rs, trial]=np.corrcoef(Phase[st,0,int(stim_lock [trial]):int(response[trial]),trial],Phase[nStim+rs,0,int(stim_lock[trial]):int(response[trial]),trial])[0,1]
print(accuracy)
print(resp)



print(Phase.shape)




for trial in range(n_trials):

    fig, axs = plt.subplots(4)
    fig.suptitle('Phase code units of input and response nodes {0}'.format(str(trial)))
    axs[0].plot(np.arange(int(Phase.shape[2])),Phase[0,0,:,trial])
    axs[0].plot(np.arange(int(Phase.shape[2])),Phase[2,0,:,trial])
    axs[0].plot(np.arange(int(Phase.shape[2])),Hit[:,trial])
    axs[1].plot(np.arange(int(Phase.shape[2])),Phase[1,0,:,trial])
    axs[1].plot(np.arange(int(Phase.shape[2])),Phase[3,0,:,trial])
    axs[1].plot(np.arange(int(Phase.shape[2])),Hit[:,trial])
    axs[2].plot(np.arange(int(Rate.shape[1])),Rate[0,:,trial])
    axs[2].plot(np.arange(int(Rate.shape[1])),Rate[2,:,trial])
    axs[3].plot(np.arange(int(Rate.shape[1])),Rate[1,:,trial])
    axs[3].plot(np.arange(int(Rate.shape[1])),Rate[3,:,trial])
    plt.show()

# plt.plot(np.arange(int(Phase.shape[2])),Phase[0,0,:,0])
# plt.plot(np.arange(int(Phase.shape[2])),Phase[2,0,:,0])
# plt.plot(np.arange(int(Phase.shape[2])),Hit[:,0])
# plt.title("First trial phase code neuron sync of square unit")
# plt.show()

# plt.plot(np.arange(int(MFC.shape[1])),MFC[0,:,1])
# plt.plot(np.arange(int(MFC.shape[1])),MFC[1,:,1])
# plt.show()
