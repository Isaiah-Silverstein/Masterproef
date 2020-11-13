#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 10:01:43 2019

@author: pieter
"""

import estimating_Parameters as est
import estimation_fun as estimation
import StimCode_Lik as stim_base
import StimCode_Mod_Lik as stim_mod
import FeatureCode_1layer_Lik as feat_easy_base
import FeatureCode_1layer_Mod_Lik as feat_easy_mod
import FeatureCode_2layer_Lik as feat_hard_base
import FeatureCode_2layer_Mod_Lik as feat_hard_mod
import StimCode_Sim as stim_base_sim
import StimCode_Mod_Sim as stim_mod_sim
import FeatureCode_1layer_Sim as feat_easy_base_sim
import FeatureCode_1layer_Mod_Sim as feat_easy_mod_sim
import FeatureCode_2layer_Sim as feat_hard_base_sim
import FeatureCode_2layer_Mod_Sim as feat_hard_mod_sim

import numpy as np
import pandas as pd
import os

pplist=np.arange(1,40)
pplist=np.delete(pplist,[3,4,16,18,20,21,27,34])

#pplist=[1]

nModels=8

def participant_fitting(core=0, pplist = pplist):
    if core ==0:
        s=est.model_fit_procedure(likfun =  stim_base, simfun=stim_base_sim, condition="Easy", model="base", pplist=pplist)
    elif core ==1:
        s=est.model_fit_procedure(likfun =  stim_base, simfun=stim_base_sim, condition="Hard", model="base", pplist=pplist)
    elif core ==2:
        s=est.model_fit_procedure(likfun =  stim_mod, simfun=stim_mod_sim, condition="Easy", model="mod", pplist=pplist)
    elif core ==3:
        s=est.model_fit_procedure(likfun =  stim_mod, simfun=stim_mod_sim, condition="Hard", model="mod", pplist=pplist)
    elif core ==4:
        s=est.model_fit_procedure(likfun =  feat_easy_base, simfun=feat_easy_base_sim, condition="Easy", model="base", pplist=pplist)
    elif core ==5:
        s=est.model_fit_procedure(likfun =  feat_hard_base, simfun=feat_hard_base_sim, condition="Hard", model="base", pplist=pplist)
    elif core ==6:
        s=est.model_fit_procedure(likfun =  feat_easy_mod, simfun=feat_easy_mod_sim, condition="Easy", model="mod", pplist=pplist)
    elif core ==7:
        s=est.model_fit_procedure(likfun =  feat_hard_mod, simfun=feat_hard_mod_sim, condition="Hard", model="mod", pplist=pplist)

    successrate= s/len(pplist)

from   multiprocessing import Process, cpu_count

cp = cpu_count()
if cp+1 > nModels:
    cpus=nModels
else:
    cpus = cp

worker_pool = []

for c in range(cpus):
    proc = Process(target = participant_fitting, args=(c,pplist))
    proc.start()
    worker_pool.append(proc)

    for processes in worker_pool:
        processes.join()  # Wait for all of the workers to finish.
