#!/usr/bin/env python3
import sys
import traceback
import pdb
import cProfile
import os
import csv
import subprocess
import tempfile
import time
import torch
from src.config import config
from src.vnnlib_fuzz import mk_vnnlib
from src.onnx_fuzz import mk_onnx
import numpy as np
import random
from mapleDNNsat import Solver  ##MapleDNNsat run

##A random function
def rng(n): 
	return np.random.randint(1, n)

def run_mapleDNNsat(vnnlib_file,onnx_file):
	timeout = 15
	strat_time = time.time()	
	try:	
		subprocess.call('mapleDNNsat --network '+onnx_file +' --property '+vnnlib_file,shell=True, timeout=timeout)
		elapsed = (time.time() - strat_time)
		return elapsed
	except:
		if (time.time() - strat_time)>timeout:
			return timeout*2
		else:
			return False
	
def run_nnenum(vnnlib_file,onnx_file):
	#return the par-2 score when running nnenum
	pass

def run_dnnf(vnnlib_file,onnx_file):
	timeout = 15
	start_time = time.time()
	try:
		subprocess.call('dnnf --vnnlib '+vnnlib_file +' --network N '+onnx_file,shell=True, timeout=timeout)
		elapsed = (time.time() - start_time)
		return elapsed
	except:
		if (time.time() - start_time)>timeout:
			return timeout*2
		else:
			return False

def scoring_function(vnnlib_file,onnx_file):
	time_maplednnsat = run_mapleDNNsat(vnnlib_file,onnx_file)
	time_dnnf = run_dnnf(vnnlib_file,onnx_file)
	if (time_maplednnsat == False) or (time_dnnf == False):
		return False
	else:
		difference = time_dnnf - time_maplednnsat
		#per_diff = (time_maplednnsat - time_dnnf)/time_dnnf
		return difference
		#return per_diff

def get_reward(previous_onnx,previous_vnnlib,updated_onnx,updated_vnnlib):
	if scoring_function(previous_vnnlib,previous_onnx) == False or scoring_function(updated_vnnlib,updated_onnx)==False:
		return False
	elif scoring_function(previous_vnnlib,previous_onnx) < scoring_function(updated_vnnlib,updated_onnx):
		return True
	else:
		return False
    	
	
class ThompsonSampling:	
	def __init__(self,n_actions,decay=0.95):
		self.n_it = 0
		self.n_actions = n_actions
		self.k = 1
		self.action = None
		self.alpha_beta = [[1,1] for i in range(self.n_actions)]
		self.decay = decay

	def select_action(self):
		samples = [0] * self.n_actions
		for a in range(self.n_actions):
			for i in range(self.k):
				samples[a]  += np.random.beta(self.alpha_beta[a][0],self.alpha_beta[a][1])				
		self.action = np.argmax(samples)
		return self.action
	
	def reward(self,reward):
		if reward:
			self.alpha_beta[self.action][0] = 1 + self.alpha_beta[self.action][0] * self.decay
			self.alpha_beta[self.action][1] = 0 + self.alpha_beta[self.action][1] * self.decay
		else:
			self.alpha_beta[self.action][0] = 0 + self.alpha_beta[self.action][0] * self.decay
			self.alpha_beta[self.action][1] = 1 + self.alpha_beta[self.action][1] * self.decay
		self.n_it += 1



if __name__ == '__main__':
	for j in range(45,60):
		n_actions = 5 #First index represent +1, second index represent +0
		n2_actions = 2
		hisotry = [0]*n_actions
		hisotry2 = [0]*n2_actions
		agent = ThompsonSampling(n_actions,decay=0.92)
		agent2 = ThompsonSampling(n2_actions,decay=0.92)
		n_steps = 80
		#Initial state:
		initial = [rng(10),rng(10),rng(10),rng(10),rng(10)]
		DNN = mk_onnx(in_size=initial[0],out_size=initial[1],depth=initial[2],width_layer=initial[3],range=initial[4])
		VNNLIB = mk_vnnlib(True,DNN)
		DNN.write_onnx(filename='initial'+str(j)+'.onnx')
		VNNLIB.write_vnnlib(filename='initial'+str(j)+'.vnnlib')
		DNN.write_onnx(filename='best'+str(j)+'.onnx')
		VNNLIB.write_vnnlib(filename='best'+str(j)+'.vnnlib')
		for i in range(n_steps):
			print('************************************************'+str(i))
			action = agent.select_action()
			action2 = agent2.select_action()
			hisotry[action]+=1
			hisotry2[action2]+=1
			new_state = initial.copy()
			noise = (action2)*rng(2) + (1-action2)*(-1*rng(2))
			new_state[action] = new_state[action] + (action2)*rng(2) + (1-action2)*(-1*rng(2))
			if new_state[action] <= 0:
				new_state[action] = 1
			DNN_new = mk_onnx(in_size=new_state[0],out_size=new_state[1],depth=new_state[2],width_layer=new_state[3],range=new_state[4])
			VNNLIB_new = mk_vnnlib(True,DNN_new)
			DNN_new.write_onnx(filename='new'+str(j)+'.onnx')
			VNNLIB_new.write_vnnlib(filename='new'+str(j)+'.vnnlib')
			if get_reward('initial'+str(j)+'.onnx','initial'+str(j)+'.vnnlib','new'+str(j)+'.onnx','new'+str(j)+'.vnnlib'):
				agent.reward(True)
				agent2.reward(True)
				if get_reward('best'+str(j)+'.onnx','best'+str(j)+'.vnnlib','new'+str(j)+'.onnx','new'+str(j)+'.vnnlib'):
					DNN_new.write_onnx('best'+str(j)+'.onnx')
					VNNLIB_new.write_vnnlib('best'+str(j)+'.vnnlib')
			else:
				agent.reward(False)
				agent2.reward(False)
		max_index = hisotry.index(max(hisotry))
		max2_index = hisotry2.index(max(hisotry2))
		initial[max_index] += max2_index*rng(2) + (1-max2_index)*(-1*rng(2))
		DNN_final = mk_onnx(in_size=initial[0],out_size=initial[1],depth=initial[2],width_layer=initial[3],range=initial[4])
		VNNLIB_final = mk_vnnlib(True,DNN_final)
		DNN_final.write_onnx(filename='final'+str(j)+'.onnx')
		VNNLIB_final.write_vnnlib(filename='final'+str(j)+'.vnnlib')
		mds_besttime = [run_mapleDNNsat('best'+str(j)+'.vnnlib','best'+str(j)+'.onnx')]
		mds_initialtime = [run_mapleDNNsat('initial'+str(j)+'.vnnlib','initial'+str(j)+'.onnx')]
		mds_finaltime= [run_mapleDNNsat('final'+str(j)+'.vnnlib','final'+str(j)+'.onnx')]
		dnnf_besttime = [run_dnnf('best'+str(j)+'.vnnlib','best'+str(j)+'.onnx')]
		dnnf_initialtime = [run_dnnf('initial'+str(j)+'.vnnlib','initial'+str(j)+'.onnx')]
		dnnf_finaltime= [run_dnnf('final'+str(j)+'.vnnlib','final'+str(j)+'.onnx')]
		with open('mds_best.csv','a',newline='') as f:
			w = csv.writer(f,dialect="excel")
			w.writerow(mds_besttime)
		with open('mds_initial.csv','a',newline='') as ff:
			w = csv.writer(ff,dialect="excel")
			w.writerow(mds_initialtime)
		with open('mds_final.csv','a',newline='') as fff:
			w = csv.writer(fff,dialect="excel")
			w.writerow(mds_finaltime)
		with open('dnnf_best.csv','a',newline='') as p:
			w = csv.writer(p,dialect="excel")
			w.writerow(dnnf_besttime)
		with open('dnnf_initial.csv','a',newline='') as pp:
			w = csv.writer(pp,dialect="excel")
			w.writerow(dnnf_initialtime)
		with open('dnnf_final.csv','a',newline='') as ppp:
			w = csv.writer(ppp,dialect="excel")
			w.writerow(dnnf_finaltime)