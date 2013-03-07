#!/usr/bin/env python
# Author: Chao Xiong <fancysimon@gmail.com>
# Check point

import os
import pickle

checkpoint_name = '.checkpoint'
checkpoint_tmp_name = checkpoint_name + '.tmp'

class _CheckPointData(object):
	'''Check point data'''
	def __init__(self):
		self.model = None
		self.sampler = None
		self.corpus = None
		self.word_id_map = None
		self.likelihoods = None
		self.next_iteration = -1

	def reset(self, model, sampler, corpus, word_id_map, likelihoods, next_iteration):
		self.model = model
		self.sampler = sampler
		self.corpus = corpus
		self.word_id_map = word_id_map
		self.likelihoods = likelihoods
		self.next_iteration = next_iteration

	def to_turple(self):
		return (self.model, self.sampler, self.corpus, self.word_id_map,
				self.likelihoods, self.next_iteration)

class CheckPointer(object):
	'''Check pointer'''
	def __init__(self):
		self.__data = _CheckPointData()

	def dump(self, model, sampler, corpus, word_id_map,
				likelihoods, next_iteration):
		self.__data.reset(model, sampler, corpus, word_id_map,
							likelihoods, next_iteration)

		checkpoint_file = open(checkpoint_tmp_name, 'w')
		pickle.dump(self.__data, checkpoint_file)
		checkpoint_file.close()
		# rename is atomic operation in linux.
		os.rename(checkpoint_tmp_name, checkpoint_name)

	def load(self):
		if not os.path.exists(checkpoint_name):
			print 'Check point file is not exists:', checkpoint_name
			exit(1)
		checkpoint_file = open(checkpoint_name, 'r')
		self.__data = pickle.load(checkpoint_file)
		checkpoint_file.close()

		return self.__data.to_turple()
