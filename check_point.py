#!/usr/bin/env python
# Author: Chao Xiong <fancysimon@gmail.com>
# Check point

import os
import pickle

checkpoint_name = '.checkpoint'
checkpoint_tmp_name = checkpoint_name + '.tmp'

class CheckPointer(object):
	'''Check pointer'''
	class CheckPointData(object):
		'''Check point data'''
		def __init__(self):
			self.model = None

	def __init__(self):
		self.__data = CheckPointData()

	def dump(self, model):
		# TODO: Add all data
		self.__data.model = model

		checkpoint_file = open(checkpoint_tmp_name, 'w')
		pickle.dump(self.__data, checkpoint_file)
		checkpoint_file.close()
		# rename is atomic operation in linux.
		os.rename(checkpoint_tmp_name, checkpoint_name)

	def load(delf):
		checkpoint_file = open(checkpoint_name, 'r')
		self.__data = pickle.load(checkpoint_file)
		checkpoint_file.close()

		return self.__data.model
