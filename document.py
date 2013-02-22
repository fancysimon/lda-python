#!/usr/bin/env python
#
# Document for LDA

class Document(object):
	""""LDA Document"""
	def __init__(self):
		self.__words = []
		self.__topics = []

	def load_data(self, data, word_id_map):
		records = data.split()
		assert(len(records) % 2 == 0)
		for index in range(len(records)):
			if index % 2 == 0:
				word = records[index]
			else:
				count = records[index]
				if not word_id_map.has_key(word):
					word_id = len(word_id_map)
					word_id_map[word] = word_id
				self.__words.append(word_id_map[word])