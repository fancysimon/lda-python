#!/usr/bin/env python
#
# Document for LDA

import random

class Document(object):
	""""LDA Document"""
	class Iterator(object):
		"""LDA Document Iterator"""
		def __init__(self, document):
			self.__document = document
			self.__start_index = 0

		def done(self):
			return self.__start_index >= len(self.__document.words())

		def next(self):
			self.__start_index += 1

		def word(self):
			return self.__document.words()[self.__start_index]

		def topics(self):
			return self.__document.word_topics()[self.__start_index]

		def set_topic(self, topic_index, new_topic):
			self.__document.set_topic(self.__start_index, topic_index, new_topic)

	def __init__(self):
		self.__words = []
		self.__word_ids = []
		self.__word_topics = []

	def load_document(self, data, word_id_map, num_topics):
		"""document format
		word count word2 count word3 count and so on.
		word,word2,word3 is not same.
		"""
		records = data.strip().split()
		assert(len(records) % 2 == 0)
		for index in range(len(records)):
			if index % 2 == 0:
				word = records[index]
			else:
				count = int(records[index])
				topics = []
				for i in range(count):
					# sample random topic
					topics.append(random.randint(0, num_topics - 1))
				if not word_id_map.has_key(word):
					# word id start from 0
					word_id = len(word_id_map)
					word_id_map[word] = word_id
				else:
					word_id = word_id_map[word]
				self.__words.append(word)
				self.__word_ids.append(word_id)
				self.__word_topics.append(topics)

	def load_document_for_inference(self, data, word_id_map, num_topics):
		"""document format
		word count word2 count word3 count and so on.
		word,word2,word3 is not same.
		Do not save |word| if |word_id_map| does not contain |word|.
		"""
		records = data.strip().split()
		assert(len(records) % 2 == 0)
		for index in range(len(records)):
			if index % 2 == 0:
				word = records[index]
			else:
				if word_id_map.has_key(word):
					count = int(records[index])
					topics = []
					for i in range(count):
						# sample random topic
						topics.append(random.randint(0, num_topics - 1))
					word_id = word_id_map[word]
					self.__words.append(word)
					self.__word_ids.append(word_id)
					self.__word_topics.append(topics)

	def words(self):
		return self.__word_ids

	def word_topics(self):
		return self.__word_topics

	def debug_string(self):
		debug_string = ""
		for i in range(len(self.__word_ids)):
			debug_string += self.__words[i] + "(" + str(self.__word_ids[i]) + "):" + \
					str(self.__word_topics[i]) + "\n"
		return debug_string

	def set_topic(self, word_index, topic_index, new_topic):
		self.__word_topics[word_index][topic_index] = new_topic
