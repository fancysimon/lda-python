#!/usr/bin/env python
#
# LDA Model

class Model(object):
	"""LDA Model"""
	def __init__(self, num_documents, num_topics, num_words):
		self.__document_topic_count = [[0]*num_topics for i in range(num_documents)]
		self.__word_topic_count = [[0]*num_topics for i in range(num_words)]
		self.__golobal_topic_count = [0]*num_topics

	def increment(self, document, topic, word, count):
		self.__document_topic_count[document][topic] += count
		self.__word_topic_count[word][topic] += count
		self.__golobal_topic_count[topic] += count

	def decrement(self, document, topic, word, count):
		assert(self.__document_topic_count[document][topic] >= count)
		assert(self.__word_topic_count[word][topic] >= count)
		assert(self.__golobal_topic_count[topic] >= count)
		self.increment(document, topic, word, -count)

	def document_topic_count(self):
		return self.__document_topic_count

	def word_topic_count(self, word):
		return self.__word_topic_count[word]

	def golobal_topic_count(self):
		return self.__golobal_topic_count

	def num_topics(self):
		return len(self.golobal_topic_count())

	def num_words(self):
		return len(self.__word_topic_count)

	def compute_model_param(self):
		pass

	def save_model(self, model_prefix):
		theta_file = model_prefix + ".theta"
		phi_file = model_prefix + ".phi"

