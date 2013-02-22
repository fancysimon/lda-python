#!/usr/bin/env python
#
# LDA Model

class Model(object):
	"""LDA Model"""
	def __init__(self, num_documents, num_topics, num_words):
		# number document, topic and word from 1
		num_documents += 1
		num_topics += 1
		num_words += 1
		self.__document_topic_count = [[0]*num_topics for i in range(num_documents)]
		self.__document_topic_sum = [0]*num_documents
		self.__topic_word_count = [[0]*num_words for i in range(num_topics)]
		self.__topic_word_sum = [0]*num_topics

	# def increment_topic(self, document, topic, word, count):
	# 	self.__document_topic_count[document][topic] += count
	# 	self.__document_topic_sum[document] += count
	# 	self.__topic_word_count[topic][word] += count
	# 	self.__topic_word_sum[topic] += count

	# def decrement_topic(self, document, topic, word, count):
	# 	assert(self.__document_topic_count[document][topic] >= count)
	# 	assert(self.__document_topic_sum[document] >= count)
	# 	assert(self.__topic_word_count[topic][word] >= count)
	# 	assert(self.__topic_word_sum[topic] >= count)
	# 	increment_topic(document, topic, word, -count)

	# def document_topic_count(self, document, topic):
	# 	return self.__document_topic_count[document][topic]

	# def document_topic_sum(self, document):
	# 	return self.__document_topic_sum[document]

	# def topic_word_count(self, document, topic):
	# 	return self.__topic_word_count[topic][word]

	# def topic_word_sum(self, document):
	# 	return self.__topic_word_sum[topic]

	class ModelIterator(object):
		"""Iterator for model"""
		def __init__(self):
			pass

		def done(self):
			pass
			
