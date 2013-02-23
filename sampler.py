#!/usr/bin/env python
#
# Gibbs Sampling for LDA

import random
from document import *

class Sampler(object):
	"""Sampler"""
	def __init__(self, alpha, beta):
		self.__alpha = alpha
		self.__beta = beta

	def init_model_given_corpus(self, corpus, model):
		# init topic, term distribution
		for m in range(len(corpus)):
			document = corpus[m]
			iter = Document.Iterator(document)
			while not iter.done():
				for topic in iter.topics():
					model.increment(m, topic, iter.word(), 1)
				iter.next()

	def sample_loop(self, corpus, model):
		for m in range(len(corpus)):
			document = corpus[m]
			iter = Document.Iterator(document)
			while not iter.done():
				for k in range(len(iter.topics())):
					topic = iter.topics()[k]
					model.decrement(m, topic, iter.word(), 1)
					topic_distributions = self.compute_topic_distributions(
							model, m, topic, iter.word())
					new_topic = self.sample_new_topic(topic_distributions)
					model.increment(m, new_topic, iter.word(), 1)
					iter.set_topic(k, new_topic)
				iter.next()

	def compute_topic_distributions(self, model, document_id, topic, word):
		num_topics = model.num_topics()
		num_words = model.num_words()
		topic_distributions = [0] * num_topics
		word_topic_count = model.word_topic_count(word)
		golobal_topic_count = model.golobal_topic_count()
		document_topic_count = model.document_topic_count()
		for topic in range(num_topics):
			topic_distributions[topic] = \
					(word_topic_count[topic] + self.__beta) * \
					(document_topic_count[document_id][topic] + self.__alpha) / \
					(golobal_topic_count[topic] + num_words * self.__beta)
		return topic_distributions

	def sample_new_topic(self, topic_distributions):
		distribution_sum = 0.0
		for distribution in topic_distributions:
			distribution_sum += distribution

		sample_distribution = random.random() * distribution_sum
		sum_so_far = 0.0
		for i in range(len(topic_distributions)):
			distribution = topic_distributions[i]
			sum_so_far += distribution
			if sum_so_far >= sample_distribution:
				return i
		return -1