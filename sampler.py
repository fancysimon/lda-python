#!/usr/bin/env python
#
# Gibbs Sampling for LDA

import math
import random
from document import *

class Sampler(object):
	"""Sampler"""
	def __init__(self, alpha, beta, update_model=True):
		self.__alpha = alpha
		self.__beta = beta
		self.__update_model = update_model

	def init_model_given_corpus(self, corpus, model):
		# init topic, term distribution
		for m in range(len(corpus)):
			document = corpus[m]
			iter = Document.Iterator(document)
			while not iter.done():
				for topic in iter.topics():
					self.increment_model(model, m, topic, iter.word(), 1)
				iter.next()

	def sample_loop(self, corpus, model):
		for m in range(len(corpus)):
			document = corpus[m]
			iter = Document.Iterator(document)
			while not iter.done():
				for k in range(len(iter.topics())):
					topic = iter.topics()[k]
					self.decrement_model(model, m, topic, iter.word(), 1)
					topic_distributions = self.compute_topic_distributions(
							model, m, topic, iter.word())
					new_topic = self.sample_new_topic(topic_distributions)
					self.increment_model(model, m, new_topic, iter.word(), 1)
					iter.set_topic(k, new_topic)
				iter.next()

	def compute_topic_distributions(self, model, document_id, topic, word):
		num_topics = model.num_topics()
		num_words = model.num_words()
		topic_distributions = [0] * num_topics
		word_topic_count = model.word_topic_count()[word]
		golobal_topic_count = model.golobal_topic_count()
		document_topic_count = model.document_topic_count()
		for topic in range(num_topics):
			topic_distributions[topic] = \
					(word_topic_count[topic] + self.__beta) * 1.0 * \
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

	def compute_log_likelihood(self, corpus, model):
		"""Compute log P(corpus) = sum_d log P(d),
		where log P(d) = sum_w log P(w), where P(w) = sum_z P(w|z)P(z|d).
		"""
		log_likehood_corpus = 0.0
		for document_id in range(len(corpus)):
			document = corpus[document_id]
			iter = Document.Iterator(document)
			# Compute P(z|d) for the document and all topics.
			prob_topic_given_document = []
			num_topics = model.num_topics()
			document_topic_sum = sum(model.document_topic_count()[document_id])
			for topic in range(num_topics):
				prob_topic_given_document.append(
						(model.document_topic_count()[document_id][topic] + \
						self.__alpha) * 1.0 / \
						(document_topic_sum + self.__alpha * num_topics))

			log_likehood = 0.0
			while not iter.done():
				# Compute P(w|z) for word and given topic.
				word = iter.word()
				prob_word_given_topic = []
				word_topic_sum = sum(model.word_topic_count()[word])
				for topic in range(num_topics):
					prob_word_given_topic.append(
							(model.word_topic_count()[word][topic] + self.__beta) * 1.0 /
							(word_topic_sum + self.__beta * num_topics))

				# Compute P(w) = sum_z P(w|z)P(z|d)
				prob_word = 0
				for topic in range(num_topics):
					prob_word += prob_word_given_topic[topic] * prob_topic_given_document[topic]
				log_likehood += math.log(prob_word)

				iter.next()

			log_likehood_corpus += log_likehood
			
		return log_likehood_corpus

	def increment_model(self, model, document, topic, word, count):
		if self.__update_model:
			model.increment_topic(topic, word, count)
		model.increment_document_topic(document, topic, count)

	def decrement_model(self, model, document, topic, word, count):
		if self.__update_model:
			model.decrement_topic(topic, word, count)
		model.decrement_document_topic(document, topic, count)
