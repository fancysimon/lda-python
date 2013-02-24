#!/usr/bin/env python
#
# LDA Model

class Model(object):
	"""LDA Model"""
	def __init__(self):
		pass

	def init_model(self, num_documents, num_topics, num_words):
		self.__document_topic_count = [[0]*num_topics for i in range(num_documents)]
		self.__word_topic_count = [[0]*num_topics for i in range(num_words)]
		self.__golobal_topic_count = [0]*num_topics

		self.__num_accumulations = 0
		self.__accumulative_word_topic_count = [[0]*num_topics for i in range(num_words)]
		self.__accumulative_golobal_topic_count = [0]*num_topics

	def load_model(self, model_filename):
		word_id_map = {}
		model_file = open(model_filename, "r")
		self.__word_topic_count = []
		for line in model_file:
			if len(line) == 0 or line[0] == "\n" or line[0] == "\r" or line[0] == "#":
				continue
			s = line.strip().split("\t")
			if len(s) < 2:
				print "model file has some error, error line is", line
			word = s[0]
			topic_counts = [int(x) for x in s[1].split(" ")]
			word_id = len(word_id_map)
			word_id_map[word] = word_id
			self.__word_topic_count.append(topic_counts)

		num_topics = len(self.__word_topic_count[0])
		self.__golobal_topic_count = [0] * num_topics
		for topic_counts in self.__word_topic_count:
			for topic in range(len(topic_counts)):
				self.__golobal_topic_count[topic] += topic_counts[topic]

		model_file.close()
		return num_topics, word_id_map

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

	def word_topic_count(self):
		return self.__word_topic_count

	def golobal_topic_count(self):
		return self.__golobal_topic_count

	def num_topics(self):
		return len(self.golobal_topic_count())

	def num_words(self):
		return len(self.__word_topic_count)

	def save_model(self, model_filename, word_id_map):
		model_file = open(model_filename, "w")
		id_word_map = [0] * len(word_id_map)
		for word, id in word_id_map.items():
			id_word_map[id] = word
		# write word and it's topic distribution
		for word_id in range(self.num_words()):
			model_file.write(id_word_map[word_id] + "\t")
			for count in self.__accumulative_word_topic_count[word_id]:
				model_file.write(str(count) + " ")
			model_file.write("\n")
		model_file.close()

	def accumulate_model(self):
		for word in range(self.num_words()):
			for topic in range(self.num_topics()):
				self.__accumulative_word_topic_count[word][topic] += \
						self.__word_topic_count[word][topic]
		for topic in range(self.num_topics()):
			self.__accumulative_golobal_topic_count[topic] += \
					self.__golobal_topic_count[topic]
		self.__num_accumulations += 1

	def average_accumulative_model(self):
		for word in range(self.num_words()):
			for topic in range(self.num_topics()):
				self.__accumulative_word_topic_count[word][topic] /= \
						1.0 * self.__num_accumulations
		for topic in range(self.num_topics()):
			self.__accumulative_golobal_topic_count[topic] /= \
					1.0 * self.__num_accumulations

