#!/usr/bin/env python
#
# MPI LDA

from optparse import OptionParser
import random

from mpi4py import MPI

from document import *
from model import *
from sampler import *

class ParallelModel(Model):
	"""Parllel model"""
	def __init__(self):
		Model.__init__()

	def AllReduceModel(self, comm):
		num_topics = self.num_topics()
		num_words = self.num_words()
		temp_word_topic_count = [[0]*num_topics for i in range(num_words)]
		temp_golobal_topic_count = [0]*num_topics
		comm.Allreduce(self.word_topic_count(), temp_word_topic_count, MPI.SUM)
		comm.Allreduce(self.golobal_topic_count(), temp_golobal_topic_count,
						MPI.SUM)
		self.set_word_topic_count(temp_word_topic_count)
		self.set_golobal_topic_count(temp_golobal_topic_count)

def parse_args():
	parser = OptionParser()

	parser.add_option("-a", "--alpha", dest="alpha", type="float", help="alpha")
	parser.add_option("-b", "--beta", dest="beta", type="float",
						default=0.01, help="beta")
	parser.add_option("-k", "--num_topics", dest="num_topics", type="int",
						help="topic numbers")
	parser.add_option("--train_name", dest="train_name", type="string",
						default="train.txt", help="file name of taining data")
	parser.add_option("--model_name", dest="model_name", type="string",
						default="model.txt", help="file prefix of model")
	parser.add_option("--total_iterations", dest="total_iterations", type="int", 
						default=10, help="total iterations")
	parser.add_option("--burn_in_iterations", dest="burn_in_iterations",
						type="int", default=5, help="burn in iterations")
	parser.add_option("--compute_likelihood", dest="compute_likelihood",
						action="store_true", help="compute log likelihood")

	(options, args) = parser.parse_args()
	if not options.num_topics:
		print "num_topics must be specified.\n"
		parser.print_help()
		exit(1)
	if not options.alpha:
		options.alpha = 50.0 / options.num_topics

	return options

def distributely_load_corpus(train_name, num_topics, myid, pnum):
	train_file = open(train_name, "r")
	corpus = []
	word_id_map = {}
	word_set = set()
	index = 0
	for line in train_file:
		if len(line) == 0 or line[0] == "\n" or \
				line[0] == "\r" or line[0] == "#":
			continue
		if index % pnum == myid:
			document = Document()
			document.load_document_for_distribute(line, num_topics, word_set)
			corpus.append(document)
		else:
			document.load_document_for_distribute(
					line, num_topics, word_set, True)
		index += 1
	train_file.close()

	for word in word_set:
		word_id = len(word_id_map)
		word_id_map[word] = word_id
	# Update real word id.
	for document in corpus:
		document.reset_word_ids(word_id_map)
	return corpus, word_id_map

def main():
	options = parse_args()
	random.seed()

	comm = MPI.COMM_WORLD
	pnum = comm.Get_size()
	myid = comm.Get_rank()
	corpus_local, word_id_map = distributely_load_corpus(
			options.train_name, options.num_topics, myid, pnum)
	# for d in corpus_local:
	# 	print d.debug_string()

	sampler = Sampler(options.alpha, options.beta)
	for i in range(options.total_iterations):
		print "Iteration:", i
		model = ParallelModel()
		model.init_model(len(corpus_local), options.num_topics, len(word_id_map))
		sampler.init_model_given_corpus(corpus_local, model)
		model.AllReduceModel(comm)
		sampler.sample_loop(corpus_local, model)
		if options.compute_likelihood:
			loglikelihood_local =
					sampler.compute_log_likelihood(corpus_local, model)
			loglikelihood_gobal = 0.0
			comm.Reduce(loglikelihood_local, loglikelihood_gobal, MPI.SUM, 0)
			if myid == 0:
				print "    Loglikehood:", loglikelihood_gobal
		if i >= options.burn_in_iterations:
			model.accumulate_model()
	model = ParallelModel()
	model.init_model(len(corpus_local), options.num_topics, len(word_id_map))
	sampler.init_model_given_corpus(corpus_local, model)
	model.AllReduceModel(comm)
	if myid == 0:
		model.average_accumulative_model()
		model.save_model(options.model_name, word_id_map)

if __name__ == "__main__":
	main()
