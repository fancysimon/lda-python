#!/usr/bin/env python
# Author: Chao Xiong <fancysimon@gmail.com>
# MPI LDA

from optparse import OptionParser
import random

from mpi4py import MPI

from document import *
from model import *
from sampler import *
from check_point import CheckPointer

likelihood_name = 'likelihood.txt'

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
	parser.add_option("--checkpoint_interval", dest="checkpoint_interval",
						type="int", default=0, help="checkpoint interval")
	parser.add_option("--restart_by_checkpoint", dest="restart_by_checkpoint",
						action="store_true", help="restart by checkpoint")
	
	(options, args) = parser.parse_args()
	if not options.num_topics:
		print "num_topics must be specified.\n"
		parser.print_help()
		exit(1)
	if not options.alpha:
		options.alpha = 50.0 / options.num_topics

	return options

class ParallelModel(Model):
	"""Parllel model"""
	def __init__(self):
		Model.__init__(self)

	def allreduce_model(self, comm, op, op2):
		num_topics = self.num_topics()
		num_words = self.num_words()
		temp_word_topic_count = [[0]*num_topics for i in range(num_words)]
		temp_golobal_topic_count = [0]*num_topics
		temp_word_topic_count = comm.allreduce(
				self.word_topic_count(), temp_word_topic_count, op2)
		temp_golobal_topic_count = comm.allreduce(
				self.golobal_topic_count(),
				temp_golobal_topic_count, op)
		self.set_word_topic_count(temp_word_topic_count)
		self.set_golobal_topic_count(temp_golobal_topic_count)

def list_sum(a, b, dt):
	assert(len(a) == len(b))
	for i in range(len(a)):
		b[i] += a[i]
	return b

def list2d_sum(a, b, dt):
	assert(len(a) == len(b))
	for i in range(len(a)):
		assert(len(a[i]) == len(b[i]))
		for j in range(len(a[i])):
			b[i][j] += a[i][j]
	return b

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
		document = Document()
		if index % pnum == myid:
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

	op = MPI.Op.Create(list_sum)
	op2 = MPI.Op.Create(list2d_sum)

	sampler = None
	model = None
	corpus_local = None
	word_id_map = None
	next_iteration = 0
	likelihoods = []
	checkpointer = CheckPointer()

	if options.restart_by_checkpoint:
		(model, sampler, corpus_local, word_id_map, likelihoods,
				next_iteration) = checkpointer.load()
		if myid == 0:
			print "Restart at iteration:", next_iteration
	else:
		corpus_local, word_id_map = distributely_load_corpus(
				options.train_name, options.num_topics, myid, pnum)
		sampler = Sampler(options.alpha, options.beta)

	for i in range(next_iteration, options.total_iterations):
		if myid == 0:
			print "Iteration:", i
		model = ParallelModel()
		model.init_model(len(corpus_local), options.num_topics, len(word_id_map))
		sampler.init_model_given_corpus(corpus_local, model)
		model.allreduce_model(comm, op, op2)
		sampler.sample_loop(corpus_local, model)
		if options.compute_likelihood:
			loglikelihood_local = \
					sampler.compute_log_likelihood(corpus_local, model)
			loglikelihood_golobal = 0.0
			loglikelihood_golobal = comm.reduce(
					loglikelihood_local, loglikelihood_golobal, MPI.SUM, 0)
			if myid == 0:
				print "    Loglikehood:", loglikelihood_golobal
				likelihoods.append(loglikelihood_golobal)

		if options.checkpoint_interval > 0:
			interval = options.checkpoint_interval
			if i % interval == interval - 1:
				checkpointer.dump(model, sampler, corpus_local, word_id_map,
									likelihoods, i + 1)
				if myid == 0:
					print "    Save check point."
	model = ParallelModel()
	model.init_model(len(corpus_local), options.num_topics, len(word_id_map))
	sampler.init_model_given_corpus(corpus_local, model)
	model.allreduce_model(comm, op, op2)
		
	if myid == 0:
		model.save_model(options.model_name, word_id_map, False)
		likelihood_file = open(likelihood_name, 'w')
		likelihood_file.writelines([str(x)+'\n' for x in likelihoods])
		likelihood_file.close()

	op.Free()
	op2.Free()

if __name__ == "__main__":
	main()
