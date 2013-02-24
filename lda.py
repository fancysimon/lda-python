#!/usr/bin/env python
#
# LDA

from optparse import OptionParser

from document import *
from model import *
from sampler import *

def parse_args():
	parser = OptionParser()

	parser.add_option("-a", "--alpha", dest="alpha", type="float", help="alpha")
	parser.add_option("-b", "--beta", dest="beta", type="float", default=0.01, help="beta")
	parser.add_option("-k", "--num_topics", dest="num_topics", type="int", help="topic numbers")
	parser.add_option("--train_name", dest="train_name", type="string", default="train.txt",
						help="file name of taining data")
	parser.add_option("--model_name", dest="model_name", type="string", default="model.txt",
						help="file prefix of model")
	parser.add_option("--total_iterations", dest="total_iterations", type="int", 
						default=10, help="total iterations")
	parser.add_option("--burn_in_iterations", dest="burn_in_iterations", type="int", 
						default=5, help="burn in iterations")
	parser.add_option("--compute_loglikehood", dest="compute_loglikehood", action="store_true",
						help="compute loglikehood")

	(options, args) = parser.parse_args()
	if not options.num_topics:
		print "num_topics must be specified.\n"
		parser.print_help()
		exit(1)
	if not options.alpha:
		options.alpha = 50.0 / options.num_topics

	return options

def load_corpus(train_name, num_topics):
	train_file = open(train_name, "r")
	corpus = []
	word_id_map = {}
	for line in train_file:
		if len(line) == 0 or line[0] == "\n" or line[0] == "\r" or line[0] == "#":
			continue
		document = Document()
		document.load_document(line, word_id_map, num_topics)
		corpus.append(document)
	train_file.close()
	return corpus, word_id_map

def main():
	options = parse_args()
	corpus, word_id_map = load_corpus(options.train_name, options.num_topics)
	# for d in corpus:
	# 	print d.debug_string()

	sampler = Sampler(options.alpha, options.beta)
	model = Model()
	model.init_model(len(corpus), options.num_topics, len(word_id_map))
	sampler.init_model_given_corpus(corpus, model)
	for i in range(options.total_iterations):
		print "Iteration:", i
		sampler.sample_loop(corpus, model)
		if options.compute_loglikehood:
			print "    Loglikehood:", sampler.compute_log_likehood(corpus, model)
		if i > options.burn_in_iterations:
			model.accumulate_model()
	model.average_accumulative_model()
	model.save_model(options.model_name, word_id_map)

if __name__ == "__main__":
	main()
