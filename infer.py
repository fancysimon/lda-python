#!/usr/bin/env python
#
# LDA inference

from optparse import OptionParser
import random

from document import *
from model import *
from sampler import *

def parse_args():
	parser = OptionParser()

	parser.add_option("-a", "--alpha", dest="alpha", type="float", default=0.1, help="alpha")
	parser.add_option("-b", "--beta", dest="beta", type="float", default=0.01, help="beta")
	parser.add_option("--inference_name", dest="inference_name", type="string", default="test.txt",
						help="file name of inference data")
	parser.add_option("--result_name", dest="result_name", type="string", default="result.txt",
						help="file name of inference result")
	parser.add_option("--model_name", dest="model_name", type="string", default="model.txt",
						help="file prefix of model")
	parser.add_option("--total_iterations", dest="total_iterations", type="int", 
						default=10, help="total iterations")
	parser.add_option("--burn_in_iterations", dest="burn_in_iterations", type="int", 
						default=5, help="burn in iterations")

	(options, args) = parser.parse_args()
	return options

def load_corpus(inference_name, num_topics, word_id_map):
	inference_file = open(inference_name, "r")
	corpus = []
	for line in inference_file:
		if len(line) == 0 or line[0] == "\n" or line[0] == "\r" or line[0] == "#":
			continue
		document = Document()
		document.load_document_for_inference(line, word_id_map, num_topics)
		corpus.append(document)
	inference_file.close()
	return corpus

def main():
	options = parse_args()
	random.seed()
	model = Model()
	num_topics, word_id_map = model.load_model(options.model_name)
	corpus = load_corpus(options.inference_name, num_topics, word_id_map)
	model.init_document_model_given_corpus(corpus)
	sampler = Sampler(options.alpha, options.beta, False)
	sampler.init_model_given_corpus(corpus, model)
	for i in range(options.total_iterations):
		print "Iteration:", i
		sampler.sample_loop(corpus, model)
		if i >= options.burn_in_iterations:
			model.accumulate_model_for_inference()
	model.average_accumulative_model_for_inference()
	model.save_inference_result(options.result_name)

if __name__ == "__main__":
	main()