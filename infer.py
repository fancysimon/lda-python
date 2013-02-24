#!/usr/bin/env python
#
# LDA inference

from optparse import OptionParser

from document import *
from model import *
from sampler import *
import lda

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

def main():
	options = parse_args()
	model = Model()
	num_topics, word_id_map = model.load_model(options.model_name)
	

if __name__ == "__main__":
	main()