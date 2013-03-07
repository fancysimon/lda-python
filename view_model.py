#!/usr/bin/env python
# Author: Chao Xiong <fancysimon@gmail.com>
# view model

import sys
from model import *

def usage():
	print sys.argv[0], "model [result]"
	exit(1)

def main():
	if len(sys.argv) < 2:
		usage()
	if len(sys.argv) < 3:
		result_file = sys.stdout
	else:
		result_file = open(sys.argv[2],"w")
	model_file = open(sys.argv[1],"r")

	view_model(model_file, result_file)

	model_file.close()
	if result_file != sys.stdout:
		result_file.close()

def view_model(model_file, result_file):
	num_top_words = 100
	topic_words = {}
	for line in model_file:
		s = line.strip().split('\t')
		word = s[0]
		topic_counts = [float(x) for x in s[1].split(' ')]
		num_topics = len(topic_counts)
		for k in range(num_topics):
			count = topic_counts[k]
			if not topic_words.has_key(k):
				topic_words[k] = {}
			topic_words[k][word] = count
	for k in range(num_topics):
		x = sorted(topic_words[k].items(), key=lambda(k, v):(v, k), reverse = True)
		num_top_words = min(num_top_words, len(x))
		topic_top_words = x[:num_top_words]
		result_file.write("Topic: " + str(k) + "\n")
		result_file.write('  ' + ' '.join([str(x[0])+":"+str(x[1]) for x in topic_top_words]) + '\n')

if __name__ == "__main__":
	main()