from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import config
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS_token = 0 
EOS_token = 1

class Lang:
	def __init__(self, name):
		self.name = name 
		self.word2index = {}
		self.word2count = {}
		self.index2word = {0: "SOS", 1: "EOS"}
		self.n_words = 2 # count SOS and EOS

	def addSentence(self, sentence):
		for word in sentence.split(' '):
			self.addWord(word)

	def addWord(self, word):
		if word not in self.word2index:
			self.word2index[word] = self.n_words
			self.word2count[word] = 1
			self.index2word[self.n_words] = word
			self.n_words += 1
		else:
			self.word2count[word] += 1

# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
	return "".join(
		c for c in unicodedata.normalize("NFD", s)
		if unicodedata.category(c) != 'Mn'
	)

# Lowercase, trim, and remove non-letter characters

def normalizeString(s):
	s = unicodeToAscii(s.lower().strip())
	s = re.sub(r"([.!?])", r" \1", s)
	s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
	return s

# To read the data file we will split the file into lines, and then split lines into pairs. The files are all English → Other Language, so if we want to translate from Other Language → English I added the reverse flag to reverse the pairs.

def readLangs(lang1, lang2, reverse=False):
	print("Reading Lines...")
	# Read the file and split into lines
	lines = open('%s-%s.txt' % (lang1, lang2), encoding='utf-8').read().strip().split('\n')
	# split everyline into pairs and normalize
	
	pairs = [
		[normalizeString(s) for s in l.split('\t')] for l in lines
	]
	# Reversed pairs, make lang instances
	if reverse:
		pairs = [list(reversed(p)) for p in pairs]
		input_lang = Lang(lang2)
		output_lang = Lang(lang1)
	else:
		input_lang = Lang(lang1)
		output_lang = Lang(lang2)
	return input_lang, output_lang, pairs



eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)

def filterPair(p):
    return len(p[0].split(' ')) < config.MAX_LENGTH and \
        len(p[1].split(' ')) < config.MAX_LENGTH and \
        p[1].startswith(eng_prefixes)


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]