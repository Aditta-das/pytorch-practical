from data_preprocessing import *
import random
# The full process for preparing the data is:

	# Read text file and split into lines, split lines into pairs
	# Normalize text, filter by length and content
	# Make word lists from sentences in pairs

def prepareData(lang1, lang2, reverse=False):
	input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
	print(f"Read sentence pairs: {len(pairs)}")
	pairs = filterPairs(pairs)
	print(f"Trimmed to sentence pairs: {len(pairs)}")
	print(f"Counting Words...")
	for pair in pairs:
		input_lang.addSentence(pair[0])
		output_lang.addSentence(pair[1])
	print("Counted words: ")
	print(input_lang.name, input_lang.n_words)
	print(output_lang.name, output_lang.n_words)
	return input_lang, output_lang, pairs  

input_lang, output_lang, pairs = prepareData('eng', 'fra', True)
print(random.choice(pairs))

