from data_preprocessing import *
from data_prepare import *
from models import *
# Training
# Preparing Training Data
# To train, for each pair we will need an input tensor (indexes of the words in the input sentence) and target tensor (indexes of the words in the target sentence). While creating these vectors we will append the EOS token to both sequences.



def indexesFromSentence(lang, sentence):
	return [lang.word2index[word] for word in sentence.split(' ')]

def tensorFromSentence(lang, sentence):
	indexes = indexesFromSentence(lang, sentence)
	indexes.append(EOS_token)
	return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def tensorFromPair(pair):
	input_tensor = tensorFromSentence(input_lang, pair[0])
	target_tensor = tensorFromSentence(output_lang, pair[1])
	return (input_tensor, target_tensor)


teacher_forcing_ratio = 0.5

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=config.MAX_LENGTH):
	encoder_hidden = encoder.initHidden()
	encoder_optimizer.zero_grad()
	decoder_optimizer.zero_grad()

	input_length = input_tensor.size(0)
	target_length = target_tensor.size(0)

	encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

	loss = 0
	for ei in range(input_length):
		encoder_output, encoder_hidden = encoder(
			input_tensor[ei], encoder_hidden
		)
		encoder_outputs[ei] = encoder_output[0, 0]

	decoder_input = torch.tensor([[SOS_token]], device=device)
	decoder_hidden = encoder_hidden

	use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

	if use_teacher_forcing:
		# Teacher forcing: Feed the target as the next input
		for di in range(target_length):
			decoder_output, decoder_hidden = decoder(
				decoder_input, decoder_hidden
			)
			loss += criterion(decoder_output, target_tensor[di])
			decoder_input = target_tensor[di]
	else:
		for di in range(target_length):
			decoder_output, decoder_hidden = decoder(
				decoder_input, decoder_hidden
			)
			topv, topi = decoder_output.topk(1)
			decoder_input = topi.squeeze().detach()
			loss += criterion(decoder_output, target_tensor[di])
			if decoder_input.item() == EOS_token:
				break
	loss.backward()
	encoder_optimizer.step()
	decoder_optimizer.step()
	return loss.item() / target_length
