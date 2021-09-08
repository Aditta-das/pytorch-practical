import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
data = [("me gusta comer en la cafeteria".split(), "SPANISH"),
        ("Give it to me".split(), "ENGLISH"),
        ("No creo que sea una buena idea".split(), "SPANISH"),
        ("No it is not a good idea to get lost at sea".split(), "ENGLISH")]

test_data = [("Yo creo que si".split(), "SPANISH"),
             ("it is lost on me".split(), "ENGLISH")]

# word_to_ix maps each word in the vocab to a unique integer, which will be its
# index into the Bag of words vector
word_to_idx = {}
for sent, _ in data + test_data:
    for word in sent:
        if word not in word_to_idx:
            word_to_idx[word] = len(word_to_idx)

VOCAB_SIZE = len(word_to_idx)
NUM_LABELS = 2

class BoWClassifier(nn.Module):
    """docstring for BoWClassifier"""
    def __init__(self, num_labels, vocab_size):
        super(BoWClassifier, self).__init__()
        self.linear = nn.Linear(vocab_size, num_labels)

    def forward(self, bow_vec):
        return F.log_softmax(self.linear(bow_vec), dim=1)

def make_bow_vector(sentence, word_to_idx):
    vec = torch.zeros(len(word_to_idx))
    for word in sentence:
        vec[word_to_idx[word]] += 1
    return vec.view(1, -1)

def make_target(label, label_to_idx):
    return torch.LongTensor([label_to_idx[label]])

model = BoWClassifier(num_labels=NUM_LABELS, vocab_size=VOCAB_SIZE)

for param in model.parameters():
    print(param)

label_to_idx = {"SPANISH": 0, "ENGLISH": 1}

# To run the model, pass in a BoW vector
# Here we don't need to train, so the code is wrapped in torch.no_grad()
with torch.no_grad():
    sample = data[0]
    bow_vector = make_bow_vector(sample[0], word_to_idx)
    log_probs = model(bow_vector)
    print(log_probs)

with torch.no_grad():
    for instance, label in test_data:
        bow_vec = make_bow_vector(instance, word_to_idx)
        log_probs = model(bow_vec)
        print(log_probs)
# Print the matrix column corresponding to "creo"
print(next(model.parameters())[:, word_to_idx["creo"]])

loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

for epoch in range(100):
    for instance, label in data:
        # Step 1. Remember that PyTorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()
        # Step 2. Make our BOW vector and also we must wrap the target in a
        # Tensor as an integer. For example, if the target is SPANISH, then
        # we wrap the integer 0. The loss function then knows that the 0th
        # element of the log probabilities is the log probability
        # corresponding to SPANISH
        bow_vec = make_bow_vector(instance, word_to_idx)
        target = make_target(label, label_to_idx)
        # step 3: Run our forward pass
        log_probs = model(bow_vec)
        # Step 4. Compute the loss, gradients, and update the parameters by
        # calling optimizer.step()
        loss = loss_function(log_probs, target)

        loss.backward()
        optimizer.step()

with torch.no_grad():
    for instance, label in test_data:
        bow_vec = make_bow_vector(instance, word_to_idx)
        log_probs = model(bow_vec)
        print(log_probs)

# Index corresponding to Spanish goes up, English goes down!
print(next(model.parameters())[:, word_to_idx["good"]])
