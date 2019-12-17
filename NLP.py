import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.nn.utils import clip_grad_norm_
from utils import Dictionary,Corpus

# device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# hyper parameter
embed_size = 128
hidden_size = 1024
num_layers = 2
num_epoches = 10
# number of words to be sampled
num_samples = 1000
batch_size = 20
seq_length = 30
learning_rate = 0.002

# loading data
corpus = Corpus()
ids = corpus.get_data("./data/train.txt", batch_size)  # get sequence of sentence
vocab_size = len(corpus.dictionary) # len(vocab)
num_batches = ids.size(1)


class RNNLM(nn.Module):
    def __init__(self,vocab_size,embed_size,hidden_size,num_layers):
        super(RNNLM,self).__init__()
        # intial the matrix
        self.embed_size = nn.Embedding(vocab_size,embed_size)
        self.lstm = nn.LSTM(embed_size,hidden_size,num_layers,batch_first=True)
        self.linear = nn.Linear(hidden_size,vocab_size)

    def forward(self, x, h):
        x = self.embed_size(x) # serializatio
        out,(h,c) = self.lstm(x,h)
        out = out.reshape(out.size(0)*out.size(1),out.size(2))
        out = self.linear(out)
        return out,(h,c)

model = RNNLM(vocab_size,embed_size,hidden_size,num_layers).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr = learning_rate)

def detach(states):
    return [state.detach() for state in states]

for epoch in range(num_epoches):
    states=(torch.zeros(num_layers,batch_size,hidden_size).to(device),
            torch.zeros(num_layers,batch_size,hidden_size).to(device))

    for i in range(0,ids.size(1)-seq_length,seq_length):
        inputs = ids[:,i:i+seq_length].to(device)
        targets = ids[:,(i+1):(i+1)+seq_length].to(device)

        states = detach(states)
        outputs,states = model(inputs,states)
        loss = criterion(outputs,targets.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(),0.5)
        optimizer.step()

        step = (i+1)//seq_length
        if step % 100 == 0:
            print("Epoch {}/{} , Step {}/{}, Loss {:.4f}".format(
                epoch+1,num_epoches,step,num_batches,loss.item()
            ))

# Test the model
with torch.no_grad():
    with open('sample.txt', 'w') as f:
        # Set intial hidden ane cell states
        state = (torch.zeros(num_layers, 1, hidden_size).to(device),
                 torch.zeros(num_layers, 1, hidden_size).to(device))

        # Select one word id randomly
        prob = torch.ones(vocab_size)
        input = torch.multinomial(prob, num_samples=1).unsqueeze(1).to(device)

        for i in range(num_samples):
            # Forward propagate RNN
            output, state = model(input, state)

            # Sample a word id
            prob = output.exp()
            word_id = torch.multinomial(prob, num_samples=1).item()

            # Fill input with sampled word id for the next time step
            input.fill_(word_id)

            # File write
            word = corpus.dictionary.idx2word[word_id]
            word = '\n' if word == '<eos>' else word + ' '
            f.write(word)

            if (i+1) % 100 == 0:
                print('Sampled [{}/{}] words and save to {}'.format(i+1, num_samples, 'sample.txt'))






