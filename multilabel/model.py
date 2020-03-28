import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F

class DecoderRNN(nn.Module):
    def __init__(self, attr_size, embed_size, hidden_size, vocab_size, num_layers, max_seq_length=30):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(attr_size, embed_size)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.max_seq_length = max_seq_length
        
    def forward(self, features, captions, lengths):
        """Decode image feature vectors and generates captions."""
        embeddings = self.embed(captions)
        embed_attr = self.fc(features).unsqueeze(1)
        embeddings = torch.cat((embed_attr, embeddings), 1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True) 
        hiddens, _ = self.lstm(packed)
        outputs = self.linear(hiddens[0])
        return outputs
    
    def sample(self, features, states=None):
        """Generate captions for given image features using greedy search."""
        sampled_ids = []
        inputs = self.fc(features).unsqueeze(1)
        for i in range(self.max_seq_length):
            hiddens, states = self.lstm(inputs, states)          # hiddens: (batch_size, 1, hidden_size)
            outputs = self.linear(hiddens.squeeze(1))            # outputs:  (batch_size, vocab_size)
            _, predicted = outputs.max(1)                        # predicted: (batch_size)
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)                       # inputs: (batch_size, embed_size)
            inputs = inputs.unsqueeze(1)                         # inputs: (batch_size, 1, embed_size)
        sampled_ids = torch.stack(sampled_ids, 1)                # sampled_ids: (batch_size, max_seq_length)
        return sampled_ids


class Merger(nn.Module):
    def __init__(self, decoder, hidden_size, attr_size, embed_size, vocab_size, max_seq_length=30):
        super(Merger, self).__init__()
        self.decoder = decoder
        self.embed_size = embed_size
        self.max_seq_length = max_seq_length
        self.fc = nn.Linear(attr_size, hidden_size)
        self.fc1 = nn.Linear(256, embed_size)
        self.fc2 = nn.Linear(attr_size, embed_size)
        self.fc3 = nn.Linear(embed_size, 1)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.linear = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, features, captions, lengths, states=None):
        embeddings= self.decoder.embed(captions)
        inputs = self.decoder.fc(features).unsqueeze(1)
        embed_attr = self.fc2(features)
        attr2hiddens = self.fc(features).unsqueeze(1)
        max_seq_length = lengths[0]
        beta = 1
        result = []
        features = features.unsqueeze(1)
        m = nn.Sigmoid()
        for i in range(max_seq_length):
            hiddens, states = self.decoder.lstm(inputs, states)
            new_hiddens = beta * self.fc4(hiddens) + (1 - beta) * attr2hiddens
            outputs = self.linear(F.relu(new_hiddens)).squeeze(1)
            result.append(outputs)
            _, predicted = outputs.max(1)
            #inputs = self.decoder.embed(predicted).unsqueeze(1)#训练阶段的输入应该是正确单词的embedding
            inputs = self.decoder.embed(captions[:, i]).unsqueeze(1)
            beta = F.tanh(self.fc1(inputs))
            beta = m(self.fc3(beta))
            #print(beta.size(), torch.mean(beta))
        result = torch.stack(result, 1)
        return result



