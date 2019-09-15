import torch
import torch.nn as nn
import torch.nn.functional as F

#Embedding module.
class Embed(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.W = nn.Parameter(torch.Tensor(vocab_size, embed_size))

    def forward(self, x):
        return self.W[x]

    def __repr__(self):
        return "Embedding(vocab: {}, embedding: {})".format(self.vocab_size, self.embed_size)

#My custom written LSTM module.
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, dropout = 0, winit = 0.1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.W_x = nn.Parameter(torch.Tensor(4 * hidden_size, input_size))
        self.W_h = nn.Parameter(torch.Tensor(4 * hidden_size, hidden_size))
        self.b_x = nn.Parameter(torch.Tensor(4 * hidden_size))
        self.b_h = nn.Parameter(torch.Tensor(4 * hidden_size))

    def __repr__(self):
        return "LSTM(input: {}, hidden: {})".format(self.input_size, self.hidden_size)

    def lstm_step(self, x, h, c, W_x, W_h, b_x, b_h):
        gx = torch.addmm(b_x, x, W_x.t())
        gh = torch.addmm(b_h, h, W_h.t())
        xi, xf, xo, xn = gx.chunk(4, 1)
        hi, hf, ho, hn = gh.chunk(4, 1)
        inputgate = torch.sigmoid(xi + hi)
        forgetgate = torch.sigmoid(xf + hf)
        outputgate = torch.sigmoid(xo + ho)
        newgate = torch.tanh(xn + hn)
        c = forgetgate * c + inputgate * newgate
        h = outputgate * torch.tanh(c)
        return h, c

    #Takes input tensor x with dimensions: [T, B, X].
    def forward(self, x, states):
        h, c = states
        outputs = []
        inputs = x.unbind(0)
        for x_t in inputs:
            h, c = self.lstm_step(x_t, h, c, self.W_x, self.W_h, self.b_x, self.b_h)
            outputs.append(h)
        return torch.stack(outputs), (h, c)

class Linear(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.b = nn.Parameter(torch.Tensor(hidden_size))

    def forward(self, x):
        #.view() flattens the input which has dimensionality [T,B,X] to dimenstionality [T*B, X].
        z = torch.addmm(self.b, x.view(-1, x.size(2)), self.W.t())
        return z

    def __repr__(self):
        return "FC(input: {}, output: {})".format(self.input_size, self.hidden_size)

#The model as described in the paper. There is also an option to use either my custom lstm implementation or the torch.nn implementation. 
#Note that torch.nn implementation is faster. 
class Model(nn.Module):
    def __init__(self, vocab_size, hidden_size, layer_num, dropout, winit, lstm_type = "pytorch"):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.layer_num = layer_num
        self.winit = winit
        self.lstm_type = lstm_type
        self.embed = Embed(vocab_size, hidden_size)
        self.rnns = [LSTM(hidden_size, hidden_size) if lstm_type == "custom" else nn.LSTM(hidden_size, hidden_size) for i in range(layer_num)]
        self.rnns = nn.ModuleList(self.rnns)
        self.fc = Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(p=dropout)
        self.reset_parameters()
        
    def reset_parameters(self):
        for param in self.parameters():
            nn.init.uniform_(param, -self.winit, self.winit)
            
    def state_init(self, batch_size):
        dev = next(self.parameters()).device
        states = [(torch.zeros(batch_size, layer.hidden_size, device = dev), torch.zeros(batch_size, layer.hidden_size, device = dev)) if self.lstm_type == "custom" 
                  else (torch.zeros(1, batch_size, layer.hidden_size, device = dev), torch.zeros(1, batch_size, layer.hidden_size, device = dev)) for layer in self.rnns]
        return states
    
    def detach(self, states):
        return [(h.detach(), c.detach()) for (h,c) in states]
    
    def forward(self, x, states):
        x = self.embed(x)
        x = self.dropout(x)
        for i, rnn in enumerate(self.rnns):
            x, states[i] = rnn(x, states[i])
            x = self.dropout(x)
        scores = self.fc(x)
        return scores, states