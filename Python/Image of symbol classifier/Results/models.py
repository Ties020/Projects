import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN3train(nn.Module):
    def __init__(self, classes):
        super(CNN3train, self).__init__()
        self.classes = classes
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(3, 100, 3),
            nn.BatchNorm2d(100),
            nn.ReLU(),
            nn.Conv2d(100, 100, 3),
            nn.BatchNorm2d(100),
            nn.ReLU(),
            nn.Conv2d(100, 100, 3),
            nn.BatchNorm2d(100),
            nn.ReLU(),
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(100, 200, 3),
            nn.BatchNorm2d(200),
            nn.ReLU(),
            nn.Conv2d(200, 200, 3),
            nn.BatchNorm2d(200),
            nn.ReLU(),
            nn.Conv2d(200, 200, 3),
            nn.BatchNorm2d(200),
            nn.ReLU(),
        )
        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(200, 300, 3),
            nn.BatchNorm2d(300),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(300, 300, 3),
            nn.BatchNorm2d(300),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(300, 300, 3),
            nn.BatchNorm2d(300),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.conv_block_4 = nn.Sequential(
            nn.Conv2d(300, 400, 3),
            nn.BatchNorm2d(400),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(400, 400, 3),
            nn.BatchNorm2d(400),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(400, 400, 3),
            nn.BatchNorm2d(400),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.conv_block_5 = nn.Sequential(
            nn.Conv2d(400, 512, 3),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(1, 2)
        self.pool3 = nn.MaxPool2d(2, 1)
        self.fc1 = nn.Linear(512 * 2 * 2, 1024)  
        self.fc2 = nn.Linear(1024, len(self.classes))

    def forward(self, x):  
        x = self.pool1(self.conv_block_1(x))
        x = self.pool1(self.conv_block_2(x))
        x = self.pool2(self.conv_block_3(x))
        x = self.pool3(self.conv_block_4(x))
        x = self.conv_block_5(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = nn.ReLU()(x)  
        x = self.fc2(x) 
        return x
    
class CNN3(nn.Module):
    def __init__(self):
        super(CNN3, self).__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(3, 100, 3),
            nn.BatchNorm2d(100),
            nn.ReLU(),
            nn.Conv2d(100, 100, 3),
            nn.BatchNorm2d(100),
            nn.ReLU(),
            nn.Conv2d(100, 100, 3),
            nn.BatchNorm2d(100),
            nn.ReLU(),
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(100, 200, 3),
            nn.BatchNorm2d(200),
            nn.ReLU(),
            nn.Conv2d(200, 200, 3),
            nn.BatchNorm2d(200),
            nn.ReLU(),
            nn.Conv2d(200, 200, 3),
            nn.BatchNorm2d(200),
            nn.ReLU(),
        )
        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(200, 300, 3),
            nn.BatchNorm2d(300),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(300, 300, 3),
            nn.BatchNorm2d(300),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(300, 300, 3),
            nn.BatchNorm2d(300),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.conv_block_4 = nn.Sequential(
            nn.Conv2d(300, 400, 3),
            nn.BatchNorm2d(400),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(400, 400, 3),
            nn.BatchNorm2d(400),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(400, 400, 3),
            nn.BatchNorm2d(400),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.conv_block_5 = nn.Sequential(
            nn.Conv2d(400, 512, 3),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(1, 2)
        self.pool3 = nn.MaxPool2d(2, 1)

    def forward(self, x):  
        x = self.pool1(self.conv_block_1(x))
        x = self.pool1(self.conv_block_2(x))
        x = self.pool2(self.conv_block_3(x))
        x = self.pool3(self.conv_block_4(x))
        x = self.conv_block_5(x) 
        return x

class RowEncoder(nn.Module):
    def __init__(self, num_features, hidden_size):
        super(RowEncoder, self).__init__()
        self.blstm = nn.LSTM(num_features, hidden_size, num_layers = 1, bidirectional = True, batch_first = True) #Make sure batch is first argument as output, full output of BLSTM is tensor of shape (batch_size, number of elements per row, hidden_size * 2)
        self.fc = nn.Linear(hidden_size * 2, hidden_size) #Output of bltsm is reduced from hidden_size * 2 to hidden_size, this is for decoding

    def forward(self, F):
        batch_size, K, L, D = F.shape
        F_reshaped = F.view(batch_size * K, L, D) #Reshape for row-wise BLSTM: (batch_size * K, L, D)
        
        blstm_out, _ = self.blstm(F_reshaped)  #Output: (batch_size * K, L, hidden_dim * 2)
        blstm_out = self.fc(blstm_out)  #Reduce to hidden_size using a fully connected layer
        F_prime = blstm_out.view(batch_size, K, L, -1) #Reshape back to (batch_size, K, L, hidden_dim
        return F_prime
    
class RowDecoder(nn.Module):
    def __init__(self, hidden_size, output_size, embedding_dim):
        super(RowDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(output_size, embedding_dim) #This converts the label id of symbol to its feature vector 
        self.lstm = nn.LSTM(input_size = embedding_dim + hidden_size, hidden_size = hidden_size, batch_first = True) #Input is context vector and current symbol's representation
        self.attn = nn.Linear(hidden_size + hidden_size, 1) #Computes attention scores, input is combination of encoder's output and decoder's hidden size
        self.fc_out = nn.Linear(hidden_size, output_size) #Takes the LSTM's hidden state and maps it to the output_size to produce the probabilities for the next token
        
    def attention(self, hidden_state, encoder_outputs):
        #Computes attention weights
        batch_size = encoder_outputs.size(0)
        K, L, hidden_size = encoder_outputs.size(1), encoder_outputs.size(2), encoder_outputs.size(3)
        #Corresponds to W_h' * h_t
        hidden_reshaped = hidden_state[-1].unsqueeze(1).repeat(1, K * L, 1)  #This is done to reshape the hidden_state so that the attention score of the current hidden state with respect to each encoder position can be computed
        #Corresponds to W_F' * F'_u,v
        encoder_outputs_flat = encoder_outputs.view(batch_size, -1, hidden_size)  #This prepares the encoder outputs to match the shape of the decoder's reshaped hidden state (multiplies K and L resulting in 1 parameter)
        attn_scores = self.attn(torch.cat((hidden_reshaped, encoder_outputs_flat), dim = 2))
        attn_scores = attn_scores.squeeze(-1)  #Removes last dimension which is redundant since second dimension holds attention scores for each encoder position
        attn_weights = F.softmax(attn_scores, dim = 1) #Computes softmax per row (dim = 1)
        return attn_weights

    def forward(self, decoder_input, hidden_state, cell, encoder_outputs):
        embedded = self.embedding(decoder_input)  #Computes y_t-1 
        attn_weights = self.attention(hidden_state, encoder_outputs)  
        batch_size = encoder_outputs.size(0)
        hidden_size = encoder_outputs.size(3)
        encoder_outputs_flat = encoder_outputs.view(batch_size, -1, hidden_size)  #Flattens encoder output to shape (batch_size, K x L, hidden_size)
        context_vector = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs_flat) #This is equation 14, performing bmm on cnn output grid 
        lstm_input = torch.cat((embedded, context_vector), dim = -1) #This corresponds to [y_t-1 , O_t-1] in equation 13
        #Even though cell state isn't in equation 13, we included it in the LSTM since it helps the model remember long-term dependencies, influencing how the hidden state is calculated at each step
        output, (hidden_state, cell) = self.lstm(lstm_input, (hidden_state, cell))  #Returns hidden state for current time step and the hidden and cell state for the next step, (hidden_state_cell) corresponds t 
        output = output.squeeze(1)  #Remove redundant second dimension which is always 1, since that is seq_len and we consider 1 token
        output = self.fc_out(output) # Transform (batch_size, hidden_size) to (batch_size, output_size), where output_size is the number of possible classes 
        return output, hidden_state, cell, attn_weights
