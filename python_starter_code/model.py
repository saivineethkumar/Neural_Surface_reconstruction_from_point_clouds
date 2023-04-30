import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.nn.utils.weight_norm as weight_norm


class Decoder(nn.Module):
    def __init__(self,args,dropout_prob=0.1,):
        super(Decoder, self).__init__()

        # **** YOU SHOULD IMPLEMENT THE MODEL ARCHITECTURE HERE ****
        # Define the network architecture based on the figure shown in the assignment page.
        # Read the instruction carefully for layer details.
        # Pay attention that your implementation should include FC layers, weight_norm layers,
        # PReLU layers, Dropout layers and a tanh layer.
        self.fc1 = weight_norm(nn.Linear(3, 512))
        self.fc2 = weight_norm(nn.Linear(512, 512))
        self.fc3 = weight_norm(nn.Linear(512, 512))
        self.fc4 = weight_norm(nn.Linear(512, 509))
        self.fc5 = weight_norm(nn.Linear(512, 512))
        self.fc6 = weight_norm(nn.Linear(512, 512))
        self.fc7 = weight_norm(nn.Linear(512, 512))
        self.fc8 = nn.Linear(512, 1)
        
        self.prelu = nn.PReLU()
        self.dropout = nn.Dropout(p=dropout_prob)
        self.tanh = nn.Tanh()


    # input: N x 3
    def forward(self, input):

        # **** YOU SHOULD IMPLEMENT THE FORWARD PASS HERE ****
        # Based on the architecture defined above, implement the feed forward procedure
        x = self.dropout(self.prelu(self.fc1(input)))
        x = self.dropout(self.prelu(self.fc2(x)))
        x = self.dropout(self.prelu(self.fc3(x)))
        x = self.fc4(x)
        x = torch.cat((x, input), dim=1)
        x = self.dropout(self.prelu(x))
        x = self.dropout(self.prelu(self.fc5(x)))
        x = self.dropout(self.prelu(self.fc6(x)))
        x = self.dropout(self.prelu(self.fc7(x)))
        x = self.fc8(x)
        x = self.tanh(x)
        return x
