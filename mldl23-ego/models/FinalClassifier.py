from torch import nn
import torch

# class Classifier(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.num_classes = 8
#         self.num_clips = 5 
#         self.lstm1 = nn.LSTM(input_size=5, hidden_size=10, num_layers=1, batch_first=True)
#         self.lstm2 = nn.LSTM(input_size=10, hidden_size=4, num_layers=1, batch_first=True)
#         self.flat =  nn.Flatten()
#         self.linear1 = nn.Linear(4096,128)
#         self.linear2 = nn.Linear(128,self.num_classes)
       


        

#     def forward(self, x):
#         x = torch.permute(x,(0,2,1))
        
        
#         x, _ = self.lstm1(x)
#         x, _ = self.lstm2(x)
        
#         x = self.flat(x)
#         print(' \n\n \n  {} \n \n '.format(x.shape))
#         x = self.linear1(x)
#         x = self.linear2(x)
        
#         return x, {}

class Classifier(nn.Module):
    def __init__(self, dim_input=1024, num_classes=8):
        super().__init__()
        #self.classifier = nn.Linear(dim_input, num_classes)
        self.model = nn.Sequential(
            nn.Linear(dim_input, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )
#should return logits and features
#features is ignored for now
    def forward(self, x):
        return self.model(x), {}
       # return self.classifier(x), {}
