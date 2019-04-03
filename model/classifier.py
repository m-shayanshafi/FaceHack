import os
import torch

from torch import nn
from torch.autograd import Variable
from torchvision import models, transforms

import torch.optim as optim

import vgg_face_dag as vgg_face

from tqdm import tqdm_notebook as tqdm

class FaceClassifier(nn.Module):

    def __init__(self, output_dim=10):
        super(FaceClassifier, self).__init__()
        
        # Load base VGG-Face, pre-trained weights
        self.model = vgg_face.Vgg_face_dag()

        weights_path = os.path.join('model', 'vgg_face_dag.pth')
        url = 'http://www.robots.ox.ac.uk/~albanie/models/pytorch-mcn/vgg_face_dag.pth'

        if not os.path.exists(weights_path):
            print('Pre-trained weights not found.\nYou must download the file from\n  ' 
                + url + '\nand place it here:\n  ' + weights_path + '\n')
        
        self.model.load_state_dict(torch.load(weights_path))
        
        # Freeze current layers
        for param in self.model.parameters():
            param.requires_grad = False

        # Replace last fully-connected layer according to desired output dimensions
        self.model._modules['fc8'] = nn.Linear(in_features=4096, out_features=output_dim, bias=True)

    def forward(self, x):
        self.model.forward(x)

    def tune(self, X, y, learning_rate=1e-3, batch_size=50, epochs=3):

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        train_data = torch.utils.data.TensorDataset(X, y)

        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)

        for epoch in tqdm(range(epochs)):

            running_loss = 0.0

            for batch_index, (batch_input, batch_labels) in enumerate(tqdm(train_loader)):
                if torch.cuda.is_available():
                    batch_input = batch_input.cuda()
                    
                # Zero the gradient buffers
                optimizer.zero_grad()
                
                # Compute the output for the batch
                batch_output = self.model(batch_input).cpu()
                            
                # Compute the loss
                loss = criterion(batch_output, batch_labels)
                            
                # Backpropogation to compute the gradients
                loss.backward()
                
                # Update the weights
                optimizer.step()
                
                # Print statistics
                running_loss += loss.item()
                if batch_index % 5 == 4:    # print every 5 mini-batches
                    print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_index + 1, running_loss / 5))
                    running_loss = 0.0

    def get_activations(self, x):
        raise NotImplementedError
