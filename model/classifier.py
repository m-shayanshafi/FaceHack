import os
import torch
import numpy as np

from torch import nn
from torch.autograd import Variable
from torchvision import models, transforms

import torch.optim as optim

import vgg_face_dag as vgg_face

from tqdm import tqdm_notebook as tqdm

class FaceClassifier(nn.Module):

    def __init__(self, output_dim=10, unfreeze=1):
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
        if unfreeze > 1:
            if unfreeze > 2:
                self.model._modules['fc6'] = nn.Linear(in_features=25088, out_features=4096, bias=True)
            
            self.model._modules['fc7'] = nn.Linear(in_features=4096, out_features=4096, bias=True)
        
        self.model._modules['fc8'] = nn.Linear(in_features=4096, out_features=output_dim, bias=True)

    def forward(self, x):
        return self.model.forward(x)

    def tune(self, X, y, X_val, y_val, learning_rate=1e-3, batch_size=50, epochs=3):
        """
        Fine-tune the model using the given input X and labels y
        
        Args:
            X: Torch Variable (Float Tensor) for inputs
            y: Torch Variable (Long Tensor) for labels
            X_val: (Optional) Torch Variable (Float Tensor) for validation set inputs
            y_val: (Optional) Torch Variable (Long Tensor) for validation set labels
        """
        if torch.cuda.is_available():
            self.model = self.model.cuda()

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
            
            if X_val is not None and y_val is not None:
                accuracy = 0

                eval_data = torch.utils.data.TensorDataset(X_val, y_val)

                eval_loader = torch.utils.data.DataLoader(eval_data, batch_size=50, shuffle=True, num_workers=0)

                for batch_index, (batch_input, batch_labels) in enumerate(tqdm(eval_loader)):
                    if torch.cuda.is_available():
                        batch_input = batch_input.cuda()

                    # Compute the output for the batch
                    batch_output = self.model(batch_input).cpu()

                    values, indices = torch.max(batch_output,1)

                    indices = indices.data.cpu().squeeze(0).numpy()

                    # Count matches
                    for i,label in enumerate(batch_labels):
                    
                        if indices[i] == label:
                            accuracy += 1

                accuracy = accuracy / len(y_val)
    
                print('\nClassification Accuracy (Validation Set): %0.3f' % accuracy)

    def forward_activations(self, x0):
        """
        Compute the activations by doing a forward pass through the network
        """

        x1 = self.model.conv1_1(x0)
        x2 = self.model.relu1_1(x1)
        x3 = self.model.conv1_2(x2)
        x4 = self.model.relu1_2(x3)
        x5 = self.model.pool1(x4)
        x6 = self.model.conv2_1(x5)
        x7 = self.model.relu2_1(x6)
        x8 = self.model.conv2_2(x7)
        x9 = self.model.relu2_2(x8)
        x10 = self.model.pool2(x9)
        x11 = self.model.conv3_1(x10)
        x12 = self.model.relu3_1(x11)
        x13 = self.model.conv3_2(x12)
        x14 = self.model.relu3_2(x13)
        x15 = self.model.conv3_3(x14)
        x16 = self.model.relu3_3(x15)
        x17 = self.model.pool3(x16)
        x18 = self.model.conv4_1(x17)
        x19 = self.model.relu4_1(x18)
        x20 = self.model.conv4_2(x19)
        x21 = self.model.relu4_2(x20)
        x22 = self.model.conv4_3(x21)
        x23 = self.model.relu4_3(x22)
        x24 = self.model.pool4(x23)
        x25 = self.model.conv5_1(x24)
        x26 = self.model.relu5_1(x25)
        x27 = self.model.conv5_2(x26)
        x28 = self.model.relu5_2(x27)
        x29 = self.model.conv5_3(x28)
        x30 = self.model.relu5_3(x29)
        x31_preflatten = self.model.pool5(x30)
        x31 = x31_preflatten.view(x31_preflatten.size(0), -1)
        x32 = self.model.fc6(x31)
        x33 = self.model.relu6(x32)
        x34 = self.model.dropout6(x33)
        x35 = self.model.fc7(x34)
        x36 = self.model.relu7(x35)
        # x37 = self.model.dropout7(x36)
        # x38 = self.model.fc8(x37)

        return x36

    def get_activations(self, X):
        """
        return the activations for a given input
        """
        
        data = torch.utils.data.TensorDataset(X,X)
        loader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=False, num_workers=0)

        activations = np.zeros((X.shape[0], 4096))

        for batch_index, (batch_input, batch_labels) in enumerate(tqdm(loader)):
            
            if torch.cuda.is_available():
                batch_input = batch_input.cuda()

            # Compute the output for the batch
            batch_output = self.forward_activations(batch_input).cpu()
            #activations[batch_index,:] = [1 if output > 0 else 0 for output in batch_output[0]]
            activations[batch_index,:] = batch_output

        return activations

    def predict(self, X):
        """
        return the predicted class for a given input
        """

        output = self.model(X).cpu()

        values, indices = torch.max(output,1)

        return indices
