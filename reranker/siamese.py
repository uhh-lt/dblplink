import torch
import sys
import os
import json
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sentence_transformers import SentenceTransformer

class YourCustomDataset(Dataset):
    def __init__(self, train_path):
        # Initialize your dataset here
        self.sentmodel = SentenceTransformer('bert-base-nli-mean-tokens')
        self.data = []  # Your data
        for line in open(train_path).readlines():
            line = line.strip()
            d = json.loads(line)
            for entsamp in d['entity_samples']:
                self.data.append({'question':d['question']['string'], 'positive': entsamp['goldemb'], 'negative': entsamp['negemb'], 'poslabel':entsamp['goldlabel'], 'goldfuzz':entsamp['goldfuzz']/100.0, 'neglabel':entsamp['neglabel'], 'negfuzz':entsamp['negfuzz']/100.0})

    def __len__(self):
        # Return the total number of samples in your dataset
        return len(self.data)

    def __getitem__(self, index):
        # Retrieve a single sample from your dataset at the given index

        # Fetch the anchor, positive, and negative samples from your data
        anchor = list(self.sentmodel.encode([self.data[index]['question']])[0])+ 201*[0.0]
        positive = list(self.sentmodel.encode([self.data[index]['poslabel']])[0])+[float(x) for x in self.data[index]['positive']]+[self.data[index]['goldfuzz']]
        negative = list(self.sentmodel.encode([self.data[index]['neglabel']])[0])+[float(x) for x in self.data[index]['negative']]+[self.data[index]['negfuzz']]

        # Convert the samples to tensors if needed
        anchor = torch.tensor(anchor)
        positive = torch.tensor(positive)
        negative = torch.tensor(negative)

        return anchor, positive, negative

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()

        # Define your network architecture
        self.embedding = nn.Sequential(
            nn.Linear(969, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

    def forward_once(self, x):
        output = self.embedding(x)
        return output

    def forward(self, anchor, positive, negative):
        anchor_output = self.forward_once(anchor)
        positive_output = self.forward_once(positive)
        negative_output = self.forward_once(negative)
        return anchor_output, positive_output, negative_output

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        distance_positive = torch.norm(anchor - positive, p=2, dim=1)
        distance_negative = torch.norm(anchor - negative, p=2, dim=1)
        loss = torch.relu(distance_positive - distance_negative + self.margin)
        return loss.mean()

# Example usage
# Assuming you have your own dataset class inheriting from torch.utils.data.Dataset

# Instantiate the Siamese network and triplet loss
model = SiameseNetwork()
triplet_loss = TripletLoss()

# Define your optimizer and other hyperparameters
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Create your data loader
traindataset = YourCustomDataset(sys.argv[1])
validdataset = YourCustomDataset(sys.argv[2])
train_data_loader = DataLoader(traindataset, batch_size=512, shuffle=True)
valid_data_loader =  DataLoader(validdataset, batch_size=512, shuffle=True)
num_epochs=10
# Training loop
model_dir = sys.argv[3]
isExist = os.path.exists(model_dir)
if not isExist:
   os.makedirs(model_dir)
for epoch in range(num_epochs):
    for batch in train_data_loader:
        anchor, positive, negative = batch

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        anchor_output, positive_output, negative_output = model(anchor, positive, negative)

        # Calculate triplet loss
        loss = triplet_loss(anchor_output, positive_output, negative_output)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Print loss or other metrics
        print('Epoch [{}/{}], Batch Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
    model_path = model_dir+f"/model_epoch_{epoch+1}.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved for epoch {epoch+1}.")
    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        for val_batch in valid_data_loader:
            val_anchor, val_positive, val_negative = val_batch

            val_anchor_output, val_positive_output, val_negative_output = model(val_anchor, val_positive, val_negative)

            val_batch_loss = triplet_loss(val_anchor_output, val_positive_output, val_negative_output)
            val_loss += val_batch_loss.item()
    
        val_loss /= len(valid_data_loader)
        print('Epoch [{}/{}], Validation Loss: {:.4f}'.format(epoch+1, num_epochs, val_loss))

