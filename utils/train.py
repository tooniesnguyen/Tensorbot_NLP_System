from preprocess import Word_Processing
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import *
import sys
import socket
import os
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib.pyplot as plt
import seaborn as sns


type_model = "rnn"
HOST = socket.gethostbyname(socket.gethostname())
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
WORK_DIR = os.path.dirname(ROOT)
writer = SummaryWriter(f'{WORK_DIR}/models/runs/{type_model}')



############## CONFIG HYPERPARAMETER ##############
NUM_EPOCHS = 800
BATCH_SIZE = 8
LEARNING_RATE = 0.0001
HIDDEN_SIZE = 8
###################################################



class Data_Preprocessing:
    def __init__(self, json_dir) -> None:
        self.content_json = self.load_json(json_dir)
        self.word_process = Word_Processing()
        self.all_words =[]
        self.tags = []
        self.data = []

    def load_json(self, json_dir):
        with open(json_dir, 'r') as f:
            return json.load(f)

    def create_data(self):
        ignore_words = ['?', '.', '!', '*']
        for content in self.content_json['intents']:
            tag = content['tag']
            self.tags.append(tag)
            for pattern in content["patterns"]:
                word = self.word_process.tokenize(pattern)
                self.all_words.extend(word)
                self.data.append([word, tag])
        self.all_words = [self.word_process.lemma(w) for w in self.all_words if w not in ignore_words]
        self.all_words = sorted(set(self.all_words))
        self.tags = sorted(set(self.tags))
        return self.all_words, self.tags, self.data
    
    def X_y_split(self):
        X = []
        y = []
        for (pattern_sentence, tag) in self.data:
            bag = self.word_process.bag_words(pattern_sentence, self.all_words)
            X.append(bag)
            label = self.tags.index(tag)
            y.append(label)
        X = np.array(X)
        y = np.array(y)
        return X, y
    
class ChatDataset(Dataset):
    def __init__(self, X_train, y_train):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples


def main():
    json_dir = f"{WORK_DIR}/data/dicts/intents.json"
    data_process = Data_Preprocessing(json_dir)
    all_words, tags, _ =data_process.create_data()
    X, y = data_process.X_y_split()
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    for train_index, valid_index in sss.split(X, y):
        X_train, X_valid = X[train_index], X[valid_index]
        y_train, y_valid = y[train_index], y[valid_index]

    print("X_shape", X_train.shape)
    num_epochs = NUM_EPOCHS
    batch_size = BATCH_SIZE
    learning_rate = LEARNING_RATE
    hidden_size = HIDDEN_SIZE
    input_size = len(X_train[0])
    output_size = len(tags)
    dataset = ChatDataset(X_train, y_train)
    train_loader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NN_Model(input_size, hidden_size, output_size, type_model=type_model).to(device)
    model.count_parameter()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    dummy_input = torch.zeros(1, input_size).to(device)
    writer.add_graph(model, dummy_input)
    writer.close()

    running_loss = 0.0
    running_correct = 0.0
    n_total_steps = len(train_loader) 
    for epoch in range(num_epochs):
        # Iterate over batches in the training data
        for (words, labels) in train_loader:
            # Move data to the appropriate device (GPU or CPU)
            words = words.to(device)
            labels = labels.to(dtype=torch.long).to(device)
            
            # Forward pass: compute predicted outputs
            outputs = model(words)

            # Compute the loss
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization: compute gradients and update model parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track training statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            running_correct += (predicted == labels).sum().item()
            # print("run corre", running_correct)
            
        # Print epoch-wise statistics and log to TensorBoard every 100 epochs
        if (epoch+1) % 100 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
            # Log training loss and accuracy to TensorBoard
            writer.add_scalar('training loss', running_loss / 10/ n_total_steps, epoch )
            running_accuracy = running_correct / (8*10) / n_total_steps 
            writer.add_scalar('accuracy', running_accuracy, epoch )

            # Reset running statistics
            running_correct = 0
            running_loss = 0.0

    # Print final loss after training
    print(f'final loss: {loss.item():.4f}')

    # Save the trained model
    model.save_model(model, all_words, tags, f"{WORK_DIR}/models/{type_model}.pth")
    
    ######################################## Validation Score #############################################
    # Create DataLoader for validation data
    valid_dataset = ChatDataset(X_valid, y_valid)
    valid_data_loader = DataLoader(dataset=valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    all_labels = []
    all_preds = []
    # Evaluate the model on validation data
    with torch.no_grad():
        for words, labels in valid_data_loader:
            # Move data to the appropriate device (GPU or CPU)
            words = words.to(device)
            labels = labels.to(dtype=torch.long).to(device)
            outputs = model(words)
            _, predicted = torch.max(outputs.data, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())


    # Print confusion matrix
    print("Confusion Matrix:")
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, cmap='Greens', annot=True, fmt='d', linewidths=4, annot_kws={'fontsize': 18}, 
            yticklabels=['info', 'thanks', 'moving', 'tour'], xticklabels=['Predict info', 'Predict thanks', 'Predict moving', 'Predict tour'])
    plt.yticks(rotation=0)
    plt.title(f'Confusion Matrix of {type_model}')
    plt.savefig(f"{WORK_DIR}/images/{type_model}.jpg")
    plt.close()
    # Print classification report
    print("\nClassification Report:")
    cr = classification_report(all_labels, all_preds, target_names=tags)
    print(cr)
    

    
if __name__ == "__main__":
    main()
