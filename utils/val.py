import json
import torch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader
from train import Data_Preprocessing, ChatDataset, Lstm_Model
import os
from pathlib import Path

# Define the type of model
type_model = "LSTM"

# Define the working directory and paths
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
WORK_DIR = os.path.dirname(ROOT)

# Configuration hyperparameters
BATCH_SIZE = 8

def load_model(model_path, input_size, hidden_size, output_size, device):
    model = Lstm_Model(input_size, hidden_size, output_size)
    model.load_state_dict(torch.load(model_path, map_location=device)['model_state'])
    model.to(device)
    model.eval()
    return model

def main():
    json_dir = f"{WORK_DIR}/data/dicts/valid.json"
    data_process = Data_Preprocessing(json_dir)
    all_words, tags, _ = data_process.create_data()
    X_train, y_train = data_process.X_y_split()
    input_size = 200
    hidden_size = 8  # Make sure this matches the hyperparameter used in training
    output_size = len(tags)
    dataset = ChatDataset(X_train, y_train)
    data_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = f"{WORK_DIR}/models/{type_model}.pth"
    model = load_model(model_path, input_size, hidden_size, output_size, device)

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for words, labels in data_loader:
            words = words.to(device)
            labels = labels.to(dtype=torch.long).to(device)
            outputs = model(words)
            _, predicted = torch.max(outputs.data, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    # Print confusion matrix
    print("Confusion Matrix:")
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)

    # Print classification report
    print("\nClassification Report:")
    cr = classification_report(all_labels, all_preds, target_names=tags)
    print(cr)

if __name__ == "__main__":
    main()
