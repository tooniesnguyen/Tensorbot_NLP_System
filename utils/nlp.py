# Import necessary modules and classes
from .preprocess import Word_Processing
from .model import NN_Model
from .conn_mongo import get_random_response
import torch
import json
import sys
import socket
import os
from pathlib import Path
import random
import time
import re

# Get host IP address
HOST = socket.gethostbyname(socket.gethostname())

# Get absolute path of the current file and its parent directories
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
WORK_DIR = os.path.dirname(ROOT)

# Define paths for JSON data and the trained model
JSON_DIR = f"{WORK_DIR}/data/dicts/intents.json"
MODEL_DIR = f"{WORK_DIR}/models/lstm.pth"

# Decorator function for measuring function execution time
def time_complexity(func):
    def warp(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f'Time inference func {func.__name__}: {(time.time() - start):.3f} second')
        return result
    return warp

# Main class representing the chatbot
class Tensorbot:
    def __init__(self, json_path=JSON_DIR, model_path=MODEL_DIR):
        # Load intents data from JSON file
        self.intents = self.load_json(json_path)
        # Set device to GPU if available, otherwise use CPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Initialize Word_Processing instance for text preprocessing
        self.word_process = Word_Processing()
        # Load the trained model
        self.model = self.load_model(model_path)
        # Define bot name
        self.bot_name = "Tensorbot"

    # Load JSON data from file
    def load_json(self, json_path):
        with open(json_path, 'r') as json_data:
            intents = json.load(json_data)
        return intents

    # Load the trained model from file
    def load_model(self, model_path):
        # Load model data from file
        data = torch.load(model_path)
        input_size = data["input_size"]
        hidden_size = data["hidden_size"]
        output_size = len(data["tags"])
        self.all_words = data['all_words']
        self.tags = data['tags']
        model_state = data["model_state"]

        # Initialize and configure the neural network model
        model = NN_Model(input_size, hidden_size, output_size).to(self.device)
        model.load_state_dict(model_state)
        model.eval()
        print("Load successful")
        return model
    
    # @time_complexity
    def feed_back(self, sentence):
        # Preprocess the input sentence
        sentence = self.word_process.tokenize(sentence)
        X = self.word_process.bag_words(sentence, self.all_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X).to(self.device)
        
        # Pass the preprocessed input through the model
        output = self.model(X)
        _, predicted = torch.max(output, dim=1)

        # Get the tag and probability of the predicted class
        tag = self.tags[predicted.item()]
        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]

        # If probability is high, fetch and return a response from intents
        if prob.item() > 0.75:
            for intent in self.intents['intents']:
                text = get_random_response(tag_name=tag)
        else:
            text = "I do not understand you"
            
        return text, tag

    # Method to start the chat mode
    def chat_mode(self):
        while True:
            sentence = input("You: ")
            if sentence == "quit":
                break
            sentence = self.word_process.tokenize(sentence)
            X = self.word_process.bag_words(sentence, self.all_words)
            X = X.reshape(1, X.shape[0])
            X = torch.from_numpy(X).to(self.device)
            
            output = self.model(X)
            _, predicted = torch.max(output, dim=1)

            tag = self.tags[predicted.item()]
            probs = torch.softmax(output, dim=1)
            prob = probs[0][predicted.item()]
            if prob.item() > 0.75:
                for intent in self.intents['intents']:
                    if tag == intent["tag"]:
                        print(f"{self.bot_name}: {random.choice(intent['responses'])}")
            else:
                print(f"{self.bot_name}: I do not understand...")

# Entry point of the script
if __name__ == "__main__":
    # Create a Tensorbot instance
    tensorbot = Tensorbot()
    # Start chat mode
    tensorbot.chat_mode()
    # Uncomment below line to test feedback method
    # print(tensorbot.feed_back("Hello"))
