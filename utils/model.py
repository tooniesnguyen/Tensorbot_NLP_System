import torch  # Import the PyTorch library
from torch import nn  # Import the neural network module from PyTorch
from prettytable import PrettyTable  # Import PrettyTable for displaying tabular data in a formatted table


# Define a neural network model class that inherits from nn.Module
class NN_Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers=2, type_model="lstm"):
        # Call the parent class's constructor
        super(NN_Model, self).__init__()
        # Initialize the input size, hidden size, and number of layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # Set the device to GPU if available, otherwise use CPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Set the type of model (LSTM, GRU, or RNN)
        self.type_model = type_model
        # Initialize the appropriate recurrent layer based on the model type
        if self.type_model == "lstm":
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        elif self.type_model == "gru":
            self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        elif self.type_model == "rnn":
            self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        # Initialize a fully connected layer for the output
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Initialize hidden state and cell state (for LSTM) or only hidden state (for GRU/RNN)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        
        # Ensure x has the correct dimensions
        x = x.unsqueeze(2)  # Add a dummy dimension for seq_len
        x = x.permute(0, 2, 1)  # Swap seq_len and input_size dimensions
        
        # Pass the input through the appropriate recurrent layer
        if self.type_model == "lstm":
            out, _ = self.lstm(x, (h0, c0))
        elif self.type_model == "gru":
            out, _ = self.gru(x, h0)
        elif self.type_model == "rnn":
            out, _ = self.rnn(x, h0)
        
        # Get only the output from the last time step
        out = out[:, -1, :]
        # Pass the output through the fully connected layer
        out = self.fc(out)
        return out
    
    def count_parameter(self):
        # Create a PrettyTable instance to display module parameters
        table = PrettyTable(["Modules", "Parameters"])
        total_params = 0
        # Iterate over all named parameters in the model
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue  # Skip parameters that do not require gradients
            param = parameter.numel()  # Get the number of elements in the parameter
            table.add_row([name, param])  # Add a row to the table with module name and parameter count
            total_params += param  # Add to the total parameter count
        print(table)  # Print the table
        print(f"Total Trainable Params: {total_params}")  # Print the total number of trainable parameters
        return total_params  # Return the total number of trainable parameters
    
    def save_model(self, model, all_words, tags, save_dir):
        # Prepare a dictionary to save the model state and other relevant information
        data = {
            "model_state": model.state_dict(),
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "output_size": self.num_layers,
            "all_words": all_words,
            "tags": tags
        }
        # Save the dictionary to the specified directory using torch.save
        torch.save(data, save_dir)
        print(f'training complete. file saved to {save_dir}')  # Print a message indicating successful save
        return True  # Return True to indicate success
