import torch
import torch.nn as nn
import os
import torch.optim as optim


# Define Transformer model
class TransformerModel(nn.Module):
    def __init__(self, feature_size, hidden_size, output_size, nhead, num_layers):
        super(TransformerModel, self).__init__()
        self.hidden_size = hidden_size
        encoder_layers = nn.TransformerEncoderLayer(feature_size, nhead, hidden_size)
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers)
        self.fc = nn.Linear(feature_size, output_size)

    def forward(self, x):
        x = self.transformer(x)
        x = self.fc(x[-1])  # take the last output of each sequence
        return x.squeeze()

    def train_model(self, train_x, train_y, criterion, optimizer):
        self.train()  # Set the model to training mode
        optimizer.zero_grad()  # Reset gradients
        output = self.forward(train_x)  # Forward pass
        loss = criterion(output, train_y)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights
        return loss.item()

    def evaluate_model(self, test_x, test_y, criterion):
        self.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            test_output = self.forward(test_x)  # Forward pass on test data
            test_loss = criterion(test_output, test_y)  # Compute loss on test data
        return test_loss.item()

    def training_step(self, train_x, train_y, test_x, test_y, learning_rate, num_epochs, save_path):
        """
        Perform a training step.

        Parameters:
        train_x (Tensor): The training input data.
        train_y (Tensor): The training target data.
        criterion: The loss function.
        optimizer: The optimization algorithm.
        num_epochs (int): The number of epochs to train for.

        Returns:
        losses: A list of loss values from each epoch.
        """
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        train_x = torch.tensor(train_x, dtype=torch.float).transpose(0, 1)
        train_y = torch.tensor(train_y, dtype=torch.float)
        test_x = torch.tensor(test_x, dtype=torch.float).transpose(0, 1)
        test_y = torch.tensor(test_y, dtype=torch.float)

        for epoch in range(num_epochs):
            train_loss = self.train_model(train_x, train_y, criterion, optimizer)
            test_loss = self.evaluate_model(test_x, test_y, criterion)
            print(f'Epoch: {epoch}, training Loss: {train_loss}, Test Loss: {test_loss}')

        # List all directories in the path
        dirs = [d for d in os.listdir(save_path) if os.path.isdir(os.path.join(save_path, d))]
        if not dirs:
            last_dir = 1
        else:
            # Convert the directory names to integers for proper sorting
            dirs = [int(d) for d in dirs]
            # Sort the directories
            dirs.sort()
            # Get the last directory
            last_dir = dirs[-1]
            last_dir += 1
        save_dir = os.path.join(save_path, '{}'.format(last_dir))
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(save_dir, 'model.pth'))

    def test_model(self, test_x, test_y):
        test_x = torch.tensor(test_x, dtype=torch.float).transpose(0, 1)
        test_y = torch.tensor(test_y, dtype=torch.float)
        self.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            test_output = self.forward(test_x)  # Forward pass on test data
        return test_output, test_y

