import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

# Set random seed for reproducibility
torch.manual_seed(420)
np.random.seed(420)

DIR = "mech_interp_data"


MAX_VAL_IN_DATASET = 148.0

# Function to standardize data
def process_data(data, is_train=False):
    if "goal" in data.columns:
        data = data.drop(columns=["goal"])
    if "response" in data.columns:
        data = data.drop(columns=["response"])
    # Fill missing values with 0
    data = data.fillna(0)

    # Select only numeric columns (if there are non-numeric ones)
    numeric_data = data.select_dtypes(include=[np.number])

    # Get the global maximum from the numeric data
    if is_train:
        max_val = numeric_data.values.max()
        print(f"Max value in dataset: {max_val}")
    else:
        max_val = MAX_VAL_IN_DATASET

    if max_val == 0:
        raise ValueError("The maximum value in the dataset is 0. Cannot scale data.")

    # Divide the entire DataFrame by the maximum value
    # If you want to keep non-numeric columns, you can merge them back afterwards
    return data / max_val


# Define the 3-layer MLP model
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)  # Add dropout for regularization
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.dropout(x)  # Apply dropout after activation
        x = self.layer2(x)
        x = self.relu(x)
        x = self.dropout(x)  # Apply dropout after activation
        x = self.output(x)
        # Note: Don't use softmax here if using nn.CrossEntropyLoss
        # as it already includes softmax
        return x


# Function to prepare data
def prepare_data(X, y, train_size=0.9):
    # Standardize features

    # Convert to PyTorch tensors
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y)

    # Create dataset
    dataset = TensorDataset(X_tensor, y_tensor)

    # Split into train and validation
    train_size = int(len(dataset) * train_size)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    return train_dataset, val_dataset


# Function to train the model
def train_model(
    model, train_loader, val_loader, criterion, optimizer, num_epochs, device
):
    model.to(device)
    train_losses = []
    val_losses = []
    val_accuracies = []

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        # Validation phase
        model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # Get predictions
                _, preds = torch.max(outputs, 1)

                running_loss += loss.item() * inputs.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        epoch_val_loss = running_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)

        # Calculate accuracy
        val_accuracy = accuracy_score(all_labels, all_preds)
        val_accuracies.append(val_accuracy)

        print(f"Epoch {epoch + 1}/{num_epochs}:")
        print(f"  Train Loss: {epoch_train_loss:.4f}")
        print(f"  Val Loss: {epoch_val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

    return train_losses, val_losses, val_accuracies


# Function to evaluate the model
def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Get predictions
            _, preds = torch.max(outputs, 1)

            running_loss += loss.item() * inputs.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_loss = running_loss / len(test_loader.dataset)
    test_accuracy = accuracy_score(all_labels, all_preds)

    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds))

    return test_loss, test_accuracy, all_preds, all_labels


# Main function to run the training pipeline
def run_classification_mlp(
    X,
    y,
    hidden_size=128,
    batch_size=32,
    learning_rate=0.001,
    num_epochs=50,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    # Get number of features and classes
    input_size = X.shape[1]
    num_classes = len(np.unique(y))

    # Prepare data
    train_dataset, val_dataset = prepare_data(X, y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Create model
    model = MLP(input_size, hidden_size, num_classes)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train model
    print(f"Training on {device}...")
    train_losses, val_losses, val_accuracies = train_model(
        model, train_loader, val_loader, criterion, optimizer, num_epochs, device
    )

    # Plot training curves
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()

    return model


# Make predictions on new data (example)
def predict_with_model(model, new_data: pd.DataFrame, device):
    model.eval()
    model.to(device)
    remove_columns = ["goal", "response", "Unnamed: 0", "Unnamed: 0.1"]
    # Process data
    X_scaled = process_data(
        new_data[[x for x in new_data.columns if x not in remove_columns]]
    )

    all_columns = list(
        set(pd.read_csv(f"{DIR}/mech_interp_false_data.csv").columns).union(
            set(pd.read_csv(f"{DIR}/mech_interp_true_data.csv").columns)
        )
    )
    filtered_columns = [col for col in all_columns if col not in remove_columns]

    # Align the columns for both DataFrames, filling missing values with 0
    X_aligned = X_scaled.reindex(columns=filtered_columns, fill_value=0)

    X_scaled = X_aligned.values
    X_tensor = torch.FloatTensor(X_scaled).to(device)

    # Make predictions
    with torch.no_grad():
        outputs = model(X_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        _, predictions = torch.max(outputs, 1)

    return predictions.cpu().numpy(), probabilities.cpu().numpy()


# Example of loading and using the model
def load_and_use_model(model_path: str, new_data: pd.DataFrame):
    # Load the saved model
    checkpoint = torch.load(model_path)

    # Recreate the model architecture
    loaded_model = MLP(
        input_size=checkpoint["input_size"],
        hidden_size=checkpoint["hidden_size"],
        num_classes=checkpoint["num_classes"],
    )

    # Load the weights
    loaded_model.load_state_dict(checkpoint["model_state_dict"])

    # Make predictions
    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictions, probabilities = predict_with_model(loaded_model, new_data, device)

    return predictions, probabilities


# Load and prepare data from two CSV files (one class per file)
if __name__ == "__main__":
    # Load data from the two CSV files
    # Assuming each CSV has features in columns and no label column
    class0_data = pd.read_csv(f"{DIR}/mech_interp_false_data.csv")
    class1_data = pd.read_csv(f"{DIR}/mech_interp_true_data.csv")
    
    class0_data_numerical = class0_data.drop(columns=["goal", "response", "Unnamed: 0"])
    class1_data_numerical = class1_data.drop(columns=["goal", "response", "Unnamed: 0", "Unnamed: 0.1"])
    # print(class0_data_numerical.columns)
    # print(class1_data_numerical.columns)    
    # exit()

    # Create labels (0 for first file, 1 for second file)
    class0_labels = np.zeros(len(class0_data_numerical))
    class1_labels = np.ones(len(class1_data_numerical))

    # Get the union of columns
    all_columns = list(set(class0_data_numerical.columns).union(set(class1_data_numerical.columns)))

    # Align the columns for both DataFrames, filling missing values with 0
    class0_aligned = class0_data_numerical.reindex(columns=all_columns, fill_value=0)
    class1_aligned = class1_data_numerical.reindex(columns=all_columns, fill_value=0)

    # Combine data and labels
    X = pd.concat([class0_aligned, class1_aligned])
    # Normalize data
    X_processed = process_data(X, is_train=True)
    y = np.concatenate([class0_labels, class1_labels])

    # Train the model
    model = run_classification_mlp(
        X_processed.values, y, hidden_size=128, batch_size=32, learning_rate=0.001, num_epochs=30
    )

    # Save the model
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_size": X.shape[1],
            "hidden_size": 128,
            "num_classes": len(np.unique(y)),
        },
        "binary_classifier.pt",
    )

    print("Model training complete and saved to 'binary_classifier.pt'")
