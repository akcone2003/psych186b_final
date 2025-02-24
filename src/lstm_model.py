"""
LSTM Battle Predictor Module

Contains the LSTM model architecture and training functionality for
predicting battle outcomes based on unit positions and actions.
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


class LSTMBattlePredictor(nn.Module):
    """LSTM model for predicting battle outcomes"""
    def __init__(self, input_size, hidden_size=64, num_layers=1, dropout=0.2):
        super(LSTMBattlePredictor, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # LSTM layer
        lstm_out, _ = self.lstm(x)
        
        # Use only the last output
        last_output = lstm_out[:, -1, :]
        
        # Apply dropout
        x = self.dropout(last_output)
        
        # Feed forward layer
        x = self.fc(x)
        
        # Output probability
        return self.sigmoid(x)


def find_latest_data_file():
    """Find the most recent battle data file"""
    files = [f for f in os.listdir('data') if f.startswith('battle_data') and f.endswith('.csv')]
    if not files:
        raise FileNotFoundError("No battle data files found")
        
    # Sort by timestamp (which is embedded in the filename)
    latest_file = sorted(files, reverse=True)[0]
    print(f"Found latest data file: data/{latest_file}")
    return f"data/{latest_file}"


def load_and_preprocess_data(filepath, test_size=0.2, random_state=42):
    """Load and preprocess battle data for training"""
    try:
        # Load data
        df = pd.read_csv(filepath)
        print(f"Loaded data with {len(df)} entries and columns: {df.columns.tolist()}")
        
        # Display data summary
        print(f"First few rows:\n{df.head()}")
        print(f"Data types:\n{df.dtypes}")
        
        # Prepare features and target
        X = df.iloc[:, :-1].values  # All columns except result
        y = df.iloc[:, -1].values   # Result column
        
        # Normalize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Convert to PyTorch tensors
        X_train = torch.FloatTensor(X_train)
        y_train = torch.FloatTensor(y_train).unsqueeze(1)
        X_test = torch.FloatTensor(X_test)
        y_test = torch.FloatTensor(y_test).unsqueeze(1)
        
        # Create DataLoaders
        batch_size = min(32, len(X_train))  # Adjust batch size for small datasets
        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        return train_loader, test_loader, X.shape[1], scaler
        
    except Exception as e:
        print(f"Error loading data: {e}")
        raise


def train_lstm_model(train_loader, test_loader, input_size, num_epochs=50):
    """Train the LSTM model with early stopping"""
    # Initialize model
    model = LSTMBattlePredictor(
        input_size=input_size,
        hidden_size=64,  # Larger for more data
        num_layers=2,    # More complex model
        dropout=0.3
    )
    
    # Loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    # Training tracking
    train_losses = []
    test_losses = []
    test_accuracies = []
    best_accuracy = 0
    patience = 10
    no_improvement = 0
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        
        for batch_x, batch_y in train_loader:
            # Add time dimension for LSTM (if not already present)
            if len(batch_x.shape) == 2:
                batch_x = batch_x.unsqueeze(1)
                
            # Forward pass
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Calculate average training loss
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Evaluation phase
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                # Add time dimension for LSTM (if not already present)
                if len(batch_x.shape) == 2:
                    batch_x = batch_x.unsqueeze(1)
                    
                # Forward pass
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                test_loss += loss.item()
                
                # Calculate accuracy
                predicted = (outputs > 0.5).float()
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        
        # Calculate average test loss and accuracy
        avg_test_loss = test_loss / len(test_loader)
        test_losses.append(avg_test_loss)
        
        accuracy = 100 * correct / total
        test_accuracies.append(accuracy)
        
        # Update learning rate
        scheduler.step(avg_test_loss)
        
        # Print progress
        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Test Loss: {avg_test_loss:.4f} | "
              f"Accuracy: {accuracy:.2f}% | "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Early stopping check
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            # Save the best model
            torch.save(model.state_dict(), 'models/best_battle_predictor.pt')
            print(f"✓ Model improved, saved checkpoint")
            no_improvement = 0
        else:
            no_improvement += 1
            if no_improvement >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    # Load best model
    model.load_state_dict(torch.load('models/best_battle_predictor.pt', weights_only=True))
    
    # Plot results
    plot_training_results(train_losses, test_losses, test_accuracies)
    
    return model


def plot_training_results(train_losses, test_losses, test_accuracies):
    """Plot training metrics"""
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(12, 6))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, test_losses, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, test_accuracies, 'g-', label='Validation Accuracy')
    plt.title('Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('visualizations/battle_predictor_training.png')
    
    # Terminal summary
    print(f"\nTraining Results:")
    print(f"Starting Loss: {train_losses[0]:.4f}")
    print(f"Final Loss: {train_losses[-1]:.4f}")
    print(f"Best Accuracy: {max(test_accuracies):.2f}%")
    print(f"Training visualization saved to 'visualizations/battle_predictor_training.png'")


def load_model(model_path='models/best_battle_predictor.pt', input_size=11):
    """Load a trained model from disk"""
    model = LSTMBattlePredictor(input_size=input_size, hidden_size=64, num_layers=2)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    return model


def predict_battle_outcome(model, battle_data):
    """
    Predict battle outcome from a single battle state
    
    Parameters:
        model: Trained LSTM model
        battle_data: List containing [Infantry_x, Infantry_y, Tank_x, ...] for a battle state
        
    Returns:
        win_probability: Probability of victory (0-1)
    """
    # Convert to tensor and add batch/time dimensions
    x = torch.FloatTensor(battle_data).unsqueeze(0).unsqueeze(0)
    
    # Make prediction
    with torch.no_grad():
        win_probability = model(x).item()
    
    return win_probability


def main():
    """Main function to train model from data"""
    try:
        # Find the latest data file
        data_file = find_latest_data_file()
        
        # Load and preprocess data
        train_loader, test_loader, input_size, _ = load_and_preprocess_data(data_file)
        
        # Train model
        model = train_lstm_model(train_loader, test_loader, input_size)
        
        print("\nTraining complete! Model saved as 'best_battle_predictor.pt'")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()