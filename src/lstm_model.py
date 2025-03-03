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
from sklearn.preprocessing import StandardScaler, RobustScaler
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
        # Add input validation to catch NaN values early
        if torch.isnan(x).any():
            print("WARNING: NaN values detected in input")
            # Replace NaNs with zeros
            x = torch.nan_to_num(x, nan=0.0)
            
        # LSTM layer
        lstm_out, _ = self.lstm(x)
        
        # Use only the last output
        last_output = lstm_out[:, -1, :]
        
        # Apply dropout
        x = self.dropout(last_output)
        
        # Feed forward layer
        x = self.fc(x)
        
        # Make sure to apply sigmoid to get values between 0 and 1
        x = self.sigmoid(x)
        
        # Double check the values are in range [0,1]
        return torch.clamp(x, min=0.0, max=1.0)


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
        
        # Check for NaN values in the dataset
        nan_count = df.isna().sum().sum()
        print(f"Total NaN values in dataset: {nan_count}")
        if nan_count > 0:
            print("Columns with NaN values:")
            print(df.isna().sum()[df.isna().sum() > 0])
        
        # Filter only completed battles
        if 'Result' in df.columns:
            # Get final states for each battle
            final_states = df[df['Result'].isin(['victory', 'defeat', 'timeout'])]
            
            # If we don't have many final states, an alternative is to look for step resets
            if len(final_states) < 10:  # Arbitrary threshold to ensure we have enough data
                # Find step resets (where step goes from a high value back to 0/1)
                step_series = df['Step'].values
                battle_ends = np.where(np.diff(step_series) < 0)[0] + 1
                battle_ranges = []
                start = 0
                for end in battle_ends:
                    battle_ranges.append((start, end))
                    start = end
                battle_ranges.append((start, len(df)))
                
                # Extract final state of each battle
                final_states = pd.DataFrame()
                for start, end in battle_ranges:
                    if end > start:  # Ensure valid range
                        battle_data = df.iloc[start:end]
                        # Take the last step of this battle section
                        final_states = pd.concat([final_states, battle_data.iloc[[-1]]])
            
            print(f"Extracted {len(final_states)} final battle states")
            
        else:
            # Without a Result column, we need another approach
            # Find where Step resets to 0, which should indicate a new battle
            battle_ids = (df['Step'] == 0).cumsum()
            # Get the max step for each battle_id
            max_steps = df.groupby(battle_ids)['Step'].transform('max')
            # Final states are where Step equals max_step for that battle
            final_states = df[df['Step'] == max_steps]
        
        # Check if we have enough data
        if len(final_states) == 0:
            raise ValueError("No final battle states found in data")
        
        # Identify target column - should be 'Result' in your data
        if 'Result' in final_states.columns:
            # Convert result to binary (1 for victory, 0 for others)
            final_states['outcome_binary'] = final_states['Result'].apply(
                lambda x: 1 if x == 'victory' else 0
            )
            print(f"Victory rate: {final_states['outcome_binary'].mean() * 100:.2f}%")
        else:
            # If no Result column, try to determine outcome some other way
            # For now, let's assume 50% victory rate
            final_states['outcome_binary'] = np.random.randint(0, 2, size=len(final_states))
            print("Warning: No 'Result' column found. Random outcomes assigned.")
        
        # Select features for training - all columns except 'Result' and 'outcome_binary'
        feature_cols = [col for col in final_states.columns 
                       if col not in ['Result', 'outcome_binary', 'source_file']]
        
        # Ensure all feature columns are numeric
        for col in feature_cols:
            if final_states[col].dtype == 'object':
                # For object columns, try to convert to categorical and then to numeric
                try:
                    final_states[col] = pd.Categorical(final_states[col]).codes
                except:
                    # If conversion fails, drop the column
                    feature_cols.remove(col)
                    print(f"Dropped non-numeric column: {col}")
        
        # Update feature_cols to only include the remaining columns
        feature_cols = [col for col in feature_cols if col in final_states.columns]
        print(f"Selected {len(feature_cols)} features for training")
        
        # Check for and remove columns with constant values (no information)
        constant_cols = [col for col in feature_cols if final_states[col].nunique() <= 1]
        if constant_cols:
            print(f"Removing {len(constant_cols)} constant columns: {constant_cols}")
            feature_cols = [col for col in feature_cols if col not in constant_cols]
        
        # Prepare features and target
        X = final_states[feature_cols].values
        y = final_states['outcome_binary'].values
        
        # Check for NaN or inf values in features
        if np.isnan(X).any() or np.isinf(X).any():
            print("WARNING: NaN or Inf values found in features, replacing with zeros")
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Use RobustScaler instead of StandardScaler to handle outliers better
        scaler = RobustScaler()
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
        
        # Print data shapes
        print(f"Training data shape: {X_train.shape}")
        print(f"Testing data shape: {X_test.shape}")
        
        # Create DataLoaders
        batch_size = min(32, len(X_train))  # Adjust batch size for small datasets
        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        return train_loader, test_loader, X.shape[1], scaler
        
    except Exception as e:
        print(f"Error loading data: {e}")
        import traceback
        traceback.print_exc()
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
    criterion = nn.BCELoss()  # Binary Cross Entropy for binary classification
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
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
        batch_count = 0
        
        for batch_x, batch_y in train_loader:
            batch_count += 1
            
            # Check for NaN values
            if torch.isnan(batch_x).any() or torch.isnan(batch_y).any():
                print(f"WARNING: NaN values detected in batch {batch_count}")
                # Replace NaNs with zeros
                batch_x = torch.nan_to_num(batch_x, nan=0.0)
                batch_y = torch.nan_to_num(batch_y, nan=0.0)
            
            # Add time dimension for LSTM (if not already present)
            if len(batch_x.shape) == 2:
                batch_x = batch_x.unsqueeze(1)
                
            # Forward pass
            outputs = model(batch_x)
            
            # Ensure outputs and targets are valid
            outputs = torch.clamp(outputs, 0.001, 0.999)  # Avoid exact 0 or 1
            
            loss = criterion(outputs, batch_y)
            
            # Check if loss is valid
            if torch.isnan(loss):
                print(f"WARNING: NaN loss detected in batch {batch_count}")
                continue
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Apply gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
        
        # Calculate average training loss
        avg_train_loss = train_loss / max(1, batch_count)
        train_losses.append(avg_train_loss)
        
        # Evaluation phase
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        batch_count = 0
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_count += 1
                
                # Check for NaN values
                if torch.isnan(batch_x).any() or torch.isnan(batch_y).any():
                    print(f"WARNING: NaN values detected in test batch {batch_count}")
                    batch_x = torch.nan_to_num(batch_x, nan=0.0)
                    batch_y = torch.nan_to_num(batch_y, nan=0.0)
                
                # Add time dimension for LSTM (if not already present)
                if len(batch_x.shape) == 2:
                    batch_x = batch_x.unsqueeze(1)
                    
                # Forward pass
                outputs = model(batch_x)
                
                # Ensure outputs are between 0 and 1
                outputs = torch.clamp(outputs, 0.001, 0.999)  # Avoid exact 0 or 1
                
                loss = criterion(outputs, batch_y)
                
                # Skip if loss is NaN
                if torch.isnan(loss):
                    print(f"WARNING: NaN test loss detected in batch {batch_count}")
                    continue
                    
                test_loss += loss.item()
                
                # Calculate accuracy
                predicted = (outputs > 0.5).float()
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        
        # Calculate average test loss and accuracy
        avg_test_loss = test_loss / max(1, batch_count)
        test_losses.append(avg_test_loss)
        
        accuracy = 100 * correct / max(1, total)
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
    model.load_state_dict(torch.load('models/best_battle_predictor.pt'))
    
    # Plot results
    plot_training_results(train_losses, test_losses, test_accuracies)
    
    return model


def plot_training_results(train_losses, test_losses, test_accuracies):
    """Plot training metrics with enhanced visualization"""
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(15, 10))
    
    # Loss plot with enhanced styling
    plt.subplot(2, 1, 1)
    plt.plot(epochs, train_losses, 'b-', linewidth=2, label='Training Loss')
    plt.plot(epochs, test_losses, 'r-', linewidth=2, label='Validation Loss')
    plt.title('Training and Validation Loss', fontsize=16)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add loss values to the plot
    if not np.isnan(train_losses[-1]):
        plt.annotate(f'Final train loss: {train_losses[-1]:.4f}', 
                    xy=(epochs[-1], train_losses[-1]), 
                    xytext=(epochs[-1]-3, train_losses[-1]+0.01),
                    fontsize=10, fontweight='bold')
    if not np.isnan(test_losses[-1]):
        plt.annotate(f'Final val loss: {test_losses[-1]:.4f}', 
                    xy=(epochs[-1], test_losses[-1]), 
                    xytext=(epochs[-1]-3, test_losses[-1]+0.01),
                    fontsize=10, fontweight='bold')
    
    # Accuracy plot with enhanced styling
    plt.subplot(2, 1, 2)
    plt.plot(epochs, test_accuracies, 'g-', linewi