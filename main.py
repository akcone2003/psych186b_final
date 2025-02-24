"""
Main module for Battlefield Strategy Simulation

This script provides multiple interfaces for running the battlefield simulation:
1. Command-line interface for running simulations, training models, and using the battle advisor
2. GUI mode for interactive use
3. Full pipeline mode that runs all steps in sequence

When run directly without arguments, it will launch the GUI by default.
"""
import argparse
import os
import sys
import time

from src.battlefield_env import run_simulation
from src.lstm_model import main as train_model
from src.battle_strategy import battle_advisor


def main():
    """Main function that parses command line arguments and runs the appropriate module"""
    parser = argparse.ArgumentParser(description='Battlefield Strategy Simulation')
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Simulation parser
    sim_parser = subparsers.add_parser('simulate', help='Run battle simulations')
    sim_parser.add_argument('--battles', type=int, default=50, 
                            help='Number of battles to simulate')
    sim_parser.add_argument('--steps', type=int, default=30,
                            help='Maximum steps per battle')
    
    # Training parser
    train_parser = subparsers.add_parser('train', help='Train LSTM model')
    train_parser.add_argument('--data', type=str, default='data/battle_data.csv',
                             help='Data file to use for training')
    
    # Advisor parser
    advisor_parser = subparsers.add_parser('advisor', help='Run battle strategy advisor')
    advisor_parser.add_argument('--model', type=str, default='models/best_battle_predictor.pt',
                                help='Path to trained model')
    
    # GUI parser
    gui_parser = subparsers.add_parser('gui', help='Launch the GUI interface')
    
    # Pipeline parser
    pipeline_parser = subparsers.add_parser('pipeline', help='Run full pipeline (simulate, train, advise)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Make sure directories exist
    ensure_directories_exist()
    
    # Execute appropriate command
    if args.command == 'simulate':
        print(f"Running simulation with {args.battles} battles, {args.steps} max steps each")
        run_simulation(num_battles=args.battles, timesteps=args.steps)
        
    elif args.command == 'train':
        # Check if data file exists
        if not os.path.exists(args.data):
            print(f"Error: Data file '{args.data}' not found")
            print("Please run 'python main.py simulate' first to generate battle data")
            return
            
        print(f"Training model using data from {args.data}")
        train_model()
        
    elif args.command == 'advisor':
        # Check if model file exists
        if not os.path.exists(args.model):
            print(f"Error: Model file '{args.model}' not found")
            print("Please run 'python main.py train' first to train a model")
            return
            
        print(f"Starting battle advisor with model {args.model}")
        battle_advisor(model_path=args.model)
    
    elif args.command == 'gui':
        launch_gui()
    
    elif args.command == 'pipeline':
        run_full_pipeline()
        
    else:
        # If no command given, launch GUI by default
        launch_gui()


def ensure_directories_exist():
    """Make sure all required directories exist"""
    directories = ['data', 'models', 'visualizations', 'src']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def launch_gui():
    """Launch the GUI interface"""
    try:
        from src.battlefield_gui import BattlefieldGUI
        import tkinter as tk
        
        print("Launching GUI interface...")
        app = BattlefieldGUI()
        app.mainloop()
    except ImportError as e:
        print(f"Error: {e}")
        print("GUI requirements not met. Please install tkinter and matplotlib.")
        print("You can run in command-line mode with: python main.py simulate")


def run_full_pipeline():
    """Run the full pipeline: simulation, training, and advisor"""
    print("=" * 50)
    print("STARTING FULL BATTLEFIELD SIMULATION PIPELINE")
    print("=" * 50)
    
    # Step 1: Run simulation to generate data
    print("\n[STEP 1/3] Running battle simulations to generate training data...")
    num_battles = 20
    max_steps = 25
    print(f"Generating {num_battles} battles with max {max_steps} steps each")
    run_simulation(num_battles=num_battles, timesteps=max_steps)
    
    # Wait a moment to ensure data is saved
    time.sleep(2)
    
    # Step 2: Train model on generated data
    print("\n[STEP 2/3] Training LSTM model on battle data...")
    if not os.path.exists("data/battle_data.csv"):
        emergency_files = [f for f in os.listdir('data') if f.startswith('battle_data_emergency_') and f.endswith('.csv')]
        if emergency_files:
            latest_file = sorted(emergency_files, reverse=True)[0]
            print(f"Using emergency backup file: {latest_file}")
            # Rename to standard filename
            os.rename(f"data/{latest_file}", "data/battle_data.csv")
        else:
            print("Error: No battle data found. Please check simulation output.")
            return

    train_model()
    
    # Wait a moment to ensure model is saved
    time.sleep(2)
    
    # Step 3: Start battle advisor
    print("\n[STEP 3/3] Starting battle strategy advisor...")
    if os.path.exists("models/best_battle_predictor.pt"):
        print("Model loaded successfully! Starting advisor...")
        battle_advisor(model_path="models/best_battle_predictor.pt")
    else:
        print("Error: Trained model not found. Please check training output.")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # If arguments provided, run normal command processing
        main()
    else:
        # If no arguments, launch the GUI by default
        launch_gui()