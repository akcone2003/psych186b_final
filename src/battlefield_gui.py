"""
Battlefield Strategy GUI

A graphical user interface for the Enhanced Battlefield Strategy Simulation system.
Makes it easy to run simulations, train models, and use the battle advisor.
"""
import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import threading
import copy
import queue
import torch
import random
import math

# Add parent directory to path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try importing from both module styles to be flexible
try:
    # Try importing directly (if files are in the root directory)
    from battlefield_env import run_simulation, BattlefieldEnv, UnitType, ActionType, TERRAIN_TYPES, WEATHER_CONDITIONS
    from lstm_model import main as train_model, load_model, predict_battle_outcome
    from battle_strategy import get_optimal_actions, get_optimal_positioning, generate_battle_heatmap, visualize_battle_heatmap
    # Import visualization if available
    try:
        from battlefield_visuals import show_latest_battlefield
    except ImportError:
        # Define a fallback function if the module is not available
        def show_latest_battlefield(file_path="data/battle_data.csv"):
            print("battlefield_visuals module not found. Cannot show battlefield visualization.")
except ImportError:
    # Fall back to src/ directory structure (original setup)
    from src.battlefield_env import run_simulation, BattlefieldEnv, UnitType, ActionType, TERRAIN_TYPES, WEATHER_CONDITIONS
    from src.lstm_model import main as train_model, load_model, predict_battle_outcome
    from src.battle_strategy import get_optimal_actions, get_optimal_positioning, generate_battle_heatmap, visualize_battle_heatmap
    # Import visualization if available
    try:
        from src.battlefield_visuals import show_latest_battlefield
    except ImportError:
        # Define a fallback function if the module is not available
        def show_latest_battlefield(file_path="data/battle_data.csv"):
            print("battlefield_visuals module not found. Cannot show battlefield visualization.")


class BattlefieldGUI(tk.Tk):
    """Main GUI application for the Enhanced Battlefield Strategy system"""
    
    def __init__(self):
        super().__init__()
        
        # Configure main window
        self.title("Enhanced Battlefield Strategy System")
        self.geometry("1000x700")
        self.minsize(800, 600)
        
        # Set up styles
        self.style = ttk.Style()
        self.style.configure("TNotebook", tabposition='n')
        self.style.configure("Header.TLabel", font=('Arial', 16, 'bold'))
        self.style.configure("SubHeader.TLabel", font=('Arial', 12, 'bold'))
        
        # Create status bar FIRST
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        self.status_bar = ttk.Label(self, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Create notebook (tabbed interface)
        self.notebook = ttk.Notebook(self)
        
        # Create tabs
        self.simulation_tab = ttk.Frame(self.notebook)
        self.training_tab = ttk.Frame(self.notebook)
        self.advisor_tab = ttk.Frame(self.notebook)
        self.visualization_tab = ttk.Frame(self.notebook)
        
        # Add tabs to notebook
        self.notebook.add(self.simulation_tab, text="Run Simulation")
        self.notebook.add(self.training_tab, text="Train Model")
        self.notebook.add(self.advisor_tab, text="Battle Advisor")
        self.notebook.add(self.visualization_tab, text="Visualizations")
        
        # Pack notebook
        self.notebook.pack(expand=True, fill='both', padx=10, pady=10)
        
        # Set up each tab
        self._setup_simulation_tab()
        self._setup_training_tab()
        self._setup_advisor_tab()
        self._setup_visualization_tab()
        
        # Initialize model
        self.model = None
        self.battlefield_env = None
        
        # Now load model if it exists (AFTER status_var is created)
        self.load_model_if_exists()
        
        # Queue for thread-safe updates
        self.queue = queue.Queue()
        self.after(100, self.process_queue)
        
        # Check for required directories
        self._ensure_directories_exist()

    def show_real_time_battlefield(self):
        """Show the latest battlefield grid-based visualization with terrain & weather from the simulation."""
        try:
            print("⚔️ Generating the latest battlefield grid visualization...")
            
            # Create a fresh battlefield environment
            env = BattlefieldEnv()
            env.reset()  # Initialize a new battlefield
            
            # Clear the current figure and create a completely fresh one
            self.viz_fig.clear()
            ax = self.viz_fig.add_subplot(111)
            
            # Generate a grid for rendering
            grid = np.zeros((env.grid_size, env.grid_size, 3))
            
            # Add terrain colors
            for x in range(env.grid_size):
                for y in range(env.grid_size):
                    terrain = env.get_terrain_at([x, y])
                    
                    # Set terrain color based on type
                    if terrain == "Plains":
                        grid[x, y] = [0.7, 0.9, 0.5]  # Light green
                    elif terrain == "Mountains":
                        grid[x, y] = [0.7, 0.7, 0.7]  # Gray
                    elif terrain == "Forest":
                        grid[x, y] = [0.2, 0.6, 0.2]  # Dark green
                    elif terrain == "Urban":
                        grid[x, y] = [0.8, 0.8, 0.8]  # Light gray
                    elif terrain == "Desert":
                        grid[x, y] = [0.9, 0.8, 0.5]  # Sand color
                    else:
                        grid[x, y] = [1.0, 1.0, 1.0]  # White (default)
            
            # Add obstacles (black)
            for obs_pos in env.obstacles:
                grid[obs_pos[0], obs_pos[1]] = [0, 0, 0]
            
            # Display the grid in our figure
            ax.imshow(grid)
            
            # Plot friendly units
            for i, unit in enumerate(env.friendly_units):
                if unit.is_alive():
                    x, y = unit.position
                    
                    # Marker depends on type
                    if unit.unit_type == UnitType.INFANTRY:
                        marker = "o"  # Circle
                    elif unit.unit_type == UnitType.ARMORED:
                        marker = "s"  # Square
                    elif unit.unit_type == UnitType.AERIAL:
                        marker = "^"  # Triangle
                    else:
                        marker = "X"  # X
                        
                    # Use a simple version of the unit type name
                    unit_type_str = str(unit.unit_type).split('.')[-1]
                    
                    # Plot on our axis
                    ax.scatter(y, x, marker=marker, s=100, color='blue', label=unit_type_str)
                    
                    # Add health bar
                    health_pct = unit.hp / unit.max_hp
                    health_bar_width = 0.8
                    health_bar_height = 0.1
                    health_rect = patches.Rectangle(
                        (y - health_bar_width/2, x + 0.3),
                        health_bar_width * health_pct,
                        health_bar_height,
                        color='green'
                    )
                    ax.add_patch(health_rect)
            
            # Plot enemy units
            for i, enemy in enumerate(env.enemies):
                if enemy.is_alive():
                    x, y = enemy.position
                    
                    # Marker depends on type
                    if enemy.unit_type == UnitType.INFANTRY:
                        marker = "o"  # Circle
                    elif enemy.unit_type == UnitType.ARMORED:
                        marker = "s"  # Square
                    elif enemy.unit_type == UnitType.AERIAL:
                        marker = "^"  # Triangle
                    elif enemy.unit_type == UnitType.ARTILLERY:
                        marker = "*"  # Star
                    elif enemy.unit_type == UnitType.STEALTH:
                        marker = "P"  # Pentagon
                    else:
                        marker = "X"  # X
                        
                    # Use a simple version of the unit type name
                    enemy_type_str = str(enemy.unit_type).split('.')[-1]
                    
                    # Plot on our axis
                    ax.scatter(y, x, marker=marker, s=100, color='red', label=enemy_type_str)
                    
                    # Add health bar
                    health_pct = enemy.hp / enemy.max_hp
                    health_bar_width = 0.8
                    health_bar_height = 0.1
                    health_rect = patches.Rectangle(
                        (y - health_bar_width/2, x + 0.3),
                        health_bar_width * health_pct,
                        health_bar_height,
                        color='red'
                    )
                    ax.add_patch(health_rect)
            
            # Add title
            ax.set_title(f"Battlefield - Terrain: {env.current_terrain}, Weather: {env.current_weather}")
            
            # Add grid
            ax.grid(True, color='black', linestyle='-', linewidth=0.5, alpha=0.2)
            
            # Update the canvas
            self.viz_canvas.draw()
            self.status_var.set("Battlefield visualization generated")
            
        except Exception as e:
            error_msg = f"Could not display battlefield: {str(e)}"
            self.status_var.set(error_msg)
            messagebox.showerror("Visualization Error", error_msg)
            import traceback
            traceback.print_exc()

    def get_optimal_actions(self):
        """Wrapper to call get_optimal_actions from battle_strategy.py"""
        if self.model is None:
            messagebox.showerror("Error", "No model loaded")
            return
        
        unit_positions = {
            'infantry': [self.inf_x_var.get(), self.inf_y_var.get()],
            'tank': [self.tank_x_var.get(), self.tank_y_var.get()],
            'drone': [self.drone_x_var.get(), self.drone_y_var.get()]
        }
        enemy_position = [self.enemy_x_var.get(), self.enemy_y_var.get()]
        
        best_actions, win_prob = get_optimal_actions(self.model, unit_positions, enemy_position)
        
        action_names = ['Move', 'Attack', 'Defend', 'Retreat', 'Support']
        result_text = f"🎲 Optimal Strategy:\n\n"
        result_text += f"Infantry: {action_names[best_actions[0]]}\n"
        result_text += f"Tank: {action_names[best_actions[1]]}\n"
        result_text += f"Drone: {action_names[best_actions[2]]}\n\n"
        result_text += f"Victory probability: {win_prob:.2%}"
        
        self.advisor_results.delete(1.0, tk.END)
        self.advisor_results.insert(tk.END, result_text)

    def get_optimal_positioning(self):
        """Wrapper to call get_optimal_positioning from battle_strategy.py"""
        if self.model is None:
            messagebox.showerror("Error", "Model not loaded")
            return
            
        enemy_position = [self.enemy_x_var.get(), self.enemy_y_var.get()]
        
        # Update status
        self.status_var.set("Calculating optimal positions...")
        self.update()
        
        best_positions, win_prob = get_optimal_positioning(self.model, enemy_position)
        
        result_text = f"🎯 Optimal Unit Positions:\n\n"
        result_text += f"Infantry: {best_positions['infantry']}\n"
        result_text += f"Tank: {best_positions['tank']}\n"
        result_text += f"Drone: {best_positions['drone']}\n\n"
        result_text += f"Victory probability: {win_prob:.2%}"
        
        self.advisor_results.delete(1.0, tk.END)
        self.advisor_results.insert(tk.END, result_text)
        
        # Update the input fields with the optimal positions
        self.inf_x_var.set(best_positions['infantry'][0])
        self.inf_y_var.set(best_positions['infantry'][1])
        self.tank_x_var.set(best_positions['tank'][0])
        self.tank_y_var.set(best_positions['tank'][1])
        self.drone_x_var.set(best_positions['drone'][0])
        self.drone_y_var.set(best_positions['drone'][1])
        
        # Reset status
        self.status_var.set("Ready")

    def show_heatmap(self):
        """Wrapper to show heatmap using generate_battle_heatmap"""
        
        # Ensure these attributes exist before trying to access them
        if not hasattr(self, 'viz_enemy_x_var') or not hasattr(self, 'viz_enemy_y_var'):
            print("⚠️ Error: vizualization variables not initialized!")
            return

        # Create a new battlefield environment for visualization
        env = BattlefieldEnv()
        
        enemy_position = [self.viz_enemy_x_var.get(), self.viz_enemy_y_var.get()]
        print(f"📍 Generating heatmap for enemy at position: {enemy_position}")
        
        # Update status
        old_status = self.status_var.get()
        self.status_var.set("Generating battlefield heatmap...")
        self.update()
        
        try:
            heatmap = generate_battle_heatmap(self.model, enemy_position)

            # Plot heatmap
            self.viz_fig.clear()
            ax = self.viz_fig.add_subplot(111)
            im = ax.imshow(heatmap, cmap='viridis', interpolation='nearest', origin='lower')
            self.viz_fig.colorbar(im, ax=ax, label='Victory Probability')

            # Mark enemy position
            ax.scatter(enemy_position[1], enemy_position[0], color='red', s=100, marker='*', label='Enemy')

            # Add terrain information if available
            if hasattr(env, 'terrain_data') and env.terrain_data is not None:
                ax.set_title(f'Victory Probability Heatmap\nTerrain: {env._get_dominant_terrain()}, Weather: {env.current_weather}')
            else:
                ax.set_title('Infantry Victory Probability Heatmap')
                
            ax.legend()
            self.viz_canvas.draw()
            
            # Reset status
            self.status_var.set("Heatmap generated successfully")
        except Exception as e:
            self.status_var.set(f"Error generating heatmap: {e}")
            messagebox.showerror("Heatmap Error", f"Error generating heatmap: {e}")
    
    def _ensure_directories_exist(self):
        """Make sure all required directories exist"""
        directories = ['data', 'models', 'visualizations', 'src']
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def load_model_if_exists(self):
        """Try to load the model if it exists"""
        if os.path.exists("models/best_battle_predictor.pt"):
            try:
                self.model = load_model("models/best_battle_predictor.pt")
                self.status_var.set("Model loaded successfully")
            except Exception as e:
                self.status_var.set(f"Error loading model: {e}")
        else:
            self.status_var.set("No model found. Please train a model first.")
    
    def process_queue(self):
        """Process messages from the queue (thread-safe updates)"""
        try:
            while True:
                message = self.queue.get_nowait()
                if message[0] == "status":
                    self.status_var.set(message[1])
                elif message[0] == "progress":
                    if hasattr(self, 'progress_var'):
                        self.progress_var.set(message[1])
                elif message[0] == "log":
                    if hasattr(self, 'simulation_log'):
                        self.simulation_log.config(state=tk.NORMAL)
                        self.simulation_log.insert(tk.END, message[1] + "\n")
                        self.simulation_log.see(tk.END)
                        self.simulation_log.config(state=tk.DISABLED)
                self.queue.task_done()
        except queue.Empty:
            pass
        self.after(100, self.process_queue)
    
    def _setup_simulation_tab(self):
        """Set up the simulation tab with enhanced options"""
        # Header
        header = ttk.Label(self.simulation_tab, text="Run Enhanced Battle Simulations", style="Header.TLabel")
        header.grid(row=0, column=0, columnspan=2, pady=10, sticky='w')
        
        # Frame for parameters
        params_frame = ttk.LabelFrame(self.simulation_tab, text="Simulation Parameters")
        params_frame.grid(row=1, column=0, padx=10, pady=10, sticky='nw')
        
        # Number of battles
        ttk.Label(params_frame, text="Number of Battles:").grid(row=0, column=0, padx=5, pady=5, sticky='w')
        self.num_battles_var = tk.IntVar(value=10)
        ttk.Spinbox(params_frame, from_=1, to=100, textvariable=self.num_battles_var, width=5).grid(row=0, column=1, padx=5, pady=5, sticky='w')
        
        # Max steps per battle
        ttk.Label(params_frame, text="Max Steps per Battle:").grid(row=1, column=0, padx=5, pady=5, sticky='w')
        self.max_steps_var = tk.IntVar(value=100)
        ttk.Spinbox(params_frame, from_=10, to=100, textvariable=self.max_steps_var, width=5).grid(row=1, column=1, padx=5, pady=5, sticky='w')
        
        # Max enemies
        ttk.Label(params_frame, text="Max Enemies:").grid(row=2, column=0, padx=5, pady=5, sticky='w')
        self.max_enemies_var = tk.IntVar(value=3)
        ttk.Spinbox(params_frame, from_=1, to=5, textvariable=self.max_enemies_var, width=5).grid(row=2, column=1, padx=5, pady=5, sticky='w')
        
        # Output file
        ttk.Label(params_frame, text="Output File:").grid(row=3, column=0, padx=5, pady=5, sticky='w')
        self.output_file_var = tk.StringVar(value="data/battle_data.csv")
        ttk.Entry(params_frame, textvariable=self.output_file_var, width=30).grid(row=3, column=1, padx=5, pady=5, sticky='w')
        
        # Render checkbox
        self.render_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(params_frame, text="Render Final States", variable=self.render_var).grid(row=4, column=0, columnspan=2, padx=5, pady=5, sticky='w')
        
        # Run button
        ttk.Button(params_frame, text="Run Simulation", command=self.run_simulation).grid(row=5, column=0, columnspan=2, padx=5, pady=10)
        
        # Progress bar
        ttk.Label(params_frame, text="Progress:").grid(row=6, column=0, padx=5, pady=5, sticky='w')
        self.progress_var = tk.DoubleVar()
        ttk.Progressbar(params_frame, variable=self.progress_var, maximum=100).grid(row=6, column=1, padx=5, pady=5, sticky='we')
        
        # Simulation log
        log_frame = ttk.LabelFrame(self.simulation_tab, text="Simulation Log")
        log_frame.grid(row=1, column=1, rowspan=3, padx=10, pady=10, sticky='nsew')
        
        # Make the log frame expandable
        self.simulation_tab.columnconfigure(1, weight=1)
        self.simulation_tab.rowconfigure(3, weight=1)
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        
        # Scrollable text widget for log
        self.simulation_log = tk.Text(log_frame, wrap=tk.WORD, state=tk.DISABLED, width=50, height=20)
        scrollbar = ttk.Scrollbar(log_frame, command=self.simulation_log.yview)
        self.simulation_log.configure(yscrollcommand=scrollbar.set)
        
        self.simulation_log.grid(row=0, column=0, sticky='nsew')
        scrollbar.grid(row=0, column=1, sticky='ns')
        
    def _setup_training_tab(self):
        """Set up the training tab"""
        # Header
        header = ttk.Label(self.training_tab, text="Train LSTM Model", style="Header.TLabel")
        header.grid(row=0, column=0, columnspan=2, pady=10, sticky='w')
        
        # Frame for parameters
        params_frame = ttk.LabelFrame(self.training_tab, text="Training Parameters")
        params_frame.grid(row=1, column=0, padx=10, pady=10, sticky='nw')
        
        # Data file
        ttk.Label(params_frame, text="Data File:").grid(row=0, column=0, padx=5, pady=5, sticky='w')
        self.data_file_var = tk.StringVar(value="data/battle_data.csv")
        file_entry = ttk.Entry(params_frame, textvariable=self.data_file_var, width=30)
        file_entry.grid(row=0, column=1, padx=5, pady=5, sticky='w')
        ttk.Button(params_frame, text="Browse", command=lambda: self.browse_file(self.data_file_var)).grid(row=0, column=2, padx=5, pady=5)
        
        # Number of epochs
        ttk.Label(params_frame, text="Number of Epochs:").grid(row=1, column=0, padx=5, pady=5, sticky='w')
        self.epochs_var = tk.IntVar(value=50)
        ttk.Spinbox(params_frame, from_=10, to=500, textvariable=self.epochs_var, width=5).grid(row=1, column=1, padx=5, pady=5, sticky='w')
        
        # Learning rate
        ttk.Label(params_frame, text="Learning Rate:").grid(row=2, column=0, padx=5, pady=5, sticky='w')
        self.lr_var = tk.DoubleVar(value=0.001)
        ttk.Spinbox(params_frame, from_=0.0001, to=0.1, increment=0.0001, textvariable=self.lr_var, width=10).grid(row=2, column=1, padx=5, pady=5, sticky='w')
        
        # Batch size
        ttk.Label(params_frame, text="Batch Size:").grid(row=3, column=0, padx=5, pady=5, sticky='w')
        self.batch_size_var = tk.IntVar(value=32)
        ttk.Spinbox(params_frame, from_=8, to=128, textvariable=self.batch_size_var, width=5).grid(row=3, column=1, padx=5, pady=5, sticky='w')
        
        # Hidden size
        ttk.Label(params_frame, text="Hidden Size:").grid(row=4, column=0, padx=5, pady=5, sticky='w')
        self.hidden_size_var = tk.IntVar(value=64)
        ttk.Spinbox(params_frame, from_=16, to=256, textvariable=self.hidden_size_var, width=5).grid(row=4, column=1, padx=5, pady=5, sticky='w')
        
        # Model save path
        ttk.Label(params_frame, text="Model Save Path:").grid(row=5, column=0, padx=5, pady=5, sticky='w')
        self.model_path_var = tk.StringVar(value="models/best_battle_predictor.pt")
        ttk.Entry(params_frame, textvariable=self.model_path_var, width=30).grid(row=5, column=1, padx=5, pady=5, sticky='w')
        
        # Train button
        ttk.Button(params_frame, text="Train Model", command=self.train_model).grid(row=6, column=0, columnspan=3, padx=5, pady=10)
        
        # Training results frame
        results_frame = ttk.LabelFrame(self.training_tab, text="Training Results")
        results_frame.grid(row=1, column=1, rowspan=2, padx=10, pady=10, sticky='nsew')
        
        # Make the results frame expandable
        self.training_tab.columnconfigure(1, weight=1)
        self.training_tab.rowconfigure(2, weight=1)
        
        # Placeholder for training curve
        self.training_fig = plt.Figure(figsize=(6, 4), dpi=100)
        self.training_canvas = FigureCanvasTkAgg(self.training_fig, results_frame)
        self.training_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=1)
        
    def _setup_advisor_tab(self):
        """Set up the battle advisor tab"""
        # Header
        header = ttk.Label(self.advisor_tab, text="Battle Strategy Advisor", style="Header.TLabel")
        header.grid(row=0, column=0, columnspan=2, pady=10, sticky='w')
        
        # Frame for unit positions
        positions_frame = ttk.LabelFrame(self.advisor_tab, text="Unit Positions")
        positions_frame.grid(row=1, column=0, padx=10, pady=10, sticky='nw')
        
        # Infantry position
        ttk.Label(positions_frame, text="Infantry Position:").grid(row=0, column=0, padx=5, pady=5, sticky='w')
        self.inf_x_var = tk.IntVar(value=3)
        self.inf_y_var = tk.IntVar(value=4)
        ttk.Label(positions_frame, text="X:").grid(row=0, column=1, padx=5, pady=5, sticky='w')
        ttk.Spinbox(positions_frame, from_=0, to=9, textvariable=self.inf_x_var, width=3).grid(row=0, column=2, padx=5, pady=5, sticky='w')
        ttk.Label(positions_frame, text="Y:").grid(row=0, column=3, padx=5, pady=5, sticky='w')
        ttk.Spinbox(positions_frame, from_=0, to=9, textvariable=self.inf_y_var, width=3).grid(row=0, column=4, padx=5, pady=5, sticky='w')
        
        # Tank position
        ttk.Label(positions_frame, text="Tank Position:").grid(row=1, column=0, padx=5, pady=5, sticky='w')
        self.tank_x_var = tk.IntVar(value=5)
        self.tank_y_var = tk.IntVar(value=2)
        ttk.Label(positions_frame, text="X:").grid(row=1, column=1, padx=5, pady=5, sticky='w')
        ttk.Spinbox(positions_frame, from_=0, to=9, textvariable=self.tank_x_var, width=3).grid(row=1, column=2, padx=5, pady=5, sticky='w')
        ttk.Label(positions_frame, text="Y:").grid(row=1, column=3, padx=5, pady=5, sticky='w')
        ttk.Spinbox(positions_frame, from_=0, to=9, textvariable=self.tank_y_var, width=3).grid(row=1, column=4, padx=5, pady=5, sticky='w')
        
        # Drone position
        ttk.Label(positions_frame, text="Drone Position:").grid(row=2, column=0, padx=5, pady=5, sticky='w')
        self.drone_x_var = tk.IntVar(value=1)
        self.drone_y_var = tk.IntVar(value=7)
        ttk.Label(positions_frame, text="X:").grid(row=2, column=1, padx=5, pady=5, sticky='w')
        ttk.Spinbox(positions_frame, from_=0, to=9, textvariable=self.drone_x_var, width=3).grid(row=2, column=2, padx=5, pady=5, sticky='w')
        ttk.Label(positions_frame, text="Y:").grid(row=2, column=3, padx=5, pady=5, sticky='w')
        ttk.Spinbox(positions_frame, from_=0, to=9, textvariable=self.drone_y_var, width=3).grid(row=2, column=4, padx=5, pady=5, sticky='w')
        
        # Enemy position
        ttk.Label(positions_frame, text="Enemy Position:").grid(row=3, column=0, padx=5, pady=5, sticky='w')
        self.enemy_x_var = tk.IntVar(value=8)
        self.enemy_y_var = tk.IntVar(value=8)
        ttk.Label(positions_frame, text="X:").grid(row=3, column=1, padx=5, pady=5, sticky='w')
        ttk.Spinbox(positions_frame, from_=0, to=9, textvariable=self.enemy_x_var, width=3).grid(row=3, column=2, padx=5, pady=5, sticky='w')
        ttk.Label(positions_frame, text="Y:").grid(row=3, column=3, padx=5, pady=5, sticky='w')
        ttk.Spinbox(positions_frame, from_=0, to=9, textvariable=self.enemy_y_var, width=3).grid(row=3, column=4, padx=5, pady=5, sticky='w')
        
        # Terrain and weather dropdowns (new in enhanced version)
        ttk.Label(positions_frame, text="Terrain:").grid(row=4, column=0, padx=5, pady=5, sticky='w')
        self.terrain_var = tk.StringVar(value="Plains")
        ttk.Combobox(positions_frame, textvariable=self.terrain_var, values=list(TERRAIN_TYPES.keys()), 
                   state="readonly", width=10).grid(row=4, column=1, columnspan=2, padx=5, pady=5, sticky='w')
        
        ttk.Label(positions_frame, text="Weather:").grid(row=5, column=0, padx=5, pady=5, sticky='w')
        self.weather_var = tk.StringVar(value="Clear")
        ttk.Combobox(positions_frame, textvariable=self.weather_var, values=list(WEATHER_CONDITIONS.keys()), 
                    state="readonly", width=10).grid(row=5, column=1, columnspan=2, padx=5, pady=5, sticky='w')
        
        # Action buttons
        actions_frame = ttk.LabelFrame(self.advisor_tab, text="Advisory Actions")
        actions_frame.grid(row=2, column=0, padx=10, pady=10, sticky='nw')
        
        ttk.Button(actions_frame, text="Get Optimal Actions", command=self.get_optimal_actions).grid(row=0, column=0, padx=5, pady=5, sticky='w')
        ttk.Button(actions_frame, text="Get Optimal Positions", command=self.get_optimal_positioning).grid(row=1, column=0, padx=5, pady=5, sticky='w')
        ttk.Button(actions_frame, text="Generate Heatmap", command=self.show_heatmap).grid(row=2, column=0, padx=5, pady=5, sticky='w')
        
        # Create simulation button (new in enhanced version)
        ttk.Button(actions_frame, text="Run Interactive Battle", command=self.run_interactive_battle).grid(row=3, column=0, padx=5, pady=5, sticky='w')
        
        # Results frame
        results_frame = ttk.LabelFrame(self.advisor_tab, text="Advisory Results")
        results_frame.grid(row=1, column=1, rowspan=3, padx=10, pady=10, sticky='nsew')
        
        # Make the results frame expandable
        self.advisor_tab.columnconfigure(1, weight=1)
        self.advisor_tab.rowconfigure(3, weight=1)
        
        # Results text
        self.advisor_results = tk.Text(results_frame, wrap=tk.WORD, width=50, height=20)
        scrollbar = ttk.Scrollbar(results_frame, command=self.advisor_results.yview)
        self.advisor_results.configure(yscrollcommand=scrollbar.set)
        
        self.advisor_results.grid(row=0, column=0, sticky='nsew')
        scrollbar.grid(row=0, column=1, sticky='ns')
        
        # Make text widget expand
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)
    
    def _setup_visualization_tab(self):
        """Set up the visualization tab with enhanced options"""
        # Header
        header = ttk.Label(self.visualization_tab, text="Battlefield Visualizations", style="Header.TLabel")
        header.grid(row=0, column=0, columnspan=2, pady=10, sticky='w')

        # Frame for visualization controls
        controls_frame = ttk.LabelFrame(self.visualization_tab, text="Visualization Controls")
        controls_frame.grid(row=1, column=0, padx=10, pady=10, sticky='nw')

        # Initialize visualization variables
        self.viz_enemy_x_var = tk.IntVar(value=5)
        self.viz_enemy_y_var = tk.IntVar(value=5)

        # Enemy position input
        ttk.Label(controls_frame, text="Enemy Position (X, Y):").grid(row=0, column=0, padx=5, pady=5, sticky='w')
        ttk.Spinbox(controls_frame, from_=0, to=9, textvariable=self.viz_enemy_x_var, width=3).grid(row=0, column=1, padx=5, pady=5, sticky='w')
        ttk.Spinbox(controls_frame, from_=0, to=9, textvariable=self.viz_enemy_y_var, width=3).grid(row=0, column=2, padx=5, pady=5, sticky='w')

        # Buttons for both heatmap and real-time battlefield visualization
        ttk.Button(controls_frame, text="Show Heatmap", command=self.show_heatmap).grid(row=1, column=0, columnspan=3, padx=5, pady=10)
        ttk.Button(controls_frame, text="Show Real-Time Battlefield", command=self.show_real_time_battlefield).grid(row=2, column=0, columnspan=3, padx=5, pady=10)
        
        # Visualization type selector
        ttk.Label(controls_frame, text="Terrain Type:").grid(row=3, column=0, padx=5, pady=5, sticky='w')
        self.viz_terrain_var = tk.StringVar(value="Plains")
        terrain_combo = ttk.Combobox(controls_frame, textvariable=self.viz_terrain_var, 
                                   values=list(TERRAIN_TYPES.keys()), state="readonly")
        terrain_combo.grid(row=3, column=1, columnspan=2, padx=5, pady=5, sticky='w')
        
        ttk.Label(controls_frame, text="Weather Condition:").grid(row=4, column=0, padx=5, pady=5, sticky='w')
        self.viz_weather_var = tk.StringVar(value="Clear")
        weather_combo = ttk.Combobox(controls_frame, textvariable=self.viz_weather_var, 
                                    values=list(WEATHER_CONDITIONS.keys()), state="readonly")
        weather_combo.grid(row=4, column=1, columnspan=2, padx=5, pady=5, sticky='w')
        
        # Create a button to generate the environmental impact visualization
        ttk.Button(controls_frame, text="Analyze Environment Impact", 
                  command=self.show_environment_impact).grid(row=5, column=0, columnspan=3, padx=5, pady=10)

        # Placeholder for visualization
        self.viz_fig = plt.Figure(figsize=(6, 6), dpi=100)
        self.viz_canvas = FigureCanvasTkAgg(self.viz_fig, self.visualization_tab)
        self.viz_canvas.get_tk_widget().grid(row=1, column=1, rowspan=3, padx=5, pady=5, sticky='nsew')
        
        # Make the visualization column expandable
        self.visualization_tab.columnconfigure(1, weight=1)
        self.visualization_tab.rowconfigure(1, weight=1)
    
    def update_viz_controls(self, event=None):
        """Update visualization controls based on selected type"""
        # Not needed anymore - we've restructured the visualization tab
        pass
    
    def browse_file(self, var):
        """Open file browser and update the variable"""
        filename = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if filename:
            var.set(filename)
    
    def run_simulation(self):
        """Run battle simulation with enhanced parameters"""
        # Check if directories exist
        os.makedirs('data', exist_ok=True)
        
        # Get parameters
        num_battles = self.num_battles_var.get()
        max_steps = self.max_steps_var.get()
        max_enemies = self.max_enemies_var.get() if hasattr(self, 'max_enemies_var') else 3
        render_final = self.render_var.get() if hasattr(self, 'render_var') else True
        
        # Clear log
        self.simulation_log.config(state=tk.NORMAL)
        self.simulation_log.delete(1.0, tk.END)
        self.simulation_log.config(state=tk.DISABLED)
        
        # Create a custom print function that writes to our log
        def custom_print(*args, **kwargs):
            message = " ".join(map(str, args))
            self.queue.put(("log", message))
        
        # Start simulation in a separate thread
        def run_sim_thread():
            try:
                # Redirect standard output to our custom print function
                import builtins
                original_print = builtins.print
                builtins.print = custom_print
                
                # Set up progress updates
                def progress_callback(battle, total):
                    progress = (battle / total) * 100
                    self.queue.put(("progress", progress))
                
                # Create a folder for battle images
                os.makedirs('visualizations/battles', exist_ok=True)
                
                # Run simulation - use the enhanced battlefield but don't render during simulation
                self.queue.put(("status", f"Running enhanced simulation with {num_battles} battles..."))
                
                try:
                    from battlefield_env import run_simulation
                except ImportError:
                    from src.battlefield_env import run_simulation
                    
                # Don't render during simulation to avoid threading issues
                results = run_simulation(
                    num_battles=num_battles, 
                    max_steps=max_steps, 
                    render_final=False,  # Important: don't render in thread
                    max_enemies=max_enemies
                )
                
                self.queue.put(("status", "Simulation complete!"))
                self.queue.put(("progress", 100))
                
                # Report battle results
                self.queue.put(("log", f"\nSimulation results:"))
                if isinstance(results, dict):
                    for outcome, count in results.items():
                        percentage = (count / num_battles) * 100
                        self.queue.put(("log", f"{outcome.capitalize()}: {count} ({percentage:.1f}%)"))
                
                messagebox.showinfo("Success", f"Simulation completed with {num_battles} battles")
                
                # Restore standard output
                builtins.print = original_print
                
            except Exception as e:
                self.queue.put(("status", f"Error: {e}"))
                self.queue.put(("log", f"ERROR: {e}"))
                messagebox.showerror("Error", f"An error occurred: {e}")
        
        # Start thread
        sim_thread = threading.Thread(target=run_sim_thread)
        sim_thread.daemon = True
        sim_thread.start()
    
    def run_interactive_battle(self):
        """Run an interactive battle with the current battlefield parameters"""
        try:
            # Initialize battlefield environment
            env = BattlefieldEnv()
            self.battlefield_env = env
            
            # Set custom positions if needed
            # This would need to modify the battlefield environment directly
            
            # Just render the battlefield for now
            env.render()
            
            # Store for future reference
            self.battlefield_env = env
            
            # Add the battlefield view to the results
            self.advisor_results.delete(1.0, tk.END)
            self.advisor_results.insert(tk.END, "Interactive battle initialized!\n\n")
            self.advisor_results.insert(tk.END, f"Terrain: {env.current_terrain}\n")
            self.advisor_results.insert(tk.END, f"Weather: {env.current_weather}\n\n")
            self.advisor_results.insert(tk.END, f"Friendly Units: {len(env.friendly_units)}\n")
            self.advisor_results.insert(tk.END, f"Enemy Units: {len(env.enemies)}\n")
            
            # Display unit types
            self.advisor_results.insert(tk.END, "\nFriendly Unit Types:\n")
            for i, unit in enumerate(env.friendly_units):
                self.advisor_results.insert(tk.END, f"Unit {i+1}: {unit.unit_type.name}\n")
            
            self.advisor_results.insert(tk.END, "\nEnemy Unit Types:\n")
            for i, enemy in enumerate(env.enemies):
                self.advisor_results.insert(tk.END, f"Enemy {i+1}: {enemy.unit_type.name}\n")
                
        except Exception as e:
            self.status_var.set(f"Error running interactive battle: {e}")
            messagebox.showerror("Error", f"Could not initialize battlefield: {e}")
    
    def show_environment_impact(self):
        """Show the impact of different terrain and weather on battle outcomes"""
        # Create a new window for the visualization
        impact_window = tk.Toplevel(self)
        impact_window.title("Environment Impact Analysis")
        impact_window.geometry("800x600")
        
        # Create figure
        fig = plt.Figure(figsize=(10, 8))
        
        # Create subplots for terrain and weather
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        
        # Example data (in a real implementation, this would come from analysis of battle data)
        terrain_types = list(TERRAIN_TYPES.keys())
        terrain_win_rates = [0.65, 0.45, 0.55, 0.60, 0.50]  # Example win rates for different terrains
        
        weather_types = list(WEATHER_CONDITIONS.keys())
        weather_win_rates = [0.60, 0.45, 0.50, 0.40, 0.55]  # Example win rates for different weather
        
        # Plot terrain impact
        terrain_bars = ax1.bar(terrain_types, terrain_win_rates, color='green', alpha=0.7)
        ax1.set_ylim(0, 1.0)
        ax1.set_ylabel("Win Rate")
        ax1.set_title("Impact of Terrain on Battle Outcomes")
        ax1.grid(axis='y', linestyle='--', alpha=0.3)
        
        # Add value labels
        for bar, rate in zip(terrain_bars, terrain_win_rates):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f"{rate:.0%}", ha='center', va='bottom')
        
        # Plot weather impact
        weather_bars = ax2.bar(weather_types, weather_win_rates, color='blue', alpha=0.7)
        ax2.set_ylim(0, 1.0)
        ax2.set_ylabel("Win Rate")
        ax2.set_title("Impact of Weather on Battle Outcomes")
        ax2.grid(axis='y', linestyle='--', alpha=0.3)
        
        # Add value labels
        for bar, rate in zip(weather_bars, weather_win_rates):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f"{rate:.0%}", ha='center', va='bottom')
        
        # Set layout
        fig.tight_layout()
        
        # Embed in window
        canvas = FigureCanvasTkAgg(fig, master=impact_window)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    
    def train_model(self):
        """Trigger the model training from lstm_model.py"""
        # Check if data file exists
        data_file = self.data_file_var.get()
        if not os.path.exists(data_file):
            messagebox.showerror("Error", f"Data file {data_file} not found. Please run simulation first.")
            return
        
        # Get parameters
        model_path = self.model_path_var.get()
        
        # Ensure directories exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        os.makedirs('visualizations', exist_ok=True)
        
        # Clear previous plot
        self.training_fig.clear()
        
        # Start training in a separate thread
        def train_thread():
            try:
                self.queue.put(("status", "Starting model training..."))
                self.queue.put(("progress", 10))
                
                # Simply call the main function from lstm_model
                # This avoids duplicating any training logic
                train_model()
                
                # After training completes, load the trained model
                self.model = load_model("models/best_battle_predictor.pt")
                
                # Update final status
                self.queue.put(("status", "Training complete! Model loaded successfully."))
                self.queue.put(("progress", 100))
                
                # Update the training plot if possible
                try:
                    if os.path.exists("visualizations/battle_predictor_training.png"):
                        from PIL import Image, ImageTk
                        img = Image.open("visualizations/battle_predictor_training.png")
                        img = img.resize((600, 400), Image.LANCZOS)
                        photo = ImageTk.PhotoImage(img)
                        
                        # Display in the training figure
                        self.training_fig.clear()
                        ax = self.training_fig.add_subplot(111)
                        ax.imshow(np.asarray(img))
                        ax.axis('off')
                        self.training_canvas.draw()
                except Exception as img_e:
                    print(f"Could not display training plot: {img_e}")
                
                messagebox.showinfo("Success", "Training completed successfully!")
                
            except Exception as e:
                self.queue.put(("status", f"Error: {e}"))
                messagebox.showerror("Error", f"An error occurred: {e}")
                import traceback
                traceback.print_exc()
        
        # Start thread
        train_thread = threading.Thread(target=train_thread)
        train_thread.daemon = True
        train_thread.start()