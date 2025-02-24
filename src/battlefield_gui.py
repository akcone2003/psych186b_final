"""
Battlefield Strategy GUI

A graphical user interface for the Battlefield Strategy Simulation system.
Makes it easy to run simulations, train models, and use the battle advisor.
"""
import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import threading
import queue
import torch

# Add parent directory to path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our modules
from src.battlefield_env import run_simulation, BattlefieldEnv
from src.lstm_model import main as train_model, load_model, predict_battle_outcome
from src.battle_strategy import get_optimal_actions, get_optimal_positioning, generate_battle_heatmap, visualize_battle_heatmap

class BattlefieldGUI(tk.Tk):
    """Main GUI application for the Battlefield Strategy system"""
    
    def __init__(self):
        super().__init__()
        
        # Configure main window
        self.title("Battlefield Strategy System")
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
        
        # Now load model if it exists (AFTER status_var is created)
        self.load_model_if_exists()
        
        # Queue for thread-safe updates
        self.queue = queue.Queue()
        self.after(100, self.process_queue)
        
        # Check for required directories
        self._ensure_directories_exist()

    def get_optimal_actions(self):
        """Wrapper to call get_optimal_actions from battle_strategy.py"""
        if self.model is None:
            messagebox.showerror("Error", "No model not loaded")
            return
        
        unit_positions = {
            'infantry': [self.inf_x_var.get(), self.inf_y_var.get()],
            'tank': [self.tank_x_var.get(), self.tank_y_var.get()],
            'drone': [self.drone_x_var.get(), self.drone_y_var.get()]
        }
        enemy_position = [self.enemy_x_var.get(), self.enemy_y_var.get()]
        
        from src.battle_strategy import get_optimal_actions
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
        
        from src.battle_strategy import get_optimal_positioning
        best_positions, win_prob = get_optimal_positioning(self.model, enemy_position)
        
        result_text = f"🎯 Optimal Unit Positions:\n\n"
        result_text += f"Infantry: {best_positions['infantry']}\n"
        result_text += f"Tank: {best_positions['tank']}\n"
        result_text += f"Drone: {best_positions['drone']}\n\n"
        result_text += f"Victory probability: {win_prob:.2%}"
        
        self.advisor_results.delete(1.0, tk.END)
        self.advisor_results.insert(tk.END, result_text)

    def show_heatmap(self):
        """Wrapper to show heatmap using generate_battle_heatmap"""
        if self.model is None:
            messagebox.showerror("Error", "Model not loaded")
            return
            
        enemy_position = [self.viz_enemy_x_var.get(), self.viz_enemy_y_var.get()]
        unit_positions = {'tank': [5, 5], 'drone': [7, 3]}
        
        from src.battle_strategy import generate_battle_heatmap
        heatmap = generate_battle_heatmap(self.model, enemy_position, unit_positions)
        
        # Plot heatmap
        self.viz_fig.clear()
        ax = self.viz_fig.add_subplot(111)
        im = ax.imshow(heatmap, cmap='viridis', interpolation='nearest', origin='lower')
        self.viz_fig.colorbar(im, ax=ax, label='Victory Probability')
        
        # Mark positions
        ax.scatter(enemy_position[1], enemy_position[0], color='red', s=100, marker='*', label='Enemy')
        ax.scatter(unit_positions['tank'][1], unit_positions['tank'][0], color='blue', s=80, marker='s', label='Tank')
        ax.scatter(unit_positions['drone'][1], unit_positions['drone'][0], color='green', s=60, marker='^', label='Drone')
        
        ax.set_title('Infantry Victory Probability Heatmap')
        ax.legend()
        
        self.viz_canvas.draw()
    
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
        """Set up the simulation tab"""
        # Header
        header = ttk.Label(self.simulation_tab, text="Run Battle Simulations", style="Header.TLabel")
        header.grid(row=0, column=0, columnspan=2, pady=10, sticky='w')
        
        # Frame for parameters
        params_frame = ttk.LabelFrame(self.simulation_tab, text="Simulation Parameters")
        params_frame.grid(row=1, column=0, padx=10, pady=10, sticky='nw')
        
        # Number of battles
        ttk.Label(params_frame, text="Number of Battles:").grid(row=0, column=0, padx=5, pady=5, sticky='w')
        self.num_battles_var = tk.IntVar(value=20)
        ttk.Spinbox(params_frame, from_=1, to=100, textvariable=self.num_battles_var, width=5).grid(row=0, column=1, padx=5, pady=5, sticky='w')
        
        # Max steps per battle
        ttk.Label(params_frame, text="Max Steps per Battle:").grid(row=1, column=0, padx=5, pady=5, sticky='w')
        self.max_steps_var = tk.IntVar(value=30)
        ttk.Spinbox(params_frame, from_=10, to=100, textvariable=self.max_steps_var, width=5).grid(row=1, column=1, padx=5, pady=5, sticky='w')
        
        # Output file
        ttk.Label(params_frame, text="Output File:").grid(row=2, column=0, padx=5, pady=5, sticky='w')
        self.output_file_var = tk.StringVar(value="data/battle_data.csv")
        ttk.Entry(params_frame, textvariable=self.output_file_var, width=30).grid(row=2, column=1, padx=5, pady=5, sticky='w')
        
        # Run button
        ttk.Button(params_frame, text="Run Simulation", command=self.run_simulation).grid(row=3, column=0, columnspan=2, padx=5, pady=10)
        
        # Progress bar
        ttk.Label(params_frame, text="Progress:").grid(row=4, column=0, padx=5, pady=5, sticky='w')
        self.progress_var = tk.DoubleVar()
        ttk.Progressbar(params_frame, variable=self.progress_var, maximum=100).grid(row=4, column=1, padx=5, pady=5, sticky='we')
        
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
        
        # Action buttons
        actions_frame = ttk.LabelFrame(self.advisor_tab, text="Advisory Actions")
        actions_frame.grid(row=2, column=0, padx=10, pady=10, sticky='nw')
        
        #----------------------------DEBUG------------------------
        ttk.Button(actions_frame, text="Get Optimal Actions", command=self.get_optimal_actions).grid(row=0, column=0, padx=5, pady=5, sticky='w')
        ttk.Button(actions_frame, text="Get Optimal Positions", command=self.get_optimal_positioning).grid(row=1, column=0, padx=5, pady=5, sticky='w')
        ttk.Button(actions_frame, text="Generate Heatmap", command=self.show_heatmap).grid(row=2, column=0, padx=5, pady=5, sticky='w')
        
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
        """Set up the visualization tab"""
        # Header
        header = ttk.Label(self.visualization_tab, text="Battlefield Visualizations", style="Header.TLabel")
        header.grid(row=0, column=0, columnspan=2, pady=10, sticky='w')
        
        # Frame for visualization controls
        controls_frame = ttk.LabelFrame(self.visualization_tab, text="Visualization Controls")
        controls_frame.grid(row=1, column=0, padx=10, pady=10, sticky='nw')
        
        # Visualization type
        ttk.Label(controls_frame, text="Visualization Type:").grid(row=0, column=0, padx=5, pady=5, sticky='w')
        self.viz_type_var = tk.StringVar(value="Heatmap")
        viz_type_combo = ttk.Combobox(controls_frame, textvariable=self.viz_type_var, values=["Heatmap", "Battle Simulation", "Training Metrics"])
        viz_type_combo.grid(row=0, column=1, padx=5, pady=5, sticky='w')
        viz_type_combo.bind("<<ComboboxSelected>>", self.update_viz_controls)
        
        # Placeholder for dynamic controls
        self.viz_controls_frame = ttk.Frame(controls_frame)
        self.viz_controls_frame.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky='nsew')
        
        # Initial controls setup
        self.update_viz_controls()
        
        # Visualization frame
        viz_frame = ttk.LabelFrame(self.visualization_tab, text="Visualization")
        viz_frame.grid(row=1, column=1, rowspan=2, padx=10, pady=10, sticky='nsew')
        
        # Make the viz frame expandable
        self.visualization_tab.columnconfigure(1, weight=1)
        self.visualization_tab.rowconfigure(2, weight=1)
        viz_frame.columnconfigure(0, weight=1)
        viz_frame.rowconfigure(0, weight=1)
        
        # Placeholder for visualization
        self.viz_fig = plt.Figure(figsize=(6, 6), dpi=100)
        self.viz_canvas = FigureCanvasTkAgg(self.viz_fig, viz_frame)
        self.viz_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=1)
    
    def update_viz_controls(self, event=None):
        """Update visualization controls based on selected type"""
        # Clear existing controls
        for widget in self.viz_controls_frame.winfo_children():
            widget.destroy()
        
        viz_type = self.viz_type_var.get()
        
        if viz_type == "Heatmap":
            # Enemy position
            ttk.Label(self.viz_controls_frame, text="Enemy Position:").grid(row=0, column=0, padx=5, pady=5, sticky='w')
            self.viz_enemy_x_var = tk.IntVar(value=8)
            self.viz_enemy_y_var = tk.IntVar(value=8)
            ttk.Label(self.viz_controls_frame, text="X:").grid(row=0, column=1, padx=5, pady=5, sticky='w')
            ttk.Spinbox(self.viz_controls_frame, from_=0, to=9, textvariable=self.viz_enemy_x_var, width=3).grid(row=0, column=2, padx=5, pady=5, sticky='w')
            ttk.Label(self.viz_controls_frame, text="Y:").grid(row=0, column=3, padx=5, pady=5, sticky='w')
            ttk.Spinbox(self.viz_controls_frame, from_=0, to=9, textvariable=self.viz_enemy_y_var, width=3).grid(row=0, column=4, padx=5, pady=5, sticky='w')
            
            # Generate button
            ttk.Button(self.viz_controls_frame, text="Generate Heatmap", command=self.show_heatmap).grid(row=1, column=0, columnspan=5, padx=5, pady=10)
            
        elif viz_type == "Battle Simulation":
            # Button to load simulation data
            ttk.Button(self.viz_controls_frame, text="Load Simulation Data", command=self.load_simulation_data).grid(row=0, column=0, padx=5, pady=5, sticky='w')
            
            # Battle selector
            ttk.Label(self.viz_controls_frame, text="Battle:").grid(row=1, column=0, padx=5, pady=5, sticky='w')
            self.battle_selector_var = tk.IntVar(value=1)
            self.battle_selector = ttk.Spinbox(self.viz_controls_frame, from_=1, to=1, textvariable=self.battle_selector_var, width=3, state="disabled")
            self.battle_selector.grid(row=1, column=1, padx=5, pady=5, sticky='w')
            
            # Play, pause, step buttons
            ttk.Button(self.viz_controls_frame, text="Play", command=self.play_battle, state="disabled").grid(row=2, column=0, padx=5, pady=5, sticky='w')
            ttk.Button(self.viz_controls_frame, text="Pause", command=self.pause_battle, state="disabled").grid(row=2, column=1, padx=5, pady=5, sticky='w')
            ttk.Button(self.viz_controls_frame, text="Step", command=self.step_battle, state="disabled").grid(row=2, column=2, padx=5, pady=5, sticky='w')
            
        elif viz_type == "Training Metrics":
            # Load training metrics
            ttk.Button(self.viz_controls_frame, text="Load Training Metrics", command=self.load_training_metrics).grid(row=0, column=0, padx=5, pady=5, sticky='w')
            
            # Metric selector
            ttk.Label(self.viz_controls_frame, text="Metric:").grid(row=1, column=0, padx=5, pady=5, sticky='w')
            self.metric_var = tk.StringVar(value="Loss")
            ttk.Combobox(self.viz_controls_frame, textvariable=self.metric_var, values=["Loss", "Accuracy"], state="readonly").grid(row=1, column=1, padx=5, pady=5, sticky='w')
            
            # Update button
            ttk.Button(self.viz_controls_frame, text="Update Visualization", command=self.update_metrics_viz).grid(row=2, column=0, columnspan=2, padx=5, pady=10)
    
    def browse_file(self, var):
        """Open file browser and update the variable"""
        filename = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if filename:
            var.set(filename)
    
    def run_simulation(self):
        """Run battle simulation"""
        # Check if directories exist
        os.makedirs('data', exist_ok=True)
        
        # Get parameters
        num_battles = self.num_battles_var.get()
        max_steps = self.max_steps_var.get()
        output_file = self.output_file_var.get()
        
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
                
                # Run simulation
                self.queue.put(("status", f"Running simulation with {num_battles} battles..."))
                for battle in range(num_battles):
                    # Create environment
                    env = BattlefieldEnv()
                    
                    # Reset environment
                    obs = env.reset()
                    step_count = 0
                    done = False
                    
                    # Run battle
                    while not done and step_count < max_steps:
                        # Random actions
                        actions = np.array([
                            np.random.randint(0, 4),
                            np.random.randint(0, 4),
                            np.random.randint(0, 4)
                        ])
                        
                        # Execute action
                        obs, reward, done, info = env.step(actions)
                        step_count += 1
                    
                    # Determine result
                    result = 1 if 'result' in info and info['result'] == 'victory' else 0
                    
                    # Save battle log
                    env.save_battle_log(result)
                    
                    # Update progress
                    progress_callback(battle + 1, num_battles)
                    
                    self.queue.put(("log", f"Battle {battle+1}/{num_battles} completed - Outcome: {'Victory' if result else 'Defeat'}"))
                
                # Restore standard output
                builtins.print = original_print
                
                self.queue.put(("status", "Simulation complete!"))
                self.queue.put(("progress", 100))
                messagebox.showinfo("Success", f"Simulation completed with {num_battles} battles")
                
            except Exception as e:
                self.queue.put(("status", f"Error: {e}"))
                self.queue.put(("log", f"ERROR: {e}"))
                messagebox.showerror("Error", f"An error occurred: {e}")
        
        # Start thread
        sim_thread = threading.Thread(target=run_sim_thread)
        sim_thread.daemon = True
        sim_thread.start()
    
    def train_model(self):
        """Train the LSTM model"""
        # Check if data file exists
        data_file = self.data_file_var.get()
        if not os.path.exists(data_file):
            messagebox.showerror("Error", f"Data file {data_file} not found. Please run simulation first.")
            return
        
        # Get parameters
        epochs = self.epochs_var.get()
        model_path = self.model_path_var.get()
        
        # Ensure directories exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        os.makedirs('visualizations', exist_ok=True)
        
        # Clear previous plot
        self.training_fig.clear()
        
        # Start training in a separate thread
        def train_thread():
            try:
                self.queue.put(("status", "Loading and preprocessing data..."))
                
                # Create a custom callback for progress updates
                class TrainingCallback:
                    def __init__(self, queue, epochs):
                        self.queue = queue
                        self.epochs = epochs
                        self.train_losses = []
                        self.test_losses = []
                        self.test_accuracies = []
                    
                    def on_epoch_end(self, epoch, train_loss, test_loss, accuracy, lr):
                        # Update progress
                        progress = ((epoch + 1) / self.epochs) * 100
                        self.queue.put(("progress", progress))
                        
                        # Update status
                        self.queue.put(("status", f"Epoch {epoch+1}/{self.epochs} | Loss: {train_loss:.4f} | Accuracy: {accuracy:.2f}%"))
                        
                        # Store metrics
                        self.train_losses.append(train_loss)
                        self.test_losses.append(test_loss)
                        self.test_accuracies.append(accuracy)
                        
                        # Update plot (every 5 epochs to reduce UI updates)
                        if (epoch + 1) % 5 == 0 or epoch == self.epochs - 1:
                            self.update_plot()
                    
                    def update_plot(self):
                        self.queue.put(("status", "Updating training plot..."))
                        
                        # Clear figure
                        self.queue.put(("update_plot", (self.train_losses, self.test_losses, self.test_accuracies)))
                
                # Create callback
                callback = TrainingCallback(self.queue, epochs)
                
                # Load and preprocess data
                from src.lstm_model import load_and_preprocess_data, LSTMBattlePredictor
                import torch.optim as optim
                import torch.nn as nn
                
                train_loader, test_loader, input_size, _ = load_and_preprocess_data(data_file)
                
                # Create model
                model = LSTMBattlePredictor(
                    input_size=input_size,
                    hidden_size=self.hidden_size_var.get(),
                    num_layers=2,
                    dropout=0.3
                )
                
                # Train model
                criterion = nn.BCELoss()
                optimizer = optim.Adam(model.parameters(), lr=self.lr_var.get())
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
                
                # Training loop
                best_accuracy = 0
                patience = 10
                no_improvement = 0
                
                for epoch in range(epochs):
                    # Training phase
                    model.train()
                    train_loss = 0
                    
                    for batch_x, batch_y in train_loader:
                        # Add time dimension if needed
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
                    
                    # Evaluation phase
                    model.eval()
                    test_loss = 0
                    correct = 0
                    total = 0
                    
                    with torch.no_grad():
                        for batch_x, batch_y in test_loader:
                            # Add time dimension if needed
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
                    accuracy = 100 * correct / total
                    
                    # Update learning rate
                    scheduler.step(avg_test_loss)
                    
                    # Call callback
                    callback.on_epoch_end(epoch, avg_train_loss, avg_test_loss, accuracy, 
                                         optimizer.param_groups[0]['lr'])
                    
                    # Early stopping check
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        # Save the best model
                        torch.save(model.state_dict(), model_path)
                        self.queue.put(("status", f"Model improved, saved checkpoint (Accuracy: {accuracy:.2f}%)"))
                        no_improvement = 0
                    else:
                        no_improvement += 1
                        if no_improvement >= patience:
                            self.queue.put(("status", f"Early stopping triggered after {epoch+1} epochs"))
                            break
                
                # Update final status
                self.queue.put(("status", f"Training complete! Best accuracy: {best_accuracy:.2f}%"))
                self.queue.put(("progress", 100))
                
                # Load the best model
                self.model = LSTMBattlePredictor(input_size=input_size, hidden_size=self.hidden_size_var.get(), num_layers=2)
                self.model.load_state_dict(torch.load(model_path, weights_only=True))
                self.model.eval()
                
                messagebox.showinfo("Success", f"Training completed successfully!\nBest accuracy: {best_accuracy:.2f}%")
                
            except Exception as e:
                self.queue.put(("status", f"Error: {e}"))
                messagebox.showerror("Error", f"An error occurred: {e}")
                import traceback
                traceback.print_exc()
        
        # Start thread
        train_thread = threading.Thread(target=train_thread)
        train_thread.daemon = True
        train_thread.start()
        
        # Set up plot update handler
        def update_plot_handler():
            try:
                while True:
                    message = self.queue.get_nowait()
                    if message[0] == "update_plot":
                        train_losses, test_losses, test_accuracies = message[1]
                        
                        # Update plot
                        self.training_fig.clear()
                        
                        # Create subplots
                        ax1 = self.training_fig.add_subplot(121)
                        ax2 = self.training_fig.add_subplot(122)
                        
                        # Plot losses
                        epochs = range(1, len(train_losses) + 1)
                        ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
                        ax1.plot(epochs, test_losses, 'r-', label='Validation Loss')
                        ax1.set_title('Training and Validation Loss')
                        ax1.set_xlabel('Epochs')
                        ax1.set_ylabel('Loss')
                        ax1.legend()
                        ax1.grid(True, alpha=0.3)
                        
                        # Plot accuracy
                        ax2.plot(epochs, test_accuracies, 'g-', label='Validation Accuracy')
                        ax2.set_title('Validation Accuracy')
                        ax2.set_xlabel('Epochs')
                        ax2.set_ylabel('Accuracy (%)')
                        ax2.legend()
                        ax2.grid(True, alpha=0.3)
                        
                        # Update canvas
                        self.training_fig.tight_layout()
                        self.training_canvas.draw()
                    
                    self.queue.task_done()
            except queue.Empty:
                pass
            self.after(500, update_plot_handler)
        
        # Start plot update handler
        update_plot_handler()