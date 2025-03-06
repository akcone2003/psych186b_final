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
    from battlefield_env import run_simulation, BattlefieldEnv, ATTACK_RANGES, UnitType, ActionType, TERRAIN_TYPES, WEATHER_CONDITIONS
    from lstm_model import main as train_model, load_model, predict_battle_outcome
    from battle_strategy import get_optimal_actions, get_optimal_positioning, generate_battle_heatmap, visualize_battle_heatmap
    from self_play import SelfPlaySimulation
    # Import visualization if available
    try:
        from battlefield_visuals import show_latest_battlefield
    except ImportError:
        # Define a fallback function if the module is not available
        def show_latest_battlefield(file_path="data/battle_data.csv"):
            print("battlefield_visuals module not found. Cannot show battlefield visualization.")
except ImportError:
    # Fall back to src/ directory structure (original setup)
    from src.battlefield_env import run_simulation, BattlefieldEnv, ATTACK_RANGES, UnitType, ActionType, TERRAIN_TYPES, WEATHER_CONDITIONS
    from src.lstm_model import main as train_model, load_model, predict_battle_outcome
    from src.battle_strategy import get_optimal_actions, get_optimal_positioning, generate_battle_heatmap, visualize_battle_heatmap
    from src.self_play import SelfPlaySimulation
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
        self.self_play_tab = ttk.Frame(self.notebook)

        
        # Add tabs to notebook
        self.notebook.add(self.
        simulation_tab, text="Run Simulation")
        self.notebook.add(self.training_tab, text="Train Model")
        self.notebook.add(self.advisor_tab, text="Battle Advisor")
        self.notebook.add(self.visualization_tab, text="Visualizations")
        self.notebook.add(self.self_play_tab, text="Self-Play")
        
        # Pack notebook
        self.notebook.pack(expand=True, fill='both', padx=10, pady=10)
        
        # Set up each tab
        self._setup_simulation_tab()
        self._setup_training_tab()
        self._setup_advisor_tab()
        self._setup_visualization_tab()
        self._setup_self_play_tab()
        
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

    def run_self_play(self):
        """Run a self-play simulation where the model controls both sides"""
        # Check if model is loaded
        if self.model is None:
            messagebox.showerror("Error", "Model not loaded. Please train or load a model first.")
            return
        
        # Get parameters from GUI
        try:
            num_battles = self.self_play_battles_var.get()
            max_steps = self.self_play_steps_var.get()
            save_viz = self.self_play_save_viz_var.get()
            max_enemies = self.self_play_max_enemies_var.get()
            
            # Get terrain type (handle "Random")
            terrain = self.self_play_terrain_var.get()
            if terrain == "Random":
                terrain = None
                
            # Get weather type (handle "Random")
            weather = self.self_play_weather_var.get()
            if weather == "Random":
                weather = None
                
            enable_artillery = self.self_play_artillery_var.get()
            enable_stealth = self.self_play_stealth_var.get()
            
        except:
            # If variables not defined yet, use defaults
            num_battles = 1
            max_steps = 50
            save_viz = True
            max_enemies = 3
            terrain = None
            weather = None
            enable_artillery = False
            enable_stealth = False
        
        # Update status
        self.status_var.set("Starting self-play simulation...")
        self.update()
        
        # Function to run in a separate thread
        def run_simulation():
            try:
                # Create simulator with all parameters
                simulator = SelfPlaySimulation(
                    model_path="models/best_battle_predictor.pt",
                    max_steps=max_steps,
                    save_visualizations=save_viz,
                    max_enemies=max_enemies,
                    terrain_type=terrain,
                    weather_type=weather,
                    enable_artillery=enable_artillery,
                    enable_stealth=enable_stealth
                )
                
                # Queue update
                self.queue.put(("status", "Self-play simulation in progress..."))
                
                # Run simulation
                if num_battles == 1:
                    # Single battle with detailed logging
                    result = simulator.run_self_play_battle(log_steps=True, render_battle=True)
                    
                    # Format result text
                    result_text = f"Self-Play Battle Results\n\n"
                    result_text += f"Outcome: {result['result']}\n"
                    result_text += f"Blue Units Remaining: {result['blue_remaining']}/{result['blue_total']}\n"
                    result_text += f"Red Units Remaining: {result['red_remaining']}/{result['red_total']}\n"
                    
                    # Add terrain and weather info
                    if hasattr(simulator.env, 'current_terrain'):
                        result_text += f"\nTerrain: {simulator.env.current_terrain}\n"
                    if hasattr(simulator.env, 'current_weather'):
                        result_text += f"Weather: {simulator.env.current_weather}\n"
                    
                    # Add unit type info if available
                    if hasattr(simulator.env, 'friendly_units') and hasattr(simulator.env, 'enemies'):
                        blue_types = [unit.unit_type.name for unit in simulator.env.friendly_units]
                        red_types = [enemy.unit_type.name for enemy in simulator.env.enemies]
                        
                        result_text += f"\nBlue Unit Types: {', '.join(blue_types)}\n"
                        result_text += f"Red Unit Types: {', '.join(red_types)}\n"
                    
                    # Queue results for display
                    self.queue.put(("self_play_result", result_text))
                    
                else:
                    # Multiple battles
                    stats = simulator.run_multiple_battles(num_battles=num_battles, log_individual=False)
                    
                    # Calculate statistics
                    win_rate_blue = stats['wins_blue'] / num_battles * 100
                    win_rate_red = stats['wins_red'] / num_battles * 100
                    draw_rate = stats['draws'] / num_battles * 100
                    
                    avg_steps = sum([sum(steps)/len(steps) for steps in stats['avg_steps']]) / len(stats['avg_steps']) if stats['avg_steps'] else 0
                    
                    avg_blue_remaining = sum(stats['blue_units_remaining']) / len(stats['blue_units_remaining']) if stats['blue_units_remaining'] else 0
                    avg_red_remaining = sum(stats['red_units_remaining']) / len(stats['red_units_remaining']) if stats['red_units_remaining'] else 0
                    
                    # Format results text
                    result_text = f"Self-Play Campaign Results ({num_battles} battles)\n\n"
                    result_text += f"Blue Victory Rate: {win_rate_blue:.1f}%\n"
                    result_text += f"Red Victory Rate: {win_rate_red:.1f}%\n"
                    result_text += f"Draw Rate: {draw_rate:.1f}%\n\n"
                    result_text += f"Average Battle Length: {avg_steps:.1f} steps\n"
                    result_text += f"Average Blue Units Remaining: {avg_blue_remaining:.2f}\n"
                    result_text += f"Average Red Units Remaining: {avg_red_remaining:.2f}\n\n"
                    
                    # Add custom parameters info
                    result_text += f"Custom Parameters:\n"
                    result_text += f"- Max Enemies: {max_enemies}\n"
                    result_text += f"- Terrain: {terrain if terrain else 'Random'}\n"
                    result_text += f"- Weather: {weather if weather else 'Random'}\n"
                    result_text += f"- Artillery Units: {'Enabled' if enable_artillery else 'Disabled'}\n"
                    result_text += f"- Stealth Units: {'Enabled' if enable_stealth else 'Disabled'}\n\n"
                    
                    if save_viz:
                        result_text += "Visualizations saved to 'visualizations/self_play/'"
                    
                    # Queue results for display
                    self.queue.put(("self_play_result", result_text))
                
                # Update status when complete
                self.queue.put(("status", "Self-play simulation complete"))
                
            except Exception as e:
                # Log error
                self.queue.put(("status", f"Error in self-play: {str(e)}"))
                self.queue.put(("log", f"Self-play error: {str(e)}"))
                import traceback
                traceback.print_exc()
        
        # Run in a separate thread
        import threading
        thread = threading.Thread(target=run_simulation)
        thread.daemon = True
        thread.start()

    def _setup_self_play_tab(self):
        """Set up the self-play tab with enhanced customization options"""
        # Header
        header = ttk.Label(self.self_play_tab, text="Model Self-Play", style="Header.TLabel")
        header.grid(row=0, column=0, columnspan=2, pady=10, sticky='w')
        
        # Create a notebook for the parameters (tabbed interface for better organization)
        params_notebook = ttk.Notebook(self.self_play_tab)
        params_notebook.grid(row=1, column=0, padx=10, pady=10, sticky='nw')
        
        # Create tabs for different parameter categories
        basic_tab = ttk.Frame(params_notebook)
        advanced_tab = ttk.Frame(params_notebook)
        
        params_notebook.add(basic_tab, text="Basic Settings")
        params_notebook.add(advanced_tab, text="Battlefield Settings")
        
        # ----- BASIC SETTINGS TAB -----
        # Number of battles
        ttk.Label(basic_tab, text="Number of Battles:").grid(row=0, column=0, padx=5, pady=5, sticky='w')
        self.self_play_battles_var = tk.IntVar(value=1)
        ttk.Spinbox(basic_tab, from_=1, to=50, textvariable=self.self_play_battles_var, width=5).grid(row=0, column=1, padx=5, pady=5, sticky='w')
        
        # Max steps per battle
        ttk.Label(basic_tab, text="Max Steps per Battle:").grid(row=1, column=0, padx=5, pady=5, sticky='w')
        self.self_play_steps_var = tk.IntVar(value=50)
        ttk.Spinbox(basic_tab, from_=10, to=200, textvariable=self.self_play_steps_var, width=5).grid(row=1, column=1, padx=5, pady=5, sticky='w')
        
        # Save visualizations checkbox
        self.self_play_save_viz_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(basic_tab, text="Save Final Battle State", variable=self.self_play_save_viz_var).grid(row=2, column=0, columnspan=2, padx=5, pady=5, sticky='w')
        
        # ----- ADVANCED SETTINGS TAB -----
        # Max enemies
        ttk.Label(advanced_tab, text="Maximum Enemies:").grid(row=0, column=0, padx=5, pady=5, sticky='w')
        self.self_play_max_enemies_var = tk.IntVar(value=3)
        ttk.Spinbox(advanced_tab, from_=1, to=5, textvariable=self.self_play_max_enemies_var, width=5).grid(row=0, column=1, padx=5, pady=5, sticky='w')
        
        # Terrain type
        ttk.Label(advanced_tab, text="Terrain Type:").grid(row=1, column=0, padx=5, pady=5, sticky='w')
        self.self_play_terrain_var = tk.StringVar(value="Random")
        terrain_choices = ["Random"] + list(TERRAIN_TYPES.keys())
        ttk.Combobox(advanced_tab, textvariable=self.self_play_terrain_var, values=terrain_choices, 
                state="readonly", width=15).grid(row=1, column=1, padx=5, pady=5, sticky='w')
        
        # Weather type
        ttk.Label(advanced_tab, text="Weather:").grid(row=2, column=0, padx=5, pady=5, sticky='w')
        self.self_play_weather_var = tk.StringVar(value="Random")
        weather_choices = ["Random"] + list(WEATHER_CONDITIONS.keys())
        ttk.Combobox(advanced_tab, textvariable=self.self_play_weather_var, values=weather_choices, 
                state="readonly", width=15).grid(row=2, column=1, padx=5, pady=5, sticky='w')
        
        # Unit type options
        ttk.Label(advanced_tab, text="Special Unit Types:").grid(row=3, column=0, padx=5, pady=5, sticky='w')
        
        # Enable artillery checkbox
        self.self_play_artillery_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(advanced_tab, text="Enable Artillery Units", variable=self.self_play_artillery_var).grid(row=4, column=0, columnspan=2, padx=20, pady=2, sticky='w')
        
        # Enable stealth checkbox
        self.self_play_stealth_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(advanced_tab, text="Enable Stealth Units", variable=self.self_play_stealth_var).grid(row=5, column=0, columnspan=2, padx=20, pady=2, sticky='w')
        
        # Frame for action buttons (below the notebook)
        actions_frame = ttk.Frame(self.self_play_tab)
        actions_frame.grid(row=2, column=0, padx=10, pady=10, sticky='nw')
        
        # Run buttons
        ttk.Button(actions_frame, text="Run Single Battle", command=lambda: self.run_self_play()).grid(row=0, column=0, padx=5, pady=10, sticky='w')
        ttk.Button(actions_frame, text="Run Campaign", command=lambda: self._set_multiple_battles_and_run()).grid(row=0, column=1, padx=5, pady=10, sticky='w')
        
        # Results frame
        results_frame = ttk.LabelFrame(self.self_play_tab, text="Self-Play Results")
        results_frame.grid(row=1, column=1, rowspan=3, padx=10, pady=10, sticky='nsew')
        
        # Make the results frame expandable
        self.self_play_tab.columnconfigure(1, weight=1)
        self.self_play_tab.rowconfigure(3, weight=1)
        
        # Results text
        self.self_play_results = tk.Text(results_frame, wrap=tk.WORD, width=50, height=20)
        scrollbar = ttk.Scrollbar(results_frame, command=self.self_play_results.yview)
        self.self_play_results.configure(yscrollcommand=scrollbar.set)
        
        self.self_play_results.grid(row=0, column=0, sticky='nsew')
        scrollbar.grid(row=0, column=1, sticky='ns')
        
        # Make text widget expand
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)
        
        # Display welcome text
        welcome_text = """Welcome to Model Self-Play

    The self-play feature allows the AI model to control both sides of the battlefield, providing insights into its tactical capabilities and strategic preferences.

    You can run either:
    • A single detailed battle with step-by-step logging
    • A multi-battle campaign with statistical summary

    To begin, make sure you have a trained model, then configure your parameters:

    Basic Settings:
    - Number of battles to run
    - Maximum steps per battle
    - Whether to save visualizations

    Battlefield Settings:
    - Number of enemies (1-5)
    - Terrain type (affects movement and combat)
    - Weather conditions (affects visibility and accuracy)
    - Special unit types (artillery, stealth)

    Once configured, click one of the run buttons.
    """
        self.self_play_results.insert(tk.END, welcome_text)



    def _set_multiple_battles_and_run(self):
        """Helper to set multiple battles and run self-play"""
        # Only set to multiple battles if currently set to 1
        if self.self_play_battles_var.get() == 1:
            self.self_play_battles_var.set(10)  # Default to 10 battles for campaign
        self.run_self_play()

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
                        attack_range = ATTACK_RANGES[UnitType.INFANTRY]
                    elif unit.unit_type == UnitType.ARMORED:
                        marker = "s"  # Square
                        attack_range = ATTACK_RANGES[UnitType.ARMORED]
                    elif unit.unit_type == UnitType.AERIAL:
                        marker = "^"  # Triangle
                        attack_range = ATTACK_RANGES[UnitType.AERIAL]
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

                    # Add attack range indicator as a dotted circle
                    range_circle = patches.Circle(
                        (y, x),  # Center at unit position
                        radius=attack_range,  # Use appropriate attack range
                        fill=False,
                        linestyle='--',
                        linewidth=1,
                        color='blue',
                        alpha=0.5
                    )
                    ax.add_patch(range_circle)
            
            # Plot enemy units
            for i, enemy in enumerate(env.enemies):
                if enemy.is_alive():
                    x, y = enemy.position
                    
                    # Marker depends on type
                    if enemy.unit_type == UnitType.INFANTRY:
                        marker = "o"  # Circle
                        attack_range = ATTACK_RANGES[UnitType.INFANTRY]
                    elif enemy.unit_type == UnitType.ARMORED:
                        marker = "s"  # Square
                        attack_range = ATTACK_RANGES[UnitType.ARMORED]
                    elif enemy.unit_type == UnitType.AERIAL:
                        marker = "^"  # Triangle
                        attack_range = ATTACK_RANGES[UnitType.AERIAL]
                    elif enemy.unit_type == UnitType.ARTILLERY:
                        marker = "*"  # Star
                        attack_range = ATTACK_RANGES[UnitType.ARTILLERY]
                    elif enemy.unit_type == UnitType.STEALTH:
                        marker = "P"  # Pentagon
                        attack_range = ATTACK_RANGES[UnitType.STEALTH]
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

                    # Add attack range indicator as a dotted circle
                    range_circle = patches.Circle(
                        (y, x),  # Center at unit position
                        radius=attack_range,  # Use appropriate attack range
                        fill=False,
                        linestyle='--',
                        linewidth=1,
                        color='red',
                        alpha=0.5
                    )
                    ax.add_patch(range_circle)
            
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
        """Get optimal actions for friendly units with multiple enemies support"""
        if self.model is None:
            messagebox.showerror("Error", "No model loaded")
            return
        
        # Get friendly unit positions
        unit_positions = {
            'infantry': [self.inf_x_var.get(), self.inf_y_var.get()],
            'tank': [self.tank_x_var.get(), self.tank_y_var.get()],
            'drone': [self.drone_x_var.get(), self.drone_y_var.get()]
        }
        
        # Get enemy positions
        enemy_positions = []
        for enemy_pos_vars in self.enemy_positions:
            enemy_positions.append([enemy_pos_vars[0].get(), enemy_pos_vars[1].get()])
        
        # Update status
        self.status_var.set("Calculating optimal actions...")
        self.update()
        
        try:
            # Call get_optimal_actions_multi_enemy for multiple enemies
            if len(enemy_positions) > 1:
                best_actions, win_prob = self.get_optimal_actions_multi_enemy(
                    self.model, unit_positions, enemy_positions)
            else:
                # Use the original function for a single enemy
                best_actions, win_prob = get_optimal_actions(
                    self.model, unit_positions, enemy_positions[0])
            
            action_names = ['Move', 'Attack', 'Defend', 'Retreat', 'Support']
            result_text = f"🎲 Optimal Strategy:\n\n"
            result_text += f"Infantry: {action_names[best_actions[0]]}\n"
            result_text += f"Tank: {action_names[best_actions[1]]}\n"
            result_text += f"Drone: {action_names[best_actions[2]]}\n\n"
            result_text += f"Victory probability: {win_prob:.2%}"
            
            self.advisor_results.delete(1.0, tk.END)
            self.advisor_results.insert(tk.END, result_text)
            
            # Reset status
            self.status_var.set("Ready")
        except Exception as e:
            self.status_var.set(f"Error: {e}")
            messagebox.showerror("Error", f"An error occurred: {e}")

    def get_optimal_actions_multi_enemy(self, model, unit_positions, enemy_positions):
        """
        Determine optimal actions for friendly units against multiple enemies
        
        Parameters:
            model: The trained LSTM model
            unit_positions: Dictionary with position of each friendly unit
            enemy_positions: List of enemy position coordinates
            
        Returns:
            best_actions: List of optimal actions [infantry, tank, drone]
            win_probability: Probability of victory with those actions
        """
        # Initialize with default actions
        best_actions = [0, 0, 0]  # Default move actions
        best_prob = 0
        
        # Get expected input size
        input_size = self._get_model_input_size(model)
        
        # Try all action combinations
        for inf_action in range(5):
            for tank_action in range(5):
                for drone_action in range(5):
                    actions = [inf_action, tank_action, drone_action]
                    
                    # Calculate probabilities against each enemy
                    enemy_probs = []
                    for enemy_position in enemy_positions:
                        # Construct battle state
                        base_battle_state = [
                            unit_positions['infantry'][0], unit_positions['infantry'][1],
                            unit_positions['tank'][0], unit_positions['tank'][1],
                            unit_positions['drone'][0], unit_positions['drone'][1],
                            inf_action, tank_action, drone_action,
                            enemy_position[0], enemy_position[1]
                        ]
                        
                        # Pad to expected input size
                        battle_state = self._pad_feature_vector(base_battle_state, input_size)
                        
                        # Get win probability against this enemy
                        prob = predict_battle_outcome(model, battle_state)
                        enemy_probs.append(prob)
                    
                    # Calculate overall probability (can be min, average, or other combination)
                    # Here we use the average probability across all enemies
                    avg_prob = sum(enemy_probs) / len(enemy_probs)
                    
                    # Update if better
                    if avg_prob > best_prob:
                        best_prob = avg_prob
                        best_actions = [inf_action, tank_action, drone_action]
        
        return best_actions, best_prob

    def get_optimal_positioning(self):
        """Get optimal positions against multiple enemies"""
        if self.model is None:
            messagebox.showerror("Error", "Model not loaded")
            return
            
        # Get enemy positions
        enemy_positions = []
        for enemy_pos_vars in self.enemy_positions:
            enemy_positions.append([enemy_pos_vars[0].get(), enemy_pos_vars[1].get()])
        
        # Update status
        self.status_var.set("Calculating optimal positions...")
        self.update()
        
        try:
            # Call function with or without multiple enemies
            if len(enemy_positions) > 1:
                best_positions, win_prob = self.get_optimal_positions_multi_enemy(
                    self.model, enemy_positions)
            else:
                # Use the original function for a single enemy
                best_positions, win_prob = get_optimal_positioning(
                    self.model, enemy_positions[0])
            
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
        except Exception as e:
            self.status_var.set(f"Error: {e}")
            messagebox.showerror("Error", f"An error occurred: {e}")

    def get_optimal_positions_multi_enemy(self, model, enemy_positions):
        """
        Find optimal positions against multiple enemies
        
        Parameters:
            model: Trained LSTM model
            enemy_positions: List of enemy position coordinates
            
        Returns:
            best_positions: Dictionary with optimal positions for each unit
            win_probability: Probability of victory with those positions
        """
        best_positions = {
            'infantry': [0, 0],
            'tank': [0, 0],
            'drone': [0, 0]
        }
        best_prob = 0
        
        # Get expected input size
        input_size = self._get_model_input_size(model)
        grid_size = 10
        
        # Sample positions (full grid search would be 10^6 combinations)
        # Instead, we'll use strategic sampling
        samples = 30  # Number of sample positions to try per unit
        positions_tried = 0
        
        # Generate position candidates for each unit
        infantry_positions = self._generate_position_candidates(samples, grid_size)
        tank_positions = self._generate_position_candidates(samples, grid_size)
        drone_positions = self._generate_position_candidates(samples, grid_size)
        
        # Try different position combinations
        for inf_pos in infantry_positions:
            for tank_pos in tank_positions:
                # Skip if units overlap
                if inf_pos == tank_pos:
                    continue
                    
                for drone_pos in drone_positions:
                    # Skip if units overlap
                    if drone_pos == inf_pos or drone_pos == tank_pos:
                        continue
                    
                    positions_tried += 1
                    
                    # Calculate probabilities against each enemy
                    enemy_probs = []
                    for enemy_position in enemy_positions:
                        # Skip if any unit is on an enemy
                        if inf_pos == enemy_position or tank_pos == enemy_position or drone_pos == enemy_position:
                            continue
                            
                        # Construct basic battle state
                        base_battle_state = [
                            inf_pos[0], inf_pos[1],
                            tank_pos[0], tank_pos[1],
                            drone_pos[0], drone_pos[1],
                            0, 0, 0,  # Default actions
                            enemy_position[0], enemy_position[1]
                        ]
                        
                        # Pad the battle state
                        battle_state = self._pad_feature_vector(base_battle_state, input_size)
                        
                        # Get win probability against this enemy
                        prob = predict_battle_outcome(model, battle_state)
                        enemy_probs.append(prob)
                    
                    # Skip if we don't have probabilities for all enemies
                    if len(enemy_probs) != len(enemy_positions):
                        continue
                    
                    # Calculate overall probability (using average)
                    avg_prob = sum(enemy_probs) / len(enemy_probs)
                    
                    # Update if better
                    if avg_prob > best_prob:
                        best_prob = avg_prob
                        best_positions = {
                            'infantry': inf_pos,
                            'tank': tank_pos,
                            'drone': drone_pos
                        }
        
        print(f"Evaluated {positions_tried} position combinations")
        return best_positions, best_prob

    def _generate_position_candidates(self, count, grid_size):
        """
        Generate a list of candidate positions for unit placement
        
        Parameters:
            count: Number of positions to generate
            grid_size: Size of the battlefield grid
            
        Returns:
            positions: List of [x, y] position coordinates
        """
        positions = []
        
        # Add some positions along the edges
        for _ in range(count // 4):
            if random.random() < 0.5:
                # Edge of the grid
                x = random.choice([0, 1, grid_size-2, grid_size-1])
                y = random.randint(0, grid_size-1)
            else:
                # Edge of the grid
                y = random.choice([0, 1, grid_size-2, grid_size-1])
                x = random.randint(0, grid_size-1)
            positions.append([x, y])
        
        # Add some positions near the center
        for _ in range(count // 4):
            x = random.randint(grid_size//4, 3*grid_size//4)
            y = random.randint(grid_size//4, 3*grid_size//4)
            positions.append([x, y])
        
        # Add completely random positions for the remainder
        remaining = count - len(positions)
        for _ in range(remaining):
            x = random.randint(0, grid_size-1)
            y = random.randint(0, grid_size-1)
            positions.append([x, y])
        
        return positions

    def _get_model_input_size(self, model):
        """Helper function to determine model's input size"""
        # Default input size
        input_size = 43
        
        # Try to infer from model architecture
        for param in model.parameters():
            if len(param.shape) >= 2:
                # First layer's weight matrix should have shape [hidden_size, input_size]
                input_size = param.shape[1]
                break
                
        return input_size

    def _pad_feature_vector(self, features, target_size):
        """Pad feature vector to match expected model input size"""
        current_size = len(features)
        
        if current_size == target_size:
            return features
        
        if current_size > target_size:
            # Truncate if too large
            return features[:target_size]
        
        # Pad with zeros
        return features + [0.0] * (target_size - current_size)

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
                elif message[0] == "training_stats":  # Handle training stats updates
                    if hasattr(self, 'training_stats_text'):
                        self.training_stats_text.insert(tk.END, message[1] + "\n")
                        self.training_stats_text.see(tk.END)
                elif message[0] == "self_play_result":
                    if hasattr(self, 'self_play_results'):
                        self.self_play_results.delete(1.0, tk.END)
                        self.self_play_results.insert(tk.END, message[1])
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
        ttk.Checkbutton

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
        ttk.Spinbox(params_frame, from_=0.0001, to=0.1, increment=0.0001, textvariable=self.lr_var, width=10).grid(row=2, colum