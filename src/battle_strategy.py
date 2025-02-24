"""
Battle Strategy Module

Uses the trained LSTM model to find optimal unit actions and positions
for maximizing battle victory probability.
"""
import torch
import numpy as np
from src.lstm_model import load_model, predict_battle_outcome


def get_optimal_actions(model, unit_positions, enemy_position):
    """
    Find the optimal actions for given unit positions to maximize victory probability
    
    Parameters:
        model: Trained LSTM model
        unit_positions: Dictionary with infantry, tank, drone positions
        enemy_position: List [x, y] of enemy position
        
    Returns:
        best_actions: List of best actions for infantry, tank, drone
        win_probability: Probability of victory with those actions
    """
    # Extract positions
    infantry_pos = unit_positions['infantry']
    tank_pos = unit_positions['tank']
    drone_pos = unit_positions['drone']
    
    best_actions = [0, 0, 0]  # Default actions
    best_prob = 0
    
    # Try all action combinations (5 actions for each of 3 units = 125 combinations)
    for inf_action in range(5):
        for tank_action in range(5):
            for drone_action in range(5):
                # Construct battle state
                battle_state = [
                    infantry_pos[0], infantry_pos[1],
                    tank_pos[0], tank_pos[1],
                    drone_pos[0], drone_pos[1],
                    inf_action, tank_action, drone_action,
                    enemy_position[0], enemy_position[1]
                ]
                
                # Get win probability
                prob = predict_battle_outcome(model, battle_state)
                
                # Update if better
                if prob > best_prob:
                    best_prob = prob
                    best_actions = [inf_action, tank_action, drone_action]
    
    return best_actions, best_prob


def get_optimal_positioning(model, enemy_position, grid_size=10, action_set=[0, 0, 0]):
    """
    Find optimal unit positions for maximizing victory probability
    
    Parameters:
        model: Trained LSTM model
        enemy_position: List [x, y] of enemy position
        grid_size: Size of the battlefield grid
        action_set: Fixed action set to use [infantry_action, tank_action, drone_action]
        
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
    
    # Sample positions (full grid search would be 10^6 combinations)
    # Instead, we'll do strategic sampling based on distance to enemy
    samples = 100
    positions_tried = 0
    
    # Helper to generate positions with various distances to enemy
    def generate_position_candidates(enemy_pos, grid_size, count=10):
        positions = []
        enemy_x, enemy_y = enemy_pos
        
        # Close positions (1-3 squares away)
        for _ in range(count // 3):
            distance = np.random.randint(1, 4)
            angle = np.random.uniform(0, 2 * np.pi)
            x = int(enemy_x + distance * np.cos(angle))
            y = int(enemy_y + distance * np.sin(angle))
            if 0 <= x < grid_size and 0 <= y < grid_size:
                positions.append([x, y])
        
        # Medium positions (4-6 squares away)
        for _ in range(count // 3):
            distance = np.random.randint(4, 7)
            angle = np.random.uniform(0, 2 * np.pi)
            x = int(enemy_x + distance * np.cos(angle))
            y = int(enemy_y + distance * np.sin(angle))
            if 0 <= x < grid_size and 0 <= y < grid_size:
                positions.append([x, y])
        
        # Far positions (7-9 squares away)
        for _ in range(count // 3):
            distance = np.random.randint(7, 10)
            angle = np.random.uniform(0, 2 * np.pi)
            x = int(enemy_x + distance * np.cos(angle))
            y = int(enemy_y + distance * np.sin(angle))
            if 0 <= x < grid_size and 0 <= y < grid_size:
                positions.append([x, y])
                
        return positions
    
    # Generate position candidates for each unit
    infantry_candidates = generate_position_candidates(enemy_position, grid_size, samples)
    tank_candidates = generate_position_candidates(enemy_position, grid_size, samples)
    drone_candidates = generate_position_candidates(enemy_position, grid_size, samples)
    
    # Try different position combinations
    for inf_pos in infantry_candidates:
        for tank_pos in tank_candidates:
            for drone_pos in drone_candidates:
                # Skip if units overlap
                if (inf_pos == tank_pos or inf_pos == drone_pos or 
                    tank_pos == drone_pos or inf_pos == enemy_position or
                    tank_pos == enemy_position or drone_pos == enemy_position):
                    continue
                    
                positions_tried += 1
                
                # Construct battle state
                battle_state = [
                    inf_pos[0], inf_pos[1],
                    tank_pos[0], tank_pos[1],
                    drone_pos[0], drone_pos[1],
                    action_set[0], action_set[1], action_set[2],
                    enemy_position[0], enemy_position[1]
                ]
                
                # Get win probability
                prob = predict_battle_outcome(model, battle_state)
                
                # Update if better
                if prob > best_prob:
                    best_prob = prob
                    best_positions = {
                        'infantry': inf_pos,
                        'tank': tank_pos,
                        'drone': drone_pos
                    }
    
    print(f"Evaluated {positions_tried} position combinations")
    return best_positions, best_prob


def generate_battle_heatmap(model, enemy_position, unit_positions=None, actions=None):
    """
    Generate a heatmap showing victory probability for different infantry positions
    
    Parameters:
        model: Trained LSTM model
        enemy_position: [x, y] position of enemy
        unit_positions: Optional fixed positions for tank and drone
        actions: Optional fixed actions for all units
        
    Returns:
        heatmap: 2D numpy array with victory probabilities
    """
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Default positions if not provided
    if unit_positions is None:
        unit_positions = {
            'tank': [5, 5],
            'drone': [7, 3]
        }
    
    # Default actions if not provided
    if actions is None:
        actions = [0, 0, 0]  # Move, Move, Move
    
    # Create empty heatmap
    grid_size = 10
    heatmap = np.zeros((grid_size, grid_size))
    
    # Calculate probability for each infantry position
    for x in range(grid_size):
        for y in range(grid_size):
            # Skip if position conflicts with other units
            if ([x, y] == unit_positions['tank'] or 
                [x, y] == unit_positions['drone'] or
                [x, y] == enemy_position):
                heatmap[x, y] = 0
                continue
            
            # Construct battle state
            battle_state = [
                x, y,  # Infantry position
                unit_positions['tank'][0], unit_positions['tank'][1],
                unit_positions['drone'][0], unit_positions['drone'][1],
                actions[0], actions[1], actions[2],
                enemy_position[0], enemy_position[1]
            ]
            
            # Get win probability
            heatmap[x, y] = predict_battle_outcome(model, battle_state)
    
    return heatmap


def visualize_battle_heatmap(heatmap, enemy_position, unit_positions=None):
    """
    Visualize the victory probability heatmap
    
    Parameters:
        heatmap: 2D numpy array with victory probabilities
        enemy_position: [x, y] position of enemy
        unit_positions: Optional dictionary with positions of other units
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    
    plt.figure(figsize=(10, 8))
    
    # Plot heatmap
    plt.imshow(heatmap, cmap='viridis', interpolation='nearest', origin='lower')
    plt.colorbar(label='Victory Probability')
    
    # Mark enemy position
    plt.scatter(enemy_position[1], enemy_position[0], 
                color='red', s=200, marker='*', label='Enemy')
    
    # Mark other unit positions if provided
    if unit_positions:
        if 'tank' in unit_positions:
            plt.scatter(unit_positions['tank'][1], unit_positions['tank'][0],
                      color='blue', s=150, marker='s', label='Tank')
        if 'drone' in unit_positions:
            plt.scatter(unit_positions['drone'][1], unit_positions['drone'][0],
                      color='green', s=100, marker='^', label='Drone')
    
    # Add grid
    plt.grid(which='both', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
    
    # Add labels and legend
    plt.title('Infantry Victory Probability Heatmap')
    plt.xlabel('Y Coordinate')
    plt.ylabel('X Coordinate')
    plt.legend(loc='lower right')
    
    # Save the visualization
    plt.savefig('visualizations/battle_heatmap.png')
    plt.close()
    
    print("Heatmap visualization saved as 'visualizations/battle_heatmap.png'")


def battle_advisor(model_path='models/best_battle_predictor.pt'):
    """
    Interactive battle advisor function that suggests optimal actions and positions
    """
    print("🎮 Battle Strategy Advisor 🎮")
    print("-" * 40)
    
    # Load model
    try:
        model = load_model(model_path)
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    while True:
        print("\nWhat would you like to do?")
        print("1. Find optimal actions for given positions")
        print("2. Find optimal positions")
        print("3. Generate victory probability heatmap")
        print("4. Exit")
        
        choice = input("Enter your choice (1-4): ")
        
        if choice == '1':
            try:
                # Get unit positions
                inf_x = int(input("Infantry X position (0-9): "))
                inf_y = int(input("Infantry Y position (0-9): "))
                tank_x = int(input("Tank X position (0-9): "))
                tank_y = int(input("Tank Y position (0-9): "))
                drone_x = int(input("Drone X position (0-9): "))
                drone_y = int(input("Drone Y position (0-9): "))
                enemy_x = int(input("Enemy X position (0-9): "))
                enemy_y = int(input("Enemy Y position (0-9): "))
                
                unit_positions = {
                    'infantry': [inf_x, inf_y],
                    'tank': [tank_x, tank_y],
                    'drone': [drone_x, drone_y]
                }
                enemy_position = [enemy_x, enemy_y]
                
                # Get optimal actions
                best_actions, win_prob = get_optimal_actions(model, unit_positions, enemy_position)
                
                # Display results
                action_names = ['Move', 'Attack', 'Defend', 'Retreat', 'Support']
                print("\n🎲 Optimal Strategy:")
                print(f"Infantry: {action_names[best_actions[0]]}")
                print(f"Tank: {action_names[best_actions[1]]}")
                print(f"Drone: {action_names[best_actions[2]]}")
                print(f"Victory probability: {win_prob:.2%}")
                
            except Exception as e:
                print(f"❌ Error: {e}")
                
        elif choice == '2':
            try:
                # Get enemy position
                enemy_x = int(input("Enemy X position (0-9): "))
                enemy_y = int(input("Enemy Y position (0-9): "))
                enemy_position = [enemy_x, enemy_y]
                
                # Get optimal positions
                best_positions, win_prob = get_optimal_positioning(model, enemy_position)
                
                # Display results
                print("\n🎯 Optimal Unit Positions:")
                print(f"Infantry: {best_positions['infantry']}")
                print(f"Tank: {best_positions['tank']}")
                print(f"Drone: {best_positions['drone']}")
                print(f"Victory probability: {win_prob:.2%}")
                
            except Exception as e:
                print(f"❌ Error: {e}")
                
        elif choice == '3':
            try:
                # Get positions
                enemy_x = int(input("Enemy X position (0-9): "))
                enemy_y = int(input("Enemy Y position (0-9): "))
                tank_x = int(input("Tank X position (0-9): "))
                tank_y = int(input("Tank Y position (0-9): "))
                drone_x = int(input("Drone X position (0-9): "))
                drone_y = int(input("Drone Y position (0-9): "))
                
                enemy_position = [enemy_x, enemy_y]
                unit_positions = {
                    'tank': [tank_x, tank_y],
                    'drone': [drone_x, drone_y]
                }
                
                # Generate and visualize heatmap
                print("Generating heatmap (this may take a moment)...")
                heatmap = generate_battle_heatmap(model, enemy_position, unit_positions)
                visualize_battle_heatmap(heatmap, enemy_position, unit_positions)
                
            except Exception as e:
                print(f"❌ Error: {e}")
                
        elif choice == '4':
            print("Thank you for using Battle Strategy Advisor!")
            break
            
        else:
            print("Invalid choice. Please enter a number between 1 and 4.")
            

if __name__ == "__main__":
    battle_advisor()