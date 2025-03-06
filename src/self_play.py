"""
Self-Play Module for Battlefield Simulation

This module implements functionality for the LSTM model to play against itself
by controlling both friendly and enemy units in the battlefield.
"""
import os
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

# Try importing from both module styles to be flexible
try:
    # Direct imports
    from battlefield_env import BattlefieldEnv, UnitType, ActionType, TERRAIN_TYPES, WEATHER_CONDITIONS, Enemy
    from lstm_model import load_model, predict_battle_outcome
    from battle_strategy import get_optimal_actions, get_optimal_positioning
except ImportError:
    # Fallback to src directory structure
    from src.battlefield_env import BattlefieldEnv, UnitType, ActionType, TERRAIN_TYPES, WEATHER_CONDITIONS, Enemy
    from src.lstm_model import load_model, predict_battle_outcome
    from src.battle_strategy import get_optimal_actions, get_optimal_positioning


class SelfPlaySimulation:
    """
    Self-play simulation where the model controls both sides of the battlefield
    """
    def __init__(self, model_path='models/best_battle_predictor.pt', max_steps=50, 
                 save_visualizations=True, visualization_dir='visualizations/self_play',
                 max_enemies=3, terrain_type=None, weather_type=None, 
                 enable_artillery=False, enable_stealth=False):
        """
        Initialize the self-play simulation
        
        Parameters:
            model_path: Path to the trained LSTM model
            max_steps: Maximum steps per battle
            save_visualizations: Whether to save battle visualizations
            visualization_dir: Directory to save visualizations
            max_enemies: Maximum number of enemies per battle (1-5)
            terrain_type: Specific terrain type to use (None for random)
            weather_type: Specific weather type to use (None for random)
            enable_artillery: Whether to enable artillery units
            enable_stealth: Whether to enable stealth units
        """
        self.max_steps = max_steps
        self.save_visualizations = save_visualizations
        self.visualization_dir = visualization_dir
        self.max_enemies = min(max(1, max_enemies), 5)  # Clamp between 1-5
        self.terrain_type = terrain_type
        self.weather_type = weather_type
        self.enable_artillery = enable_artillery
        self.enable_stealth = enable_stealth
        
        # Create visualization directory if it doesn't exist
        if save_visualizations:
            os.makedirs(visualization_dir, exist_ok=True)
        
        # Load the model
        try:
            self.model = load_model(model_path)
            print(f"✓ Model loaded successfully from {model_path}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.model = None
        
        # Initialize the environment
        self.env = None
        
        # Statistics tracking
        self.battle_stats = {
            'wins_blue': 0,
            'wins_red': 0,
            'draws': 0,
            'avg_steps': [],
            'blue_units_remaining': [],
            'red_units_remaining': []
        }
        
    def run_self_play_battle(self, log_steps=True, render_battle=True):
        """
        Run a single self-play battle where the model controls both sides
        
        Parameters:
            log_steps: Whether to log step information
            render_battle: Whether to render the battle state (only final state)
            
        Returns:
            result: Dictionary with battle outcome information
        """
        # Initialize environment with custom parameters
        self.env = self._create_custom_environment()
        obs = self.env.reset()
        
        if log_steps:
            print(f"\n{'='*50}")
            print(f"Starting Self-Play Battle | Terrain: {self.env.current_terrain} | Weather: {self.env.current_weather}")
            print(f"{'='*50}")
            
            # Initial state
            blue_units = [unit.unit_type.name for unit in self.env.friendly_units]
            red_units = [enemy.unit_type.name for enemy in self.env.enemies]
            
            print("Initial setup:")
            print(f"Blue forces: {len(self.env.friendly_units)} units ({', '.join(blue_units)})")
            print(f"Red forces: {len(self.env.enemies)} units ({', '.join(red_units)})")
            
        # Battle loop
        done = False
        step = 0
        
        while not done and step < self.max_steps:
            # Get blue actions (friendly units)
            blue_actions = self._get_model_actions_for_blue()
            
            # Execute blue actions
            obs, reward, done, info = self.env.step(blue_actions)
            
            # If battle ended after blue's turn, break
            if done:
                break
            
            # Get red actions (enemies) - this requires overriding the enemy AI
            red_actions = self._get_model_actions_for_red()
            
            # Execute red actions manually
            for i, (enemy, action) in enumerate(zip(self.env.enemies, red_actions)):
                if enemy.is_alive():
                    action_type = ActionType(action)
                    target = self._select_target_for_enemy(enemy, action_type)
                    result = enemy.take_action(action_type, self.env, target)
                    if log_steps and step % 5 == 0:
                        print(f"Enemy {i} performs {action_type.name} → {result.get('message', '')}")
            
            # Check if battle ended after red's moves
            if self._check_battle_end():
                done = True
                info = self._get_battle_result()
            
            step += 1
            
            # Log every 5 steps or final step
            if log_steps and (step % 5 == 0 or done):
                blue_alive = sum(1 for unit in self.env.friendly_units if unit.is_alive())
                red_alive = sum(1 for enemy in self.env.enemies if enemy.is_alive())
                
                blue_health = sum(unit.hp for unit in self.env.friendly_units if unit.is_alive())
                red_health = sum(enemy.hp for enemy in self.env.enemies if enemy.is_alive())
                
                print(f"\nStep {step}: Blue forces: {blue_alive}/{len(self.env.friendly_units)} units ({blue_health} HP)")
                print(f"Step {step}: Red forces: {red_alive}/{len(self.env.enemies)} units ({red_health} HP)")
        
        # Only render the final battle state if requested
        if render_battle:
            self._render_and_save(step, is_final=True)
            
        # Get battle result
        result = self._get_battle_result()
        
        if log_steps:
            print(f"\n{'='*50}")
            print(f"Battle ended after {step} steps")
            print(f"Result: {result['result']}")
            print(f"Blue units remaining: {result['blue_remaining']}/{len(self.env.friendly_units)}")
            print(f"Red units remaining: {result['red_remaining']}/{len(self.env.enemies)}")
            print(f"{'='*50}")
        
        # Update statistics
        self._update_stats(result, step)
        
        return result
        
        
    def _create_custom_environment(self):
        """
        Create a battlefield environment with custom parameters
        
        Returns:
            env: Customized BattlefieldEnv instance
        """
        # Create base environment
        env = BattlefieldEnv(max_steps=self.max_steps, max_enemies=self.max_enemies)
        
        # Set custom terrain if specified
        if self.terrain_type is not None and self.terrain_type in TERRAIN_TYPES:
            # We need to override the terrain generation
            # This is a bit hacky, but we're working with what we have
            if hasattr(env, 'current_terrain'):
                env.current_terrain = self.terrain_type
                
            # Try to update terrain data if possible
            if hasattr(env, 'terrain_data'):
                # Set all cells to this terrain type
                for x in range(env.grid_size):
                    for y in range(env.grid_size):
                        env.terrain_data[x][y] = TERRAIN_TYPES[self.terrain_type]
                        
        # Set custom weather if specified
        if self.weather_type is not None and self.weather_type in WEATHER_CONDITIONS:
            if hasattr(env, 'current_weather'):
                env.current_weather = self.weather_type
                
        # Handle unit type customization
        # We need to reset the environment to apply these changes
        env.reset()
        
        # Customize enemy types if needed
        if self.enable_artillery or self.enable_stealth:
            # Remove existing enemies
            original_enemy_count = len(env.enemies)
            env.enemies = []
            
            # Regenerate enemies with custom types
            for i in range(original_enemy_count):
                # Choose a type with emphasis on enabling the requested types
                available_types = [UnitType.INFANTRY, UnitType.ARMORED, UnitType.AERIAL]
                
                if self.enable_artillery:
                    available_types.append(UnitType.ARTILLERY)
                    # Add it twice to increase probability
                    available_types.append(UnitType.ARTILLERY)
                    
                if self.enable_stealth:
                    available_types.append(UnitType.STEALTH)
                    # Add it twice to increase probability
                    available_types.append(UnitType.STEALTH)
                    
                # Random position that doesn't conflict with existing units
                positions = env._generate_non_overlapping_positions(1, 
                    exclude_positions=[tuple(unit.position) for unit in env.friendly_units] + 
                                     [tuple(enemy.position) for enemy in env.enemies])
                
                # Select random type from available types
                unit_type = random.choice(available_types)
                
                # Create enemy with appropriate stats based on type
                aggression = random.uniform(0.6, 0.95)
                
                if unit_type == UnitType.INFANTRY:
                    enemy = Enemy(
                        unit_type=unit_type,
                        position=positions[0],
                        hp=100,
                        attack_power=12,
                        speed=2,
                        detection_range=2,
                        stealth_level=0,
                        ai_aggression=aggression
                    )
                elif unit_type == UnitType.ARMORED:
                    enemy = Enemy(
                        unit_type=unit_type,
                        position=positions[0],
                        hp=200, 
                        attack_power=35,
                        speed=1,
                        detection_range=2,
                        stealth_level=0,
                        ai_aggression=aggression
                    )
                elif unit_type == UnitType.AERIAL:
                    enemy = Enemy(
                        unit_type=unit_type,
                        position=positions[0],
                        hp=60,
                        attack_power=15,
                        speed=3,
                        detection_range=3,
                        stealth_level=1,
                        ai_aggression=aggression
                    )
                elif unit_type == UnitType.ARTILLERY:
                    enemy = Enemy(
                        unit_type=unit_type,
                        position=positions[0],
                        hp=70,
                        attack_power=40,
                        speed=1,
                        detection_range=4,
                        stealth_level=0,
                        ai_aggression=aggression * 0.8  # Artillery is more defensive
                    )
                elif unit_type == UnitType.STEALTH:
                    enemy = Enemy(
                        unit_type=unit_type,
                        position=positions[0],
                        hp=60,
                        attack_power=20,
                        speed=2,
                        detection_range=3,
                        stealth_level=3,
                        ai_aggression=aggression
                    )
                    
                env.enemies.append(enemy)
            
        return env
    
    def run_multiple_battles(self, num_battles=10, log_individual=False):
        """
        Run multiple self-play battles and collect statistics
        
        Parameters:
            num_battles: Number of battles to run
            log_individual: Whether to log details of each battle
            
        Returns:
            stats: Dictionary with overall battle statistics
        """
        print(f"Running {num_battles} self-play battles...")
        
        # Reset stats
        self.battle_stats = {
            'wins_blue': 0,
            'wins_red': 0,
            'draws': 0,
            'avg_steps': [],
            'blue_units_remaining': [],
            'red_units_remaining': []
        }
        
        for i in range(num_battles):
            print(f"\nBattle {i+1}/{num_battles}")
            result = self.run_self_play_battle(log_steps=log_individual, render_battle=self.save_visualizations)
            
            # Brief summary if not logging individual battles
            if not log_individual:
                print(f"Battle {i+1}: {result['result']} after {len(self.battle_stats['avg_steps'][-1])} steps")
        
        # Calculate overall statistics
        win_rate_blue = self.battle_stats['wins_blue'] / num_battles * 100
        win_rate_red = self.battle_stats['wins_red'] / num_battles * 100
        draw_rate = self.battle_stats['draws'] / num_battles * 100
        
        avg_steps = sum([sum(steps)/len(steps) for steps in self.battle_stats['avg_steps']]) / len(self.battle_stats['avg_steps']) if self.battle_stats['avg_steps'] else 0
        
        avg_blue_remaining = sum(self.battle_stats['blue_units_remaining']) / len(self.battle_stats['blue_units_remaining']) if self.battle_stats['blue_units_remaining'] else 0
        avg_red_remaining = sum(self.battle_stats['red_units_remaining']) / len(self.battle_stats['red_units_remaining']) if self.battle_stats['red_units_remaining'] else 0
        
        print(f"\n{'='*50}")
        print(f"Self-Play Battle Statistics (Total: {num_battles} battles)")
        print(f"{'='*50}")
        print(f"Blue victory rate: {win_rate_blue:.1f}%")
        print(f"Red victory rate: {win_rate_red:.1f}%")
        print(f"Draw rate: {draw_rate:.1f}%")
        print(f"Average battle length: {avg_steps:.1f} steps")
        print(f"Average blue units remaining: {avg_blue_remaining:.2f}")
        print(f"Average red units remaining: {avg_red_remaining:.2f}")
        
        # Visualize statistics
        self._visualize_stats(num_battles)
        
        return self.battle_stats
    
    def _get_model_actions_for_blue(self):
        """
        Use the model to decide actions for blue forces (friendly units)
        
        Returns:
            actions: List of action indices for each friendly unit
        """
        # If no model is loaded, use random actions
        if self.model is None:
            return [random.randint(0, 4) for _ in range(len(self.env.friendly_units))]
        
        # Get unit positions
        unit_positions = {
            'infantry': self.env.friendly_units[0].position if self.env.friendly_units[0].is_alive() else [0, 0],
            'tank': self.env.friendly_units[1].position if self.env.friendly_units[1].is_alive() else [0, 0],
            'drone': self.env.friendly_units[2].position if self.env.friendly_units[2].is_alive() else [0, 0]
        }
        
        # Find nearest enemy for targeting
        nearest_enemy = None
        nearest_distance = float('inf')
        
        for enemy in self.env.enemies:
            if enemy.is_alive():
                # Average distance to all friendly units
                total_distance = 0
                for unit in self.env.friendly_units:
                    if unit.is_alive():
                        dx = unit.position[0] - enemy.position[0]
                        dy = unit.position[1] - enemy.position[1]
                        total_distance += (dx*dx + dy*dy)**0.5
                
                avg_distance = total_distance / sum(1 for unit in self.env.friendly_units if unit.is_alive())
                
                if avg_distance < nearest_distance:
                    nearest_distance = avg_distance
                    nearest_enemy = enemy
        
        # If no enemies alive, use default actions
        if nearest_enemy is None:
            return [0, 0, 0]  # Default to 'move' for all units
        
        # Get optimal actions
        try:
            best_actions, _ = get_optimal_actions(self.model, unit_positions, nearest_enemy.position)
            return best_actions
        except Exception as e:
            print(f"Error getting model actions for blue: {e}")
            return [0, 0, 0]  # Default to 'move' for all units
    
    def _get_model_actions_for_red(self):
        """
        Use the model to decide actions for red forces (enemy units)
        This method uses the model to control enemy behavior instead of the default AI
        
        Returns:
            actions: List of action indices for each enemy
        """
        # If no model is loaded, use default AI behavior
        if self.model is None:
            return [0] * len(self.env.enemies)  # Default to 'move' for all enemies
        
        # Actions for each enemy
        actions = []
        
        # For each enemy, get optimal action
        for enemy in self.env.enemies:
            if not enemy.is_alive():
                actions.append(0)  # Default action for dead units
                continue
            
            # Find nearest friendly unit for targeting
            nearest_friendly = None
            nearest_distance = float('inf')
            
            for unit in self.env.friendly_units:
                if unit.is_alive():
                    dx = enemy.position[0] - unit.position[0]
                    dy = enemy.position[1] - unit.position[1]
                    distance = (dx*dx + dy*dy)**0.5
                    
                    if distance < nearest_distance:
                        nearest_distance = distance
                        nearest_friendly = unit
            
            # If no friendly units alive, use default action
            if nearest_friendly is None:
                actions.append(0)  # Default to 'move'
                continue
            
            # Create state representation for the model
            try:
                # Simplified state for the model (enemy perspective)
                # We need to reverse the roles here - enemy becomes "friendly" for prediction
                unit_positions = {
                    'infantry': enemy.position,
                    'tank': enemy.position,  # Use same position for simplicity
                    'drone': enemy.position  # Use same position for simplicity
                }
                
                # Try to predict the best action against the nearest friendly
                best_actions, _ = get_optimal_actions(self.model, unit_positions, nearest_friendly.position)
                
                # Use only the first action (since we're treating all enemies as "infantry")
                actions.append(best_actions[0])
            except Exception as e:
                print(f"Error getting model action for enemy: {e}")
                # Fallback to a heuristic method
                
                # If close to a friendly unit, attack
                if nearest_distance < 3:
                    actions.append(1)  # Attack
                elif nearest_distance < 5:
                    actions.append(2)  # Defend
                else:
                    actions.append(0)  # Move
        
        return actions
    
    def _select_target_for_enemy(self, enemy, action_type):
        """
        Select an appropriate target for enemy action
        
        Parameters:
            enemy: The enemy unit taking action
            action_type: The ActionType enum value
            
        Returns:
            target: The target unit or None
        """
        # If not an action that needs a target
        if action_type not in [ActionType.ATTACK, ActionType.SUPPORT]:
            return None
        
        # Find appropriate targets
        potential_targets = []
        
        if action_type == ActionType.ATTACK:
            # Target friendly units
            for unit in self.env.friendly_units:
                if unit.is_alive():
                    # Calculate distance
                    dx = enemy.position[0] - unit.position[0]
                    dy = enemy.position[1] - unit.position[1]
                    distance = (dx*dx + dy*dy)**0.5
                    
                    # Score based on distance and unit health
                    score = (10 - min(distance, 10)) + (1 - unit.hp/unit.max_hp) * 5
                    potential_targets.append((unit, score))
        else:  # SUPPORT
            # Target other enemies
            for other_enemy in self.env.enemies:
                if other_enemy.is_alive() and other_enemy != enemy:
                    # Calculate distance
                    dx = enemy.position[0] - other_enemy.position[0]
                    dy = enemy.position[1] - other_enemy.position[1]
                    distance = (dx*dx + dy*dy)**0.5
                    
                    # Score based on distance and unit health
                    score = (10 - min(distance, 10)) + (1 - other_enemy.hp/other_enemy.max_hp) * 5
                    potential_targets.append((other_enemy, score))
        
        # Sort by score (higher is better)
        potential_targets.sort(key=lambda x: x[1], reverse=True)
        
        # Return the best target or None if no targets
        return potential_targets[0][0] if potential_targets else None
    
    def _check_battle_end(self):
        """
        Check if the battle should end
        
        Returns:
            ended: True if battle should end, False otherwise
        """
        # Check if all units of one side are defeated
        blue_alive = sum(1 for unit in self.env.friendly_units if unit.is_alive())
        red_alive = sum(1 for enemy in self.env.enemies if enemy.is_alive())
        
        return blue_alive == 0 or red_alive == 0
    
    def _get_battle_result(self):
        """
        Determine the outcome of the battle
        
        Returns:
            result: Dictionary with battle outcome information
        """
        # Count alive units
        blue_alive = sum(1 for unit in self.env.friendly_units if unit.is_alive())
        red_alive = sum(1 for enemy in self.env.enemies if enemy.is_alive())
        
        # Determine outcome
        if blue_alive == 0 and red_alive == 0:
            outcome = "draw"
        elif blue_alive == 0:
            outcome = "red_victory"
        elif red_alive == 0:
            outcome = "blue_victory"
        else:
            # If both sides have units, compare relative strength
            blue_health = sum(unit.hp / unit.max_hp for unit in self.env.friendly_units if unit.is_alive())
            red_health = sum(enemy.hp / enemy.max_hp for enemy in self.env.enemies if enemy.is_alive())
            
            blue_strength = blue_alive + blue_health * 0.5
            red_strength = red_alive + red_health * 0.5
            
            if blue_strength > red_strength * 1.5:
                outcome = "blue_advantage"
            elif red_strength > blue_strength * 1.5:
                outcome = "red_advantage"
            else:
                outcome = "ongoing"
        
        return {
            'result': outcome,
            'blue_remaining': blue_alive,
            'blue_total': len(self.env.friendly_units),
            'red_remaining': red_alive,
            'red_total': len(self.env.enemies)
        }
    
    def _update_stats(self, result, steps):
        """
        Update battle statistics with improved survival rate tracking
        
        Parameters:
            result: Battle result dictionary
            steps: Number of steps the battle took
        """
        if result['result'] == 'blue_victory' or result['result'] == 'blue_advantage':
            self.battle_stats['wins_blue'] += 1
        elif result['result'] == 'red_victory' or result['result'] == 'red_advantage':
            self.battle_stats['wins_red'] += 1
        elif result['result'] == 'draw':
            self.battle_stats['draws'] += 1
        else:  # Handle 'ongoing' outcomes by classifying based on relative strength
            # If we reach max steps, force a classification based on unit count/health
            blue_remaining = result['blue_remaining'] / result['blue_total']
            red_remaining = result['red_remaining'] / result['red_total']
            
            if blue_remaining > red_remaining * 1.1:
                self.battle_stats['wins_blue'] += 1
            elif red_remaining > blue_remaining * 1.1:
                self.battle_stats['wins_red'] += 1
            else:
                self.battle_stats['draws'] += 1
        
        # Update steps
        self.battle_stats['avg_steps'].append([steps])
        
        # Update units remaining - calculate survival rates for both sides in ALL battles
        blue_survival_rate = result['blue_remaining'] / result['blue_total']
        red_survival_rate = result['red_remaining'] / result['red_total']
        
        # If Blue won, then Red's survival rate should be 0 (all Red units were eliminated)
        # If Red won, then Blue's survival rate should be 0 (all Blue units were eliminated)
        if result['result'] == 'blue_victory':
            red_survival_rate = 0.0  # All Red units were eliminated
        elif result['result'] == 'red_victory':
            blue_survival_rate = 0.0  # All Blue units were eliminated
        
        # Now track the survival rates that reflect the actual battle outcome
        self.battle_stats['blue_units_remaining'].append(blue_survival_rate)
        self.battle_stats['red_units_remaining'].append(red_survival_rate)
    
    def _render_and_save(self, step, is_final=False):
        """
        Render the battle state and save as an image
        
        Parameters:
            step: Current step number
            is_final: Whether this is the final state
        """
        if not self.save_visualizations:
            return
        
        # Only save the final battle state
        if not is_final:
            return
        
        try:
            # Create filename - use timestamps for uniqueness
            import time
            battle_id = int(time.time()) % 10000  # Last 4 digits of timestamp
            
            filename = f"{self.visualization_dir}/battle_{battle_id}_final.png"
            
            # Render using the environment's render method
            fig = self.env.render(mode='human', save_path=filename)
            
            # Close the figure to prevent memory leaks
            plt.close(fig)
            
            print(f"Final battle state saved to {filename}")
            
        except Exception as e:
            print(f"Error saving visualization: {e}")
    
    def _visualize_stats(self, num_battles):
        """
        Visualize battle statistics
        
        Parameters:
            num_battles: Total number of battles run
        """
        if not self.save_visualizations:
            return
        
        try:
            # Create a figure with two subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Plot 1: Win rates
            labels = ['Blue Wins', 'Red Wins', 'Draws']
            sizes = [
                self.battle_stats['wins_blue'],
                self.battle_stats['wins_red'],
                self.battle_stats['draws']
            ]
            colors = ['#3498db', '#e74c3c', '#95a5a6']
            
            ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', 
                   shadow=True, startangle=90)
            ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
            ax1.set_title('Battle Outcomes')
            
            # Plot 2: Average units remaining
            categories = ['Blue', 'Red']
            values = [
                sum(self.battle_stats['blue_units_remaining']) / len(self.battle_stats['blue_units_remaining']) if self.battle_stats['blue_units_remaining'] else 0,
                sum(self.battle_stats['red_units_remaining']) / len(self.battle_stats['red_units_remaining']) if self.battle_stats['red_units_remaining'] else 0
            ]
            
            ax2.bar(categories, values, color=['#3498db', '#e74c3c'])
            ax2.set_ylim(0, 1)
            ax2.set_ylabel('Average Proportion of Units Remaining')
            ax2.set_title('Unit Survival Rate')
            
            # Add value labels on top of bars
            for i, v in enumerate(values):
                ax2.text(i, v + 0.02, f'{v:.2f}', ha='center')
            
            # Add overall title
            plt.suptitle(f'Self-Play Statistics ({num_battles} battles)')
            
            # Save figure
            plt.tight_layout()
            fig.subplots_adjust(top=0.88)  # Make room for suptitle
            
            stats_file = f"{self.visualization_dir}/battle_statistics.png"
            plt.savefig(stats_file)
            plt.close()
            
            print(f"Battle statistics visualization saved to {stats_file}")
            
        except Exception as e:
            print(f"Error creating statistics visualization: {e}")


def run_self_play_demo(model_path='models/best_battle_predictor.pt', num_battles=1, 
                 max_enemies=3, terrain_type=None, weather_type=None,
                 enable_artillery=False, enable_stealth=False):
    """
    Run a self-play demonstration
    
    Parameters:
        model_path: Path to the trained model
        num_battles: Number of battles to run
        max_enemies: Maximum number of enemies per battle
        terrain_type: Specific terrain type to use (None for random)
        weather_type: Specific weather type to use (None for random)
        enable_artillery: Whether to enable artillery units
        enable_stealth: Whether to enable stealth units
    """
    # Create the self-play simulator with custom parameters
    simulator = SelfPlaySimulation(
        model_path=model_path, 
        max_steps=50,
        save_visualizations=True,
        max_enemies=max_enemies,
        terrain_type=terrain_type,
        weather_type=weather_type,
        enable_artillery=enable_artillery,
        enable_stealth=enable_stealth
    )
    
    # Run the specified number of battles
    if num_battles == 1:
        # Run a single battle with detailed logging
        simulator.run_self_play_battle(log_steps=True, render_battle=True)
    else:
        # Run multiple battles with summary statistics
        simulator.run_multiple_battles(num_battles=num_battles, log_individual=False)


if __name__ == "__main__":
    # If run directly, perform a self-play demonstration
    import argparse
    
    parser = argparse.ArgumentParser(description='Self-Play Battlefield Simulation')
    parser.add_argument('--model', type=str, default='models/best_battle_predictor.pt',
                       help='Path to trained model')
    parser.add_argument('--battles', type=int, default=1,
                       help='Number of battles to run')
    
    args = parser.parse_args()
    
    run_self_play_demo(model_path=args.model, num_battles=args.battles)