"""
Self-Play Module for Battlefield Simulation with Reinforcement Learning

This module implements functionality for the LSTM model to play against itself
by controlling both friendly and enemy units in the battlefield, with reinforcement
learning capabilities to improve the model through self-play iterations.
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from torch.utils.data import DataLoader, TensorDataset
import time
import json

# Try importing from both module styles to be flexible
try:
    # Direct imports
    from battlefield_env import BattlefieldEnv, UnitType, ActionType, TERRAIN_TYPES, WEATHER_CONDITIONS, Enemy
    from lstm_model import load_model, predict_battle_outcome, LSTMBattlePredictor
    from battle_strategy import get_optimal_actions, get_optimal_positioning
except ImportError:
    # Fallback to src directory structure
    from src.battlefield_env import BattlefieldEnv, UnitType, ActionType, TERRAIN_TYPES, WEATHER_CONDITIONS, Enemy
    from src.lstm_model import load_model, predict_battle_outcome, LSTMBattlePredictor
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
            if model_path:
                self.model = load_model(model_path)
                print(f"✓ Model loaded successfully from {model_path}")
            else:
                self.model = None  # Will be set externally
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
    
    def run_self_play_battle_with_memory(self, log_steps=False):
        """
        Run a single self-play battle and collect experiences for reinforcement learning
        
        Parameters:
            log_steps: Whether to log step information
            
        Returns:
            experiences: List of (state, action, reward, next_state, done) tuples
        """
        # Initialize environment and state
        self.env = self._create_custom_environment()
        obs = self.env.reset()
        
        experiences = []  # List to store experience tuples
        done = False
        step = 0
        cumulative_reward = 0
        
        while not done and step < self.max_steps:
            # Get current state representation
            current_state = self._extract_state_features(obs)
            
            # Get blue actions (friendly units)
            blue_actions = self._get_model_actions_for_blue()
            
            # Execute blue actions
            next_obs, reward, done, info = self.env.step(blue_actions)
            
            # Update cumulative reward
            cumulative_reward += reward
            
            # Get next state representation
            next_state = self._extract_state_features(next_obs)
            
            # Store experience tuple
            experiences.append((
                current_state, 
                blue_actions, 
                reward,         # Immediate reward
                next_state, 
                done
            ))
            
            # If battle ended, break
            if done:
                break
            
            # Update current observation
            obs = next_obs
            
            # For enemy's turn, we'll use the same approach but add negative rewards
            # This simplifies training as we're focusing on the Blue side's perspective
            
            # Get enemy actions
            red_actions = self._get_model_actions_for_red()
            
            # Store the state before enemy actions
            current_state = self._extract_state_features(obs)
            
            # Execute enemy actions manually
            for i, (enemy, action) in enumerate(zip(self.env.enemies, red_actions)):
                if enemy.is_alive():
                    action_type = ActionType(action)
                    target = self._select_target_for_enemy(enemy, action_type)
                    enemy.take_action(action_type, self.env, target)
            
            # Calculate "reward" from enemy actions (negative for any damage to friendly units)
            red_reward = -self._calculate_damage_to_friendly_units()
            
            # Check battle end after enemy actions
            battle_ended = self._check_battle_end()
            if battle_ended:
                done = True
                result = self._get_battle_result()
                # Additional reward for win/loss after enemy turn
                if result['result'] == 'blue_victory':
                    red_reward += 10  # Big reward for victory
                elif result['result'] == 'red_victory':
                    red_reward -= 10  # Penalty for defeat
            
            # Get new state after enemy actions
            next_state = self._extract_state_features(obs)
            
            # Store experience tuple for enemy's turn
            experiences.append((
                current_state,
                blue_actions,  # Same actions (not used for update, just to maintain tuple structure)
                red_reward,    # Reward from enemy's perspective
                next_state,
                done
            ))
            
            # Check if battle ended after enemy actions
            if done:
                break
            
            step += 1
        
        # Add the final outcome reward to the last experience
        if len(experiences) > 0:
            result = self._get_battle_result()
            last_state, last_action, last_reward, last_next_state, _ = experiences[-1]
            
            # Determine final reward based on battle outcome
            final_reward = last_reward
            if result['result'] == 'blue_victory':
                final_reward += 10  # Big reward for victory
            elif result['result'] == 'red_victory':
                final_reward -= 10  # Penalty for defeat
            
            # Replace the last experience with updated reward
            experiences[-1] = (last_state, last_action, final_reward, last_next_state, True)
        
        return experiences
    
    def _calculate_damage_to_friendly_units(self):
        """
        Calculate the damage inflicted to friendly units in the last step
        Used to compute rewards for reinforcement learning
        
        Returns:
            damage: Total damage to friendly units (higher is worse)
        """
        # In a real implementation, you would track unit health before and after
        # enemy actions to calculate this precisely. For simplicity, we'll use
        # a rough heuristic based on current health percentages.
        
        total_health_percent = sum(
            unit.hp / unit.max_hp for unit in self.env.friendly_units if unit.is_alive()
        )
        
        # Normalize by number of units (to get average health percentage)
        num_alive = sum(1 for unit in self.env.friendly_units if unit.is_alive())
        if num_alive == 0:
            return -10  # Heavy penalty if all units are dead
        
        avg_health = total_health_percent / num_alive
        
        # Convert to a damage measure (higher damage = lower health)
        # Scale to be between -5 (all full health) and 0 (all low health)
        damage_measure = -5 * avg_health
        
        return damage_measure
    
    def _extract_state_features(self, observation):
        """
        Extract features from the current battlefield state
        For RL, we need a fixed-size representation
        
        Parameters:
            observation: Current battlefield observation
            
        Returns:
            features: Array of state features
        """
        # Create a fixed-size feature vector from the current battlefield state
        # This is a simplified version - you'd want to include more features in a real implementation
        
        # Initialize feature vector with zeros
        # We'll include:
        # - Unit positions and health (3 units x 3 features)
        # - Enemy positions and health (up to 5 enemies x 3 features)
        # - Terrain and weather encodings
        feature_size = 9 + 15 + 10  # 34 features total
        features = np.zeros(feature_size)
        
        # Add friendly unit features
        for i, unit in enumerate(self.env.friendly_units):
            if i < 3 and unit.is_alive():  # Limit to 3 units
                base_idx = i * 3
                features[base_idx] = unit.position[0] / 10.0  # Normalize X position
                features[base_idx + 1] = unit.position[1] / 10.0  # Normalize Y position
                features[base_idx + 2] = unit.hp / unit.max_hp  # Health percentage
        
        # Add enemy unit features
        for i, enemy in enumerate(self.env.enemies):
            if i < 5 and enemy.is_alive():  # Limit to 5 enemies
                base_idx = 9 + i * 3
                features[base_idx] = enemy.position[0] / 10.0  # Normalize X position
                features[base_idx + 1] = enemy.position[1] / 10.0  # Normalize Y position
                features[base_idx + 2] = enemy.hp / enemy.max_hp  # Health percentage
        
        # Add terrain encoding (one-hot)
        terrain_idx = list(TERRAIN_TYPES.keys()).index(self.env.current_terrain)
        features[24 + terrain_idx] = 1.0
        
        # Add weather encoding (one-hot)
        weather_idx = list(WEATHER_CONDITIONS.keys()).index(self.env.current_weather)
        features[29 + weather_idx] = 1.0
        
        return features
        
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


class SelfPlayRL:
    """
    Reinforcement Learning framework for self-play improvement
    
    This class implements reinforcement learning techniques to improve
    the battle prediction model through repeated self-play.
    """
    def __init__(self, model_path='models/best_battle_predictor.pt', learning_rate=0.0001):
        """
        Initialize the reinforcement learning system
        
        Parameters:
            model_path: Path to the initial model
            learning_rate: Learning rate for model updates
        """
        # Load the model
        self.model = load_model(model_path)
        self.model_path = model_path
        
        # Set up optimizer for RL updates
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.loss_fn = nn.BCELoss()  # Binary Cross Entropy for win/loss prediction
        
        # Experience replay buffer
        self.replay_buffer = []
        self.buffer_size = 10000  # Maximum size of replay buffer
        
        # Create directories for saving progress
        os.makedirs('models/rl_checkpoints', exist_ok=True)
        os.makedirs('visualizations/rl_training', exist_ok=True)
        
        # Training statistics
        self.training_stats = {
            'cycle_rewards': [],
            'validation_scores': [],
            'losses': [],
            'win_rates': []
        }
    
    def collect_experience(self, num_games=100, save_visualizations=False, max_enemies=3):
        """
        Play games and collect experience for training
        
        Parameters:
            num_games: Number of self-play games to run
            save_visualizations: Whether to save battle visualizations
            max_enemies: Maximum number of enemies per battle
            
        Returns:
            total_experiences: Total number of experiences collected
        """
        print(f"Collecting experiences from {num_games} self-play games...")
        
        # Create self-play simulator with the current model
        simulator = SelfPlaySimulation(
            model_path=None,  # Don't load from file
            save_visualizations=save_visualizations,
            max_enemies=max_enemies
        )
        simulator.model = self.model  # Pass model directly
        
        total_experiences = 0
        
        for i in range(num_games):
            print(f"Game {i+1}/{num_games}", end="\r")
            
            # Run a game and collect experiences
            experiences = simulator.run_self_play_battle_with_memory(log_steps=False)
            total_experiences += len(experiences)
            
            # Add experiences to replay buffer
            self.replay_buffer.extend(experiences)
            
            # Manage buffer size
            if len(self.replay_buffer) > self.buffer_size:
                # Keep only the most recent experiences
                self.replay_buffer = self.replay_buffer[-self.buffer_size:]
        
        print(f"\nCollected {total_experiences} experiences from {num_games} games.")
        print(f"Replay buffer size: {len(self.replay_buffer)} experiences")
        
        return total_experiences
    
    def train_on_experiences(self, batch_size=32, epochs=5):
        """
        Train model on collected experiences using replay buffer
        
        Parameters:
            batch_size: Batch size for training
            epochs: Number of training epochs
            
        Returns:
            avg_loss: Average loss across all epochs
        """
        if len(self.replay_buffer) < batch_size:
            print("Not enough experiences to train. Run collect_experience first.")
            return 0
            
        print(f"Training on replay buffer with {len(self.replay_buffer)} experiences...")
        
        # Extract data from replay buffer
        states = []
        rewards = []
        
        for state, _, reward, _, _ in self.replay_buffer:
            # We use the immediate rewards rather than future discounted rewards
            # This simplifies the RL problem for battlefield prediction
            states.append(state)
            rewards.append(max(min(reward, 1.0), 0.0))  # Clamp between 0 and 1
        
        # Get input size from model
        input_size = None
        for param in self.model.parameters():
            if len(param.shape) >= 2:
                # First layer's weight matrix should have shape [hidden_size, input_size]
                input_size = param.shape[1]
                break
        
        if input_size is None:
            input_size = 43  # Default if we can't determine from model
        
        # Pad state vectors to match expected input size
        padded_states = []
        for state in states:
            current_size = len(state)
            if current_size < input_size:
                # Convert to list if it's not already
                if isinstance(state, np.ndarray):
                    state_list = state.tolist()
                else:
                    state_list = list(state)
                    
                # Pad with zeros
                padded_state = state_list + [0.0] * (input_size - current_size)
                padded_states.append(padded_state)
            elif current_size > input_size:
                # Truncate if too large
                if isinstance(state, np.ndarray):
                    padded_states.append(state[:input_size])
                else:
                    padded_states.append(state[:input_size])
            else:
                # Already correct size
                padded_states.append(state)
        
        # Convert to tensors
        states_tensor = torch.FloatTensor(padded_states).unsqueeze(1)  # Add time dimension for LSTM
        rewards_tensor = torch.FloatTensor(rewards).unsqueeze(1)
        
        # Create dataset and dataloader
        dataset = TensorDataset(states_tensor, rewards_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Training loop
        self.model.train()
        epoch_losses = []
        
        for epoch in range(epochs):
            total_loss = 0
            batches = 0
            
            for batch_states, batch_rewards in dataloader:
                # Forward pass
                predictions = self.model(batch_states)
                
                # Compute loss
                loss = self.loss_fn(predictions, batch_rewards)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                
                # Apply gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                total_loss += loss.item()
                batches += 1
            
            # Calculate average loss for this epoch
            avg_epoch_loss = total_loss / max(batches, 1)
            epoch_losses.append(avg_epoch_loss)
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_epoch_loss:.6f}")
        
        # Calculate average loss across all epochs
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        
        # Update training statistics
        self.training_stats['losses'].append(avg_loss)
        
        return avg_loss
    
    def validate_model(self, num_games=20, max_enemies=3):
        """
        Validate the model by testing it in self-play
        
        Parameters:
            num_games: Number of validation games to run
            max_enemies: Maximum number of enemies per battle
            
        Returns:
            validation_score: Average reward across validation games
        """
        print(f"Validating model on {num_games} games...")
        
        # Create self-play simulator with the current model
        simulator = SelfPlaySimulation(
            model_path=None,
            save_visualizations=False,
            max_enemies=max_enemies
        )
        simulator.model = self.model
        
        # Run validation games
        results = []
        total_reward = 0
        
        for i in range(num_games):
            print(f"Validation game {i+1}/{num_games}", end="\r")
            
            # Run a game and get the result
            result = simulator.run_self_play_battle(log_steps=False, render_battle=False)
            results.append(result)
            
            # Calculate reward for this game
            if result['result'] == 'blue_victory':
                reward = 1.0
            elif result['result'] == 'blue_advantage':
                reward = 0.7
            elif result['result'] == 'draw':
                reward = 0.5
            elif result['result'] == 'red_advantage':
                reward = 0.3
            else:  # red_victory
                reward = 0.0
                
            total_reward += reward
        
        # Calculate win rate
        blue_wins = sum(1 for r in results if r['result'] in ['blue_victory', 'blue_advantage'])
        win_rate = blue_wins / num_games * 100
        
        # Calculate validation score (average reward)
        validation_score = total_reward / num_games
        
        print(f"\nValidation results:")
        print(f"Average reward: {validation_score:.4f}")
        print(f"Blue win rate: {win_rate:.1f}%")
        
        # Update training statistics
        self.training_stats['validation_scores'].append(validation_score)
        self.training_stats['win_rates'].append(win_rate)
        
        return validation_score
    
    def compare_models(self, old_model_path, new_model, num_games=20):
        """
        Compare new model against old model in head-to-head battles
        
        Parameters:
            old_model_path: Path to the old model
            new_model: The new model to evaluate
            num_games: Number of comparison games to run
            
        Returns:
            win_rate: Percentage of games won by new model
        """
        print(f"Comparing new model against baseline in {num_games} head-to-head battles...")
        
        # Load old model
        old_model = load_model(old_model_path)
        
        # Create environment for battles
        env = BattlefieldEnv()
        
        # Run head-to-head battles
        new_model_wins = 0
        draws = 0
        
        for i in range(num_games):
            print(f"Battle {i+1}/{num_games}", end="\r")
            
            # Reset environment
            env.reset()
            
            # Run battle with new model controlling blue and old model controlling red
            done = False
            step = 0
            
            while not done and step < 100:
                # Get actions for blue units using new model
                blue_actions = self._get_model_actions(new_model, env, is_blue=True)
                
                # Execute blue actions
                obs, reward, done, info = env.step(blue_actions)
                
                if done:
                    break
                
                # Get actions for red units using old model
                red_actions = self._get_model_actions(old_model, env, is_blue=False)
                
                # Execute red actions manually
                for i, (enemy, action) in enumerate(zip(env.enemies, red_actions)):
                    if enemy.is_alive():
                        action_type = ActionType(action)
                        target = self._select_target(env, enemy, action_type)
                        enemy.take_action(action_type, env, target)
                
                # Check if battle ended after red's moves
                blue_alive = sum(1 for unit in env.friendly_units if unit.is_alive())
                red_alive = sum(1 for enemy in env.enemies if enemy.is_alive())
                
                if blue_alive == 0 or red_alive == 0:
                    done = True
                    
                step += 1
            
            # Determine winner
            blue_alive = sum(1 for unit in env.friendly_units if unit.is_alive())
            red_alive = sum(1 for enemy in env.enemies if enemy.is_alive())
            
            if blue_alive > 0 and red_alive == 0:
                # New model (blue) won
                new_model_wins += 1
            elif blue_alive == 0 and red_alive > 0:
                # Old model (red) won
                pass
            else:
                # Draw or timeout
                draws += 1
        
        # Calculate win rate
        win_rate = new_model_wins / num_games * 100
        draw_rate = draws / num_games * 100
        
        print(f"\nHead-to-head results:")
        print(f"New model win rate: {win_rate:.1f}%")
        print(f"Draw rate: {draw_rate:.1f}%")
        print(f"Old model win rate: {100 - win_rate - draw_rate:.1f}%")
        
        return win_rate
    
    def _get_model_actions(self, model, env, is_blue=True):
        """
        Get actions for units using the specified model
        
        Parameters:
            model: The model to use for decision making
            env: The battlefield environment
            is_blue: Whether we're controlling blue or red side
            
        Returns:
            actions: List of action indices
        """
        if is_blue:
            # Get blue unit positions
            unit_positions = {
                'infantry': env.friendly_units[0].position if env.friendly_units[0].is_alive() else [0, 0],
                'tank': env.friendly_units[1].position if env.friendly_units[1].is_alive() else [0, 0],
                'drone': env.friendly_units[2].position if env.friendly_units[2].is_alive() else [0, 0]
            }
            
            # Find nearest enemy
            nearest_enemy = None
            nearest_distance = float('inf')
            
            for enemy in env.enemies:
                if enemy.is_alive():
                    # Calculate distance to all friendly units
                    total_distance = 0
                    alive_units = 0
                    for unit in env.friendly_units:
                        if unit.is_alive():
                            dx = unit.position[0] - enemy.position[0]
                            dy = unit.position[1] - enemy.position[1]
                            total_distance += (dx*dx + dy*dy)**0.5
                            alive_units += 1
                    
                    if alive_units > 0:
                        avg_distance = total_distance / alive_units
                        
                        if avg_distance < nearest_distance:
                            nearest_distance = avg_distance
                            nearest_enemy = enemy
            
            # If no enemies alive, use default actions
            if nearest_enemy is None:
                return [0, 0, 0]  # Default to 'move' for all units
            
            # Get optimal actions
            try:
                best_actions, _ = get_optimal_actions(model, unit_positions, nearest_enemy.position)
                return best_actions
            except:
                return [0, 0, 0]  # Default to 'move' for all units
        else:
            # Get actions for red units (enemies)
            actions = []
            
            for enemy in env.enemies:
                if not enemy.is_alive():
                    actions.append(0)  # Default action for dead units
                    continue
                
                # Find nearest friendly unit
                nearest_friendly = None
                nearest_distance = float('inf')
                
                for unit in env.friendly_units:
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
                
                # Use the model to determine action
                try:
                    # Simplified state for the model (enemy perspective)
                    unit_positions = {
                        'infantry': enemy.position,
                        'tank': enemy.position,  # Use same position for simplicity
                        'drone': enemy.position  # Use same position for simplicity
                    }
                    
                    best_actions, _ = get_optimal_actions(model, unit_positions, nearest_friendly.position)
                    actions.append(best_actions[0])
                except:
                    # Fallback to simple heuristic
                    if nearest_distance < 3:
                        actions.append(1)  # Attack
                    else:
                        actions.append(0)  # Move
            
            return actions
    
    def _select_target(self, env, unit, action_type):
        """
        Select an appropriate target for a unit's action
        
        Parameters:
            env: The battlefield environment
            unit: The unit taking action
            action_type: The ActionType enum value
            
        Returns:
            target: The target unit or None
        """
        is_enemy = unit in env.enemies
        
        # If not an action that needs a target
        if action_type not in [ActionType.ATTACK, ActionType.SUPPORT]:
            return None
        
        # Find appropriate targets
        potential_targets = []
        
        if action_type == ActionType.ATTACK:
            # Target units on opposite side
            target_units = env.friendly_units if is_enemy else env.enemies
            
            for target in target_units:
                if target.is_alive():
                    # Calculate distance
                    dx = unit.position[0] - target.position[0]
                    dy = unit.position[1] - target.position[1]
                    distance = (dx*dx + dy*dy)**0.5
                    
                    # Score based on distance and unit health
                    score = (10 - min(distance, 10)) + (1 - target.hp/target.max_hp) * 5
                    potential_targets.append((target, score))
        else:  # SUPPORT
            # Target units on same side
            target_units = env.enemies if is_enemy else env.friendly_units
            
            for target in target_units:
                if target.is_alive() and target != unit:
                    # Calculate distance
                    dx = unit.position[0] - target.position[0]
                    dy = unit.position[1] - target.position[1]
                    distance = (dx*dx + dy*dy)**0.5
                    
                    # Score based on distance and unit health
                    score = (10 - min(distance, 10)) + (1 - target.hp/target.max_hp) * 5
                    potential_targets.append((target, score))
        
        # Sort by score (higher is better)
        potential_targets.sort(key=lambda x: x[1], reverse=True)
        
        # Return the best target or None if no targets
        return potential_targets[0][0] if potential_targets else None
    
    def run_rl_improvement_cycle(self, cycles=10, games_per_cycle=50, 
                                validation_games=20, max_enemies=3,
                                save_visualizations=False):
        """
        Run multiple cycles of reinforcement learning improvement
        
        Parameters:
            cycles: Number of RL improvement cycles to run
            games_per_cycle: Number of self-play games per cycle
            validation_games: Number of validation games to run
            max_enemies: Maximum number of enemies per battle
            save_visualizations: Whether to save battle visualizations
            
        Returns:
            training_stats: Dictionary with training statistics
        """
        print(f"Starting reinforcement learning improvement process with {cycles} cycles...")
        print(f"Each cycle will play {games_per_cycle} games and train on the experiences.")
        
        # Store initial model copy for later comparison
        initial_model_path = 'models/rl_checkpoints/initial_model.pt'
        torch.save(self.model.state_dict(), initial_model_path)
        
        # Reset training statistics
        self.training_stats = {
            'cycle_rewards': [],
            'validation_scores': [],
            'losses': [],
            'win_rates': []
        }
        
        best_validation = 0
        best_cycle = -1
        
        # Create timestamp for this training run
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        
        for cycle in range(cycles):
            print(f"\n{'='*60}")
            print(f"Reinforcement Learning Cycle {cycle+1}/{cycles}")
            print(f"{'='*60}")
            
            # Collect experiences through self-play
            self.collect_experience(
                num_games=games_per_cycle,
                save_visualizations=save_visualizations,
                max_enemies=max_enemies
            )
            
            # Train on collected experiences
            loss = self.train_on_experiences(epochs=5)
            
            # Validate the improved model
            validation_score = self.validate_model(
                num_games=validation_games,
                max_enemies=max_enemies
            )
            
            # Store cycle reward for tracking progress
            self.training_stats['cycle_rewards'].append(validation_score)
            
            # Save checkpoint of this cycle's model
            cycle_model_path = f'models/rl_checkpoints/model_cycle_{cycle+1}.pt'
            torch.save(self.model.state_dict(), cycle_model_path)
            
            # Check if this is the best model so far
            if validation_score > best_validation:
                best_validation = validation_score
                best_cycle = cycle
                print(f"New best model! Validation score: {validation_score:.4f}")
                
                # Save as the best model
                torch.save(self.model.state_dict(), 'models/rl_improved_model.pt')
                
            print(f"Cycle {cycle+1} complete. Validation score: {validation_score:.4f}")
            
            # Visualize training progress
            self._visualize_training_progress(cycle+1, timestamp)
        
        # Final validation against the initial model
        print("\nFinal validation: comparing best model against initial model...")
        
        # Load the best model
        if best_cycle >= 0:
            best_model_path = f'models/rl_checkpoints/model_cycle_{best_cycle+1}.pt'
            self.model.load_state_dict(torch.load(best_model_path))
            
            # Compare against initial model
            win_rate = self.compare_models(
                initial_model_path,
                self.model,
                num_games=validation_games
            )
            
            print(f"Best model (from cycle {best_cycle+1}) win rate against initial model: {win_rate:.1f}%")
            
            if win_rate > 60:  # Significant improvement
                print("Significant improvement detected! Updating the main model.")
                # Save as the main model
                torch.save(self.model.state_dict(), self.model_path)
            else:
                print("Improvement not significant enough to update the main model.")
        else:
            print("No improved model found during training.")
        
        # Save full training history
        self._save_training_history(timestamp)
        
        return self.training_stats
    
    def _visualize_training_progress(self, current_cycle, timestamp):
        """
        Visualize the training progress so far
        
        Parameters:
            current_cycle: Current cycle number
            timestamp: Timestamp for this training run
        """
        # Create figure with multiple subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Reinforcement Learning Training Progress (Cycle {current_cycle})', fontsize=16)
        
        # Plot 1: Validation scores over cycles
        axes[0, 0].plot(range(1, current_cycle+1), self.training_stats['validation_scores'], 'b-o')
        axes[0, 0].set_title('Validation Scores')
        axes[0, 0].set_xlabel('Cycle')
        axes[0, 0].set_ylabel('Validation Score')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Win rates over cycles
        axes[0, 1].plot(range(1, current_cycle+1), self.training_stats['win_rates'], 'g-o')
        axes[0, 1].set_title('Win Rates')
        axes[0, 1].set_xlabel('Cycle')
        axes[0, 1].set_ylabel('Win Rate (%)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Training losses
        axes[1, 0].plot(range(1, current_cycle+1), self.training_stats['losses'], 'r-o')
        axes[1, 0].set_title('Training Losses')
        axes[1, 0].set_xlabel('Cycle')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Best validation score visualization
        if self.training_stats['validation_scores']:
            best_cycle = np.argmax(self.training_stats['validation_scores'])
            best_score = self.training_stats['validation_scores'][best_cycle]
            
            all_cycles = list(range(1, current_cycle+1))
            
            # Highlight the best cycle
            bars = axes[1, 1].bar(all_cycles, self.training_stats['validation_scores'])
            bars[best_cycle].set_color('gold')
            
            axes[1, 1].set_title(f'Best Model: Cycle {best_cycle+1} (Score: {best_score:.4f})')
            axes[1, 1].set_xlabel('Cycle')
            axes[1, 1].set_ylabel('Validation Score')
            axes[1, 1].grid(True, alpha=0.3)
        
        # Adjust layout and save
        plt.tight_layout()
        fig.subplots_adjust(top=0.9)  # Make room for suptitle
        
        # Save visualization
        viz_path = f'visualizations/rl_training/progress_{timestamp}_cycle_{current_cycle}.png'
        plt.savefig(viz_path)
        plt.close()
        
        print(f"Training progress visualization saved to {viz_path}")
    
    def _save_training_history(self, timestamp):
        """
        Save the full training history to a JSON file
        
        Parameters:
            timestamp: Timestamp for this training run
        """
        # Convert training stats to serializable format
        history = {
            'timestamp': timestamp,
            'cycles_completed': len(self.training_stats['validation_scores']),
            'validation_scores': self.training_stats['validation_scores'],
            'win_rates': self.training_stats['win_rates'],
            'losses': self.training_stats['losses'],
            'cycle_rewards': self.training_stats['cycle_rewards']
        }
        
        # Add best model information
        if self.training_stats['validation_scores']:
            best_idx = np.argmax(self.training_stats['validation_scores'])
            history['best_cycle'] = int(best_idx) + 1
            history['best_validation_score'] = float(self.training_stats['validation_scores'][best_idx])
            history['best_win_rate'] = float(self.training_stats['win_rates'][best_idx])
        
        # Save to file
        history_path = f'visualizations/rl_training/history_{timestamp}.json'
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=4)
        
        print(f"Training history saved to {history_path}")


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


def run_reinforcement_learning(model_path='models/best_battle_predictor.pt', cycles=5, 
                              games_per_cycle=20, validation_games=10, max_enemies=3,
                              save_visualizations=True):
    """
    Run reinforcement learning improvement on the model
    
    Parameters:
        model_path: Path to the initial model
        cycles: Number of improvement cycles to run
        games_per_cycle: Number of games to play in each cycle
        validation_games: Number of games for validation
        max_enemies: Maximum number of enemies per battle
        save_visualizations: Whether to save battle visualizations
    """
    # Create the reinforcement learning system
    rl_system = SelfPlayRL(model_path=model_path)
    
    # Run the improvement cycles
    stats = rl_system.run_rl_improvement_cycle(
        cycles=cycles,
        games_per_cycle=games_per_cycle,
        validation_games=validation_games,
        max_enemies=max_enemies,
        save_visualizations=save_visualizations
    )
    
    # Print final results
    if stats['validation_scores']:
        best_idx = np.argmax(stats['validation_scores'])
        best_cycle = best_idx + 1
        best_score = stats['validation_scores'][best_idx]
        best_win_rate = stats['win_rates'][best_idx]
        
        print("\nReinforcement Learning Summary:")
        print(f"Best model found at cycle {best_cycle} with:")
        print(f"- Validation score: {best_score:.4f}")
        print(f"- Win rate: {best_win_rate:.1f}%")
        
        # Provide information about the saved models
        print("\nSaved models:")
        print(f"- Initial model: models/rl_checkpoints/initial_model.pt")
        print(f"- Best model from cycle {best_cycle}: models/rl_checkpoints/model_cycle_{best_cycle}.pt")
        print(f"- Best overall model: models/rl_improved_model.pt")
        
        # If significant improvement, the main model was updated
        if best_win_rate > 60:
            print(f"- Main model (updated): {model_path}")
    else:
        print("No training cycles completed.")


if __name__ == "__main__":
    # If run directly, perform a self-play demonstration or reinforcement learning
    import argparse
    
    parser = argparse.ArgumentParser(description='Self-Play Battlefield Simulation with Reinforcement Learning')
    parser.add_argument('--model', type=str, default='models/best_battle_predictor.pt',
                       help='Path to trained model')
    parser.add_argument('--mode', type=str, choices=['demo', 'rl'], default='demo',
                       help='Mode: demo (self-play) or rl (reinforcement learning)')
    parser.add_argument('--battles', type=int, default=1,
                       help='Number of battles to run in demo mode')
    parser.add_argument('--cycles', type=int, default=5,
                       help='Number of RL cycles to run in rl mode')
    parser.add_argument('--games', type=int, default=20,
                       help='Number of games per cycle in rl mode')
    
    args = parser.parse_args()
    
    if args.mode == 'demo':
        run_self_play_demo(model_path=args.model, num_battles=args.battles)
    else:  # rl mode
        run_reinforcement_learning(
            model_path=args.model,
            cycles=args.cycles,
            games_per_cycle=args.games
        )