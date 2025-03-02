"""
Advanced Battlefield Environment Module

Contains an enhanced BattlefieldEnv class with improved tactical mechanics:
- Advanced battle success rules
- Unit-specific attack strategies
- Type-based advantages/disadvantages
- Multi-enemy support
- Terrain and weather effects
- Line of sight and stealth mechanics
"""
import os
import pandas as pd
import numpy as np
import random
import heapq
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from enum import Enum
import gym
from gym import spaces
from typing import Dict, List, Tuple, Set, Optional, Union

# Constants for grid size and distance scaling
GRID_SIZE = 10
GRID_CELL_SIZE = 1000  # Each cell is 1000m x 1000m (1 km²)
TOTAL_AREA = GRID_SIZE * GRID_SIZE * GRID_CELL_SIZE  # 100,000,000 m²

# Define possible battlefield conditions with combat modifiers
WEATHER_CONDITIONS = {
    "Clear": {"visibility": 1.0, "movement": 1.0, "attack": 1.0},
    "Foggy": {"visibility": 0.5, "movement": 0.8, "attack": 0.8},
    "Rainy": {"visibility": 0.7, "movement": 0.7, "attack": 0.9},
    "Stormy": {"visibility": 0.4, "movement": 0.5, "attack": 0.7},
    "Snowy": {"visibility": 0.6, "movement": 0.6, "attack": 0.8}
}

TERRAIN_TYPES = {
    "Plains": {"infantry_move": 1.0, "tank_move": 1.0, "drone_move": 1.0, "cover": 0.0},
    "Mountains": {"infantry_move": 0.7, "tank_move": 0.3, "drone_move": 0.8, "cover": 0.4},
    "Forest": {"infantry_move": 0.8, "tank_move": 0.6, "drone_move": 0.9, "cover": 0.3},
    "Urban": {"infantry_move": 0.9, "tank_move": 0.7, "drone_move": 0.7, "cover": 0.5},
    "Desert": {"infantry_move": 0.8, "tank_move": 0.9, "drone_move": 1.0, "cover": 0.1}
}

# Unit types for type advantage system
class UnitType(Enum):
    INFANTRY = 0
    ARMORED = 1
    AERIAL = 2
    ARTILLERY = 3
    STEALTH = 4

# Action types
class ActionType(Enum):
    MOVE = 0      # Standard movement
    ATTACK = 1    # Direct attack
    DEFEND = 2    # Defensive posture (reduced damage taken)
    RETREAT = 3   # Move away from enemies
    SUPPORT = 4   # Enhance allied units' capabilities

# Type advantage matrix [attacker][defender]
# Each value represents damage multiplier
TYPE_ADVANTAGES = [
    # Infantry, Armored, Aerial, Artillery, Stealth
    [1.0, 0.6, 0.3, 0.8, 1.2],  # Infantry attacking...
    [1.5, 1.0, 0.4, 1.2, 0.9],  # Armored attacking...
    [1.2, 0.7, 1.0, 1.5, 0.5],  # Aerial attacking...
    [1.7, 1.4, 0.9, 1.0, 0.8],  # Artillery attacking...
    [1.3, 0.8, 0.7, 1.0, 1.0]   # Stealth attacking...
]

# Attack range based on unit type (in grid cells)
ATTACK_RANGES = {
    UnitType.INFANTRY: 1,
    UnitType.ARMORED: 1,
    UnitType.AERIAL: 2,
    UnitType.ARTILLERY: 3,
    UnitType.STEALTH: 1
}

# A* Pathfinding Algorithm with terrain consideration
def astar(start, goal, grid_size, obstacles, terrain_cost, unit_type):
    """
    Enhanced A* pathfinding algorithm with terrain costs.
    
    Parameters:
        start: Tuple (x,y) of starting position
        goal: Tuple (x,y) of goal position
        grid_size: Size of the grid
        obstacles: Set of obstacle positions as tuples
        terrain_cost: 2D grid of terrain movement costs
        unit_type: UnitType enum for unit-specific movement costs
        
    Returns:
        List of positions forming the path
    """
    def heuristic(a, b):
        return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)  # Euclidean distance

    open_list = []
    heapq.heappush(open_list, (0, start))
    came_from = {}
    cost_so_far = {start: 0}

    # Movement variations including diagonals
    moves = [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]
    
    # Get terrain multiplier based on unit type
    def get_terrain_multiplier(pos):
        # Default cost if no terrain data
        if terrain_cost is None:
            return 1.0
            
        x, y = pos
        terrain_value = terrain_cost[x][y]
        
        # Unit-specific terrain costs
        if unit_type == UnitType.INFANTRY:
            return terrain_value.get('infantry_move', 1.0)
        elif unit_type == UnitType.ARMORED:
            return terrain_value.get('tank_move', 1.0)
        elif unit_type == UnitType.AERIAL:
            return terrain_value.get('drone_move', 1.0)
        elif unit_type == UnitType.ARTILLERY:
            return terrain_value.get('tank_move', 1.0)  # Use tank costs
        elif unit_type == UnitType.STEALTH:
            return terrain_value.get('infantry_move', 1.0)  # Use infantry costs
        return 1.0

    while open_list:
        _, current = heapq.heappop(open_list)

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)  # Add the start position
            path.reverse()
            return path

        for dx, dy in moves:
            next_node = (current[0] + dx, current[1] + dy)
            # Check if next node is within grid bounds and not an obstacle
            if (0 <= next_node[0] < grid_size and 0 <= next_node[1] < grid_size 
                and next_node not in obstacles):
                
                # Calculate movement cost with terrain consideration
                # Diagonal moves cost more (√2 times straight moves)
                move_cost = 1.414 if (dx != 0 and dy != 0) else 1.0
                
                # Apply terrain multiplier (higher = more difficult)
                terrain_multiplier = get_terrain_multiplier(next_node)
                move_cost *= terrain_multiplier
                
                new_cost = cost_so_far[current] + move_cost
                
                if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                    cost_so_far[next_node] = new_cost
                    priority = new_cost + heuristic(goal, next_node)
                    heapq.heappush(open_list, (priority, next_node))
                    came_from[next_node] = current

    # If no path found, try to return partial path toward goal
    if start in came_from:  # Sanity check
        path = []
        current = start
        while current in came_from:
            path.append(current)
            current = came_from[current]
        return path
    return [start]  # If nothing else, stay in place


class Unit:
    """Enhanced Unit class with detailed attributes and behaviors"""
    
    def __init__(self, unit_type, position, hp, attack_power, speed, 
                 detection_range=2, stealth_level=0):
        self.unit_type = unit_type
        self.position = list(position)
        self.max_hp = hp
        self.hp = hp
        self.attack_power = attack_power
        self.speed = speed
        self.detection_range = detection_range
        self.stealth_level = stealth_level
        self.action_state = ActionType.MOVE
        self.cooldown = 0  # Turns until next action
        self.target = None  # Current target (if any)
        self.visible_enemies = []  # Enemies this unit can currently see
        
    def is_alive(self) -> bool:
        """Check if the unit is still alive"""
        return self.hp > 0
        
    def can_see(self, target_position, target_stealth_level, weather_visibility):
        """
        Check if this unit can see the target based on distance, stealth, and weather
        
        Returns: Visibility factor (0-1), 0 means cannot see at all
        """
        # Calculate distance
        dx = self.position[0] - target_position[0]
        dy = self.position[1] - target_position[1]
        distance = math.sqrt(dx*dx + dy*dy)
        
        # Base visibility decreases with distance
        # If distance <= detection range, start with perfect visibility
        # Otherwise, visibility drops off quickly
        if distance <= self.detection_range:
            visibility = 1.0
        else:
            visibility = max(0, 1.0 - (distance - self.detection_range) * 0.3)
        
        # Apply stealth reduction
        stealth_factor = max(0, 1.0 - target_stealth_level * 0.2)
        visibility *= stealth_factor
        
        # Apply weather effects
        visibility *= weather_visibility
        
        return visibility
    
    def calculate_damage(self, target, critical_hit=False):
        """
        Calculate damage against target considering type advantages
        
        Parameters:
            target: The target Unit
            critical_hit: Whether this is a critical hit (e.g. flank attack)
            
        Returns:
            damage: The calculated damage amount
        """
        # Base damage
        damage = self.attack_power
        
        # Apply type advantage/disadvantage
        type_multiplier = TYPE_ADVANTAGES[self.unit_type.value][target.unit_type.value]
        damage *= type_multiplier
        
        # Critical hit (flanking, surprise attack, etc.)
        if critical_hit:
            damage *= 1.5
            
        # Target defense state reduces damage
        if target.action_state == ActionType.DEFEND:
            damage *= 0.6
            
        # Randomize slightly (±10%)
        damage_variance = random.uniform(0.9, 1.1)
        damage *= damage_variance
        
        return int(damage)
        
    def attack(self, target, terrain_cover, weather_attack_modifier):
        """
        Attack the target unit and calculate damage
        
        Parameters:
            target: The target Unit
            terrain_cover: Cover bonus from terrain (0-1)
            weather_attack_modifier: Weather effect on attack accuracy/power
            
        Returns:
            damage_dealt: Amount of damage actually dealt
            hit_success: Whether the attack successfully hit
        """
        # Calculate hit chance based on distance and conditions
        # Base hit chance
        hit_chance = 0.8
        
        # Distance modifier
        dx = self.position[0] - target.position[0]
        dy = self.position[1] - target.position[1]
        distance = math.sqrt(dx*dx + dy*dy)
        
        if distance > ATTACK_RANGES[self.unit_type]:
            # Beyond attack range
            return 0, False
            
        # Distance penalty
        distance_modifier = 1.0 - (distance / (ATTACK_RANGES[self.unit_type] * 2))
        hit_chance *= max(0.4, distance_modifier)
        
        # Cover bonus for defender
        hit_chance *= (1.0 - terrain_cover * 0.5)
        
        # Weather effects
        hit_chance *= weather_attack_modifier
        
        # Determine if the attack hits
        hit_success = random.random() < hit_chance
        
        if not hit_success:
            return 0, False
            
        # Determine if it's a critical hit (10% chance)
        critical_hit = random.random() < 0.1
        
        # Calculate damage
        damage = self.calculate_damage(target, critical_hit)
        
        # Apply damage to target
        target.hp = max(0, target.hp - damage)
        
        return damage, True
    
    def select_best_action(self, enemies, allies, terrain_data, weather_data):
        """
        Selects the most appropriate action using simple heuristics
        
        Returns: ActionType enum value
        """
        # If no enemies visible, continue moving/searching
        if not enemies:
            return ActionType.MOVE
            
        # If very low health, consider retreating
        if self.hp < self.max_hp * 0.2:
            return ActionType.RETREAT
            
        # If multiple enemies nearby, consider defensive posture
        nearby_enemies = [e for e in enemies 
                          if math.dist(self.position, e.position) <= 2]
        if len(nearby_enemies) >= 2:
            return ActionType.DEFEND
            
        # Find closest enemy
        closest_enemy = min(enemies, 
                           key=lambda e: math.dist(self.position, e.position))
        
        # If enemy is within range, attack
        if math.dist(self.position, closest_enemy.position) <= ATTACK_RANGES[self.unit_type]:
            return ActionType.ATTACK
            
        # Otherwise, move toward enemy
        return ActionType.MOVE

    def take_action(self, action_type, battle_env, target=None):
        """
        Execute the specified action
        
        Parameters:
            action_type: ActionType enum value
            battle_env: Reference to the battlefield environment
            target: Optional target for attack or support
            
        Returns:
            result_dict: Dictionary with action results
        """
        # Update internal state
        self.action_state = action_type
        
        result = {"action": action_type.name, "success": True, "message": ""}
        
        # Process action based on type
        if action_type == ActionType.MOVE:
            # If we have a target, move toward it
            if target:
                path = astar(
                    tuple(self.position), 
                    tuple(target.position), 
                    GRID_SIZE, 
                    battle_env.obstacles,
                    battle_env.terrain_data,
                    self.unit_type
                )
                
                # Move up to speed steps along path
                max_steps = min(self.speed, len(path) - 1)
                if max_steps > 0:
                    self.position = list(path[max_steps])
                    result["message"] = f"Moved to {self.position}"
                else:
                    result["message"] = "Could not move closer to target"
            else:
                # No target, random movement
                dx = random.randint(-1, 1)
                dy = random.randint(-1, 1)
                new_x = max(0, min(GRID_SIZE - 1, self.position[0] + dx))
                new_y = max(0, min(GRID_SIZE - 1, self.position[1] + dy))
                if (new_x, new_y) not in battle_env.obstacles:
                    self.position = [new_x, new_y]
                    result["message"] = f"Moved randomly to {self.position}"
                
        elif action_type == ActionType.ATTACK:
            if not target:
                result["success"] = False
                result["message"] = "No target specified for attack"
            else:
                # Get terrain cover bonus
                terrain = battle_env.get_terrain_at(target.position)
                cover = TERRAIN_TYPES[terrain]["cover"]
                
                # Get weather attack modifier
                weather_mod = WEATHER_CONDITIONS[battle_env.current_weather]["attack"]
                
                # Execute attack
                damage, hit = self.attack(target, cover, weather_mod)
                
                if hit:
                    result["message"] = f"Attack hit for {damage} damage"
                    result["damage"] = damage
                    # Check if target was defeated
                    if target.hp <= 0:
                        result["message"] += f" - Target defeated!"
                else:
                    result["success"] = False
                    result["message"] = "Attack missed"
                
        elif action_type == ActionType.DEFEND:
            # Defensive stance reduces incoming damage
            result["message"] = "Defensive stance activated"
            
        elif action_type == ActionType.RETREAT:
            # Find safest direction (away from enemies)
            if battle_env.enemies:
                # Calculate average enemy position
                avg_enemy_x = sum(e.position[0] for e in battle_env.enemies) / len(battle_env.enemies)
                avg_enemy_y = sum(e.position[1] for e in battle_env.enemies) / len(battle_env.enemies)
                
                # Move in opposite direction
                retreat_dir_x = self.position[0] - avg_enemy_x
                retreat_dir_y = self.position[1] - avg_enemy_y
                
                # Normalize
                magnitude = math.sqrt(retreat_dir_x**2 + retreat_dir_y**2)
                if magnitude > 0:
                    retreat_dir_x /= magnitude
                    retreat_dir_y /= magnitude
                    
                    # Calculate new position
                    new_x = int(self.position[0] + retreat_dir_x * self.speed)
                    new_y = int(self.position[1] + retreat_dir_y * self.speed)
                    
                    # Ensure within bounds
                    new_x = max(0, min(GRID_SIZE - 1, new_x))
                    new_y = max(0, min(GRID_SIZE - 1, new_y))
                    
                    # Check for obstacles
                    if (new_x, new_y) not in battle_env.obstacles:
                        self.position = [new_x, new_y]
                        result["message"] = f"Retreated to {self.position}"
                    else:
                        result["message"] = "Retreat blocked by obstacle"
            else:
                result["message"] = "No enemies to retreat from"
                
        elif action_type == ActionType.SUPPORT:
            # Support allies (heal or buff them)
            if not target:
                result["success"] = False
                result["message"] = "No target specified for support"
            else:
                # Heal target if infantry supporting another unit
                if self.unit_type == UnitType.INFANTRY:
                    heal_amount = int(self.attack_power * 0.5)
                    target.hp = min(target.hp + heal_amount, target.max_hp)
                    result["message"] = f"Healed ally for {heal_amount} HP"
                
                # Increase target's attack power temporarily if drone
                elif self.unit_type == UnitType.AERIAL:
                    target.attack_power = int(target.attack_power * 1.2)  # 20% boost
                    result["message"] = "Boosted ally's attack power"
                    
                else:
                    result["message"] = "Provided generic support"
                
        return result


class Enemy(Unit):
    """Enemy unit class that extends Unit with AI behavior"""
    
    def __init__(self, unit_type, position, hp, attack_power, speed, 
                detection_range=2, stealth_level=0, ai_aggression=0.7):
        super().__init__(unit_type, position, hp, attack_power, speed, 
                        detection_range, stealth_level)
        self.ai_aggression = ai_aggression  # 0 to 1 (defensive to aggressive)
        
    def choose_action(self, friendly_units, other_enemies, battle_env):
        """AI logic to choose the next action"""
        # If no friendly units in range, move randomly
        visible_friendlies = []
        
        # Check which friendly units are visible
        for unit in friendly_units:
            visibility = self.can_see(
                unit.position, 
                unit.stealth_level,
                WEATHER_CONDITIONS[battle_env.current_weather]["visibility"]
            )
            
            if visibility > 0.3:  # Threshold to detect
                visible_friendlies.append(unit)
        
        if not visible_friendlies:
            # No targets visible, continue patrolling
            return ActionType.MOVE, None
        
        # Choose target based on several factors
        best_target = None
        best_score = -float('inf')
        
        for unit in visible_friendlies:
            # Calculate target score based on:
            # - Distance (closer = higher score)
            # - HP (lower = higher score)
            # - Type advantage
            
            distance = math.dist(self.position, unit.position)
            
            # Type advantage (how effective we are against this unit)
            type_factor = TYPE_ADVANTAGES[self.unit_type.value][unit.unit_type.value]
            
            # Combine factors into score
            hp_factor = 1.0 - (unit.hp / unit.max_hp)  # Lower HP = higher value
            distance_factor = 1.0 - (distance / GRID_SIZE)  # Closer = higher value
            
            # Calculate final score
            target_score = (
                (type_factor * 2) +  # Type advantage is most important
                (hp_factor * 1.5) +  # Low HP targets are tempting
                (distance_factor * 1)  # Distance is less important
            )
            
            if target_score > best_score:
                best_score = target_score
                best_target = unit
        
        # Determine action based on distance and aggression
        if best_target:
            distance = math.dist(self.position, best_target.position)
            
            # If within attack range, decide whether to attack or defend
            if distance <= ATTACK_RANGES[self.unit_type]:
                # More aggressive enemies attack more often
                if random.random() < self.ai_aggression:
                    return ActionType.ATTACK, best_target
                else:
                    return ActionType.DEFEND, None
            else:
                # Move toward target
                return ActionType.MOVE, best_target
                
        # Fallback
        return ActionType.MOVE, None


class BattlefieldEnv(gym.Env):
    """
    Enhanced Battlefield Environment with advanced tactical mechanics
    
    Features:
    - Multiple enemies with type-based strengths/weaknesses
    - Unit-specific attack and movement
    - Terrain and weather effects
    - Line of sight and stealth
    """
    
    def __init__(self, grid_size=GRID_SIZE, max_enemies=3, max_steps=50):
        """Initialize the battlefield environment"""
        super(BattlefieldEnv, self).__init__()
        
        self.grid_size = grid_size
        self.max_enemies = max_enemies
        self.max_steps = max_steps
        
        # Define observation and action spaces
        # We're using MultiDiscrete for actions (each unit can take one of 5 actions)
        self.num_friendly_units = 3  # Infantry, Tank, Drone
        self.action_space = spaces.MultiDiscrete([5] * self.num_friendly_units)
        
        # Observation space is more complex now, includes:
        # - Unit positions and stats
        # - Enemy positions and types
        # - Terrain type for each cell
        # - Weather conditions
        self.observation_space = spaces.Box(
            low=0, 
            high=1, 
            shape=(self.grid_size, self.grid_size, 7),  # More channels for more information
            dtype=np.float32
        )
        
        # Initialize attributes
        self.friendly_units = []
        self.enemies = []
        self.obstacles = set()
        self.battle_log = []
        self.step_count = 0
        self.current_weather = "Clear"
        self.current_terrain = "Plains"
        self.terrain_data = None  # Will be initialized in reset()
        self.battle_outcome = None
        
        # Call reset to initialize state
        self.reset()
        
    def reset(self):
        """Reset environment with new battlefield conditions"""
        # Reset step counter
        self.step_count = 0
        self.battle_outcome = None
        
        # Generate terrain grid
        self.terrain_data = self._generate_terrain_grid()
        
        # Choose random weather
        self.current_weather = random.choice(list(WEATHER_CONDITIONS.keys()))
        self.current_terrain = self._get_dominant_terrain()
        
        # Generate obstacles
        self.obstacles = set()
        num_obstacles = random.randint(5, 15)
        self._generate_obstacles(num_obstacles)
        
        # Create friendly units in non-overlapping positions
        self.friendly_units = self._create_friendly_units()
        
        # Create enemy units
        self.enemies = self._create_enemy_units()
        
        # Initialize empty battle log
        self.battle_log = []
        
        # Log initial battle state
        self._log_battle_state(None, None)
        
        # Generate and return observation
        return self.get_observation()
        
    def _generate_terrain_grid(self):
        """
        Generate a 2D grid of terrain types
        Returns 2D grid where each cell contains terrain modifier data
        """
        # Start with a base terrain type
        base_terrain = random.choice(list(TERRAIN_TYPES.keys()))
        
        # Create grid filled with base terrain
        grid = [[TERRAIN_TYPES[base_terrain] for _ in range(self.grid_size)] 
                for _ in range(self.grid_size)]
        
        # Add terrain "blobs" of different types
        num_blobs = random.randint(2, 5)
        for _ in range(num_blobs):
            terrain_type = random.choice(list(TERRAIN_TYPES.keys()))
            center_x = random.randint(0, self.grid_size-1)
            center_y = random.randint(0, self.grid_size-1)
            radius = random.randint(1, 3)
            
            # Create a blob of this terrain type
            for x in range(max(0, center_x-radius), min(self.grid_size, center_x+radius+1)):
                for y in range(max(0, center_y-radius), min(self.grid_size, center_y+radius+1)):
                    # If within radius
                    if math.sqrt((x-center_x)**2 + (y-center_y)**2) <= radius:
                        grid[x][y] = TERRAIN_TYPES[terrain_type]
        
        return grid
    
    def _get_dominant_terrain(self):
        """Return the most common terrain type in the grid"""
        if not self.terrain_data:
            return "Plains"  # Default
            
        # Count terrain types
        terrain_counts = {}
        for row in self.terrain_data:
            for cell in row:
                # We need to identify terrain type from the cell data
                # This requires matching terrain modifiers back to terrain types
                terrain_type = self._identify_terrain(cell)
                terrain_counts[terrain_type] = terrain_counts.get(terrain_type, 0) + 1
        
        # Return the most common
        return max(terrain_counts.items(), key=lambda x: x[1])[0]
    
    def _identify_terrain(self, terrain_cell):
        """Match terrain modifiers back to the terrain type"""
        for terrain_name, values in TERRAIN_TYPES.items():
            if terrain_cell == values:
                return terrain_name
        return "Plains"  # Default fallback
        
    def _generate_obstacles(self, num_obstacles):
        """Generate random obstacles on the battlefield"""
        self.obstacles = set()
        
        # Ensure we don't place obstacles where units are
        excluded_positions = set()
        for unit in self.friendly_units:
            excluded_positions.add(tuple(unit.position))
        for enemy in self.enemies:
            excluded_positions.add(tuple(enemy.position))
        
        # Generate obstacles in valid locations
        while len(self.obstacles) < num_obstacles:
            pos = (random.randint(0, self.grid_size-1), random.randint(0, self.grid_size-1))
            if pos not in excluded_positions and pos not in self.obstacles:
                self.obstacles.add(pos)
    
    def _create_friendly_units(self):
        """Create friendly units with appropriate stats"""
        # Generate non-overlapping positions
        positions = self._generate_non_overlapping_positions(3)
        
        # Create friendly units
        return [
            # Infantry unit
            Unit(
                unit_type=UnitType.INFANTRY,
                position=positions[0],
                hp=100,
                attack_power=15,
                speed=2,
                detection_range=2,
                stealth_level=1
            ),
            # Tank unit
            Unit(
                unit_type=UnitType.ARMORED,
                position=positions[1],
                hp=200,
                attack_power=30,
                speed=1,
                detection_range=2,
                stealth_level=0
            ),
            # Drone unit
            Unit(
                unit_type=UnitType.AERIAL,
                position=positions[2],
                hp=60,
                attack_power=10,
                speed=3,
                detection_range=4,
                stealth_level=2
            )
        ]
    
    def _create_enemy_units(self):
        """Create enemy units with varying types"""
        num_enemies = random.randint(1, self.max_enemies)
        
        # Generate non-overlapping positions
        positions = self._generate_non_overlapping_positions(
            num_enemies, 
            exclude_positions=[tuple(unit.position) for unit in self.friendly_units]
        )
        
        enemies = []
        for i in range(num_enemies):
            # Choose random enemy type (weighted toward infantry)
            enemy_type_weights = [0.5, 0.25, 0.15, 0.05, 0.05]  # Weights for each type
            enemy_type = random.choices(
                list(UnitType), 
                weights=enemy_type_weights, 
                k=1
            )[0]
            
            # Randomize aggression
            aggression = random.uniform(0.4, 0.9)
            
            # Create enemy with type-specific stats
            if enemy_type == UnitType.INFANTRY:
                enemy = Enemy(
                    unit_type=enemy_type,
                    position=positions[i],
                    hp=80,
                    attack_power=12,
                    speed=2,
                    detection_range=2,
                    stealth_level=0,
                    ai_aggression=aggression
                )
            elif enemy_type == UnitType.ARMORED:
                enemy = Enemy(
                    unit_type=enemy_type,
                    position=positions[i],
                    hp=160,
                    attack_power=25,
                    speed=1,
                    detection_range=2,
                    stealth_level=0,
                    ai_aggression=aggression
                )
            elif enemy_type == UnitType.AERIAL:
                enemy = Enemy(
                    unit_type=enemy_type,
                    position=positions[i],
                    hp=50,
                    attack_power=15,
                    speed=3,
                    detection_range=3,
                    stealth_level=1,
                    ai_aggression=aggression
                )
            elif enemy_type == UnitType.ARTILLERY:
                enemy = Enemy(
                    unit_type=enemy_type,
                    position=positions[i],
                    hp=70,
                    attack_power=40,
                    speed=1,
                    detection_range=4,
                    stealth_level=0,
                    ai_aggression=aggression * 0.8  # Artillery is more defensive
                )
            elif enemy_type == UnitType.STEALTH:
                enemy = Enemy(
                    unit_type=enemy_type,
                    position=positions[i],
                    hp=60,
                    attack_power=20,
                    speed=2,
                    detection_range=3,
                    stealth_level=3,
                    ai_aggression=aggression
                )
            
            enemies.append(enemy)
            
        return enemies
    
    def _generate_non_overlapping_positions(self, num_positions, exclude_positions=None):
        """Generate non-overlapping random positions"""
        if exclude_positions is None:
            exclude_positions = []
            
        positions = []
        excluded = set(exclude_positions)
        
        while len(positions) < num_positions:
            # For friendly units, start them on one side of the grid
            if not exclude_positions and len(positions) < 3:
                # Friendly units start on left side
                x = random.randint(0, self.grid_size // 3)
                y = random.randint(0, self.grid_size - 1)
            elif exclude_positions:
                # Enemy units start on right side (if friendlies are excluded)
                x = random.randint(self.grid_size * 2 // 3, self.grid_size - 1)
                y = random.randint(0, self.grid_size - 1)
            else:
                # Otherwise random position
                x = random.randint(0, self.grid_size - 1)
                y = random.randint(0, self.grid_size - 1)
                
            pos = (x, y)
            
            # Check if position is valid
            if pos not in excluded and pos not in self.obstacles:
                positions.append(pos)
                excluded.add(pos)
                
        return positions
        
    def get_observation(self):
        """
        Generate battle observation as a grid with multiple channels of information
        
        Channel 0: Friendly units (infantry=0.25, tank=0.5, drone=0.75)
        Channel 1: Enemy units (by type, normalized value)
        Channel 2: Obstacles (-1)
        Channel 3: Terrain (normalized values for different types)
        Channel 4: Unit health percentage
        Channel 5: Attack range indicators
        Channel 6: Visibility information
        """
        # Initialize all channels with zeros
        obs = np.zeros((self.grid_size, self.grid_size, 7), dtype=np.float32)
        
        # Channel 0: Friendly units
        for i, unit in enumerate(self.friendly_units):
            if unit.is_alive():
                value = (i + 1) / 4  # Scale to 0.25, 0.5, 0.75
                obs[unit.position[0], unit.position[1], 0] = value
        
        # Channel 1: Enemy units
        for enemy in self.enemies:
            if enemy.is_alive():
                value = (enemy.unit_type.value + 1) / 5  # Scale to 0.2, 0.4, 0.6, 0.8, 1.0
                obs[enemy.position[0], enemy.position[1], 1] = value
        
        # Channel 2: Obstacles
        for obs_pos in self.obstacles:
            obs[obs_pos[0], obs_pos[1], 2] = -1
            
        # Channel 3: Terrain data
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                # Encode terrain as normalized value
                terrain_cell = self.terrain_data[x][y]
                # Use cover value as proxy for terrain type (higher cover = more difficult terrain)
                terrain_value = terrain_cell.get('cover', 0) * 2  # Scale for visibility
                obs[x, y, 3] = terrain_value
                
        # Channel 4: Unit health
        for unit in self.friendly_units:
            if unit.is_alive():
                health_pct = unit.hp / unit.max_hp
                obs[unit.position[0], unit.position[1], 4] = health_pct
                
        for enemy in self.enemies:
            if enemy.is_alive():
                health_pct = enemy.hp / enemy.max_hp
                obs[enemy.position[0], enemy.position[1], 4] = -health_pct  # Negative to distinguish from friendly
                
        # Channel 5: Attack ranges
        for unit in self.friendly_units:
            if unit.is_alive():
                attack_range = ATTACK_RANGES[unit.unit_type]
                # Mark cells within attack range
                for x in range(self.grid_size):
                    for y in range(self.grid_size):
                        dist = math.dist(unit.position, [x, y])
                        if dist <= attack_range:
                            obs[x, y, 5] = max(obs[x, y, 5], 1 - (dist / (attack_range + 1)))
        
        # Channel 6: Visibility information
        weather_visibility = WEATHER_CONDITIONS[self.current_weather]["visibility"]
        
        # For each friendly unit, calculate what they can see
        for unit in self.friendly_units:
            if unit.is_alive():
                for x in range(self.grid_size):
                    for y in range(self.grid_size):
                        # Check line of sight
                        target_pos = [x, y]
                        # Dummy stealth level for terrain
                        terrain_stealth = 0
                        visibility = unit.can_see(target_pos, terrain_stealth, weather_visibility)
                        obs[x, y, 6] = max(obs[x, y, 6], visibility)
                
        return obs
    
    def get_terrain_at(self, position):
        """Get terrain type at the given position"""
        x, y = position
        if 0 <= x < self.grid_size and 0 <= y < self.grid_size and self.terrain_data:
            terrain_cell = self.terrain_data[x][y]
            return self._identify_terrain(terrain_cell)
        return "Plains"  # Default if out of bounds
    
    def step(self, actions):
        """
        Execute actions for all friendly units and then enemy units
        
        Parameters:
            actions: Array of action indices for friendly units
            
        Returns:
            observation: Updated observation
            reward: Reward from this step
            done: Whether the episode is done
            info: Additional information
        """
        # Initialize step result values
        reward = 0
        done = False
        info = {}
        
        # Increment step counter
        self.step_count += 1
        
        # Convert actions to ActionType enum
        action_types = [ActionType(min(int(a), 4)) for a in actions]
        
        # Process friendly unit actions
        action_results = []
        for i, (unit, action_type) in enumerate(zip(self.friendly_units, action_types)):
            if not unit.is_alive():
                continue
                
            # Determine target based on action type
            target = None
            if action_type == ActionType.ATTACK:
                # Find closest enemy as target
                visible_enemies = []
                
                # Check which enemies are visible
                weather_visibility = WEATHER_CONDITIONS[self.current_weather]["visibility"]
                for enemy in self.enemies:
                    if enemy.is_alive():
                        visibility = unit.can_see(
                            enemy.position, 
                            enemy.stealth_level,
                            weather_visibility
                        )
                        
                        if visibility > 0.3:  # Threshold to detect
                            visible_enemies.append(enemy)
                
                if visible_enemies:
                    # Target closest visible enemy
                    target = min(visible_enemies, 
                                key=lambda e: math.dist(unit.position, e.position))
            
            # Execute the action
            result = unit.take_action(action_type, self, target)
            action_results.append(result)
            
            # Update reward based on action result
            if action_type == ActionType.ATTACK and result["success"] and "damage" in result:
                reward += result["damage"] / 20  # Scale damage to reasonable reward
                
        # Process enemy actions
        enemy_results = []
        for enemy in self.enemies:
            if not enemy.is_alive():
                continue
                
            # Enemy AI chooses action and target
            action_type, target = enemy.choose_action(self.friendly_units, self.enemies, self)
            
            # Execute the action
            result = enemy.take_action(action_type, self, target)
            enemy_results.append(result)
            
            # Update reward based on enemy action
            if action_type == ActionType.ATTACK and result["success"] and "damage" in result:
                reward -= result["damage"] / 10  # Negative reward for taking damage
        
        # Check victory/defeat conditions
        friendly_alive = sum(1 for unit in self.friendly_units if unit.is_alive())
        enemies_alive = sum(1 for enemy in self.enemies if enemy.is_alive())
        
        if enemies_alive == 0:
            # All enemies defeated - Victory!
            done = True
            reward += 10  # Big reward for victory
            self.battle_outcome = "victory"
            info["result"] = "victory"
            info["message"] = "All enemies defeated!"
            
        elif friendly_alive == 0:
            # All friendly units defeated - Defeat
            done = True
            reward -= 10  # Penalty for defeat
            self.battle_outcome = "defeat"
            info["result"] = "defeat"
            info["message"] = "All friendly units defeated!"
            
        elif self.step_count >= self.max_steps:
            # Max steps reached - Draw
            done = True
            self.battle_outcome = "timeout"
            info["result"] = "timeout"
            info["message"] = f"Maximum steps ({self.max_steps}) reached"
        
        # Log battle state
        self._log_battle_state(actions, action_results)
        
        # Return results
        return self.get_observation(), reward, done, info
    
    def _log_battle_state(self, actions, action_results):
        """Log the current battle state to the battle log"""
        # Create log entry with battlefield conditions
        log_entry = {
            "step": self.step_count,
            "weather": self.current_weather,
            "terrain": self.current_terrain,
            "friendly_units": [],
            "enemy_units": [],
            "actions": [] if actions is not None else None,
            "action_results": [] if action_results is not None else None
        }
        
        # Log friendly unit states
        for i, unit in enumerate(self.friendly_units):
            unit_type_name = unit.unit_type.name
            unit_data = {
                "id": i,
                "type": unit_type_name,
                "position": unit.position,
                "hp": unit.hp,
                "max_hp": unit.max_hp,
                "action_state": unit.action_state.name if unit.is_alive() else "DEAD"
            }
            log_entry["friendly_units"].append(unit_data)
            
            # Add action if available
            if actions is not None and action_results is not None:
                if i < len(actions) and i < len(action_results):
                    action_data = {
                        "unit_id": i,
                        "action_type": ActionType(actions[i]).name,
                        "result": action_results[i]
                    }
                    log_entry["actions"].append(action_data)
        
        # Log enemy unit states
        for i, enemy in enumerate(self.enemies):
            enemy_type_name = enemy.unit_type.name
            enemy_data = {
                "id": i,
                "type": enemy_type_name,
                "position": enemy.position,
                "hp": enemy.hp,
                "max_hp": enemy.max_hp,
                "action_state": enemy.action_state.name if enemy.is_alive() else "DEAD"
            }
            log_entry["enemy_units"].append(enemy_data)
        
        # Append to battle log
        self.battle_log.append(log_entry)
    
    def save_battle_log(self, filename=None):
        """Save the battle log to a CSV file for later analysis"""
        # Default filename
        if filename is None:
            filename = f"data/battle_data_{self.current_terrain}_{self.current_weather}_{len(self.enemies)}_enemies.csv"
        
        try:
            # Convert battle log to dataframe-compatible format
            processed_data = []
            
            for entry in self.battle_log:
                # Base data for this step
                base_row = {
                    "Step": entry["step"],
                    "Weather": entry["weather"],
                    "Terrain": entry["terrain"],
                    "NumEnemies": len(entry["enemy_units"]),
                    "Result": self.battle_outcome if self.battle_outcome else "ongoing"
                }
                
                # Add friendly unit data
                for i, unit in enumerate(entry["friendly_units"]):
                    unit_prefix = f"F{i}_"
                    base_row.update({
                        f"{unit_prefix}Type": unit["type"],
                        f"{unit_prefix}X": unit["position"][0],
                        f"{unit_prefix}Y": unit["position"][1],
                        f"{unit_prefix}HP": unit["hp"],
                        f"{unit_prefix}MaxHP": unit["max_hp"],
                        f"{unit_prefix}State": unit["action_state"]
                    })
                    
                    # Add action data if available
                    if entry["actions"] and i < len(entry["actions"]):
                        action = entry["actions"][i]
                        base_row.update({
                            f"{unit_prefix}Action": action["action_type"],
                            f"{unit_prefix}ActionSuccess": action["result"].get("success", False),
                            f"{unit_prefix}ActionMsg": action["result"].get("message", "")
                        })
                
                # Add enemy unit data
                for i, enemy in enumerate(entry["enemy_units"]):
                    enemy_prefix = f"E{i}_"
                    base_row.update({
                        f"{enemy_prefix}Type": enemy["type"],
                        f"{enemy_prefix}X": enemy["position"][0],
                        f"{enemy_prefix}Y": enemy["position"][1],
                        f"{enemy_prefix}HP": enemy["hp"],
                        f"{enemy_prefix}MaxHP": enemy["max_hp"],
                        f"{enemy_prefix}State": enemy["action_state"]
                    })
                
                processed_data.append(base_row)
            
            # Convert to pandas DataFrame and save
            import pandas as pd
            df = pd.DataFrame(processed_data)
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            
            # Save to CSV
            df.to_csv(filename, index=False)
            print(f"Battle log saved to {filename}")
            
            return filename
                
        except Exception as e:
            print(f"Error saving battle log: {e}")
            import traceback
            traceback.print_exc()
            
            # Try an emergency backup
            try:
                import time
                emergency_filename = f"data/battle_log_emergency_{int(time.time())}.csv"
                
                # Very simple CSV with minimal processing
                with open(emergency_filename, 'w') as f:
                    f.write("Step,Weather,Terrain,NumFriendly,NumEnemy,Outcome\n")
                    for entry in self.battle_log:
                        friendly_alive = sum(1 for u in entry["friendly_units"] if u["hp"] > 0)
                        enemy_alive = sum(1 for e in entry["enemy_units"] if e["hp"] > 0)
                        f.write(f"{entry['step']},{entry['weather']},{entry['terrain']},{friendly_alive},{enemy_alive},{self.battle_outcome or 'ongoing'}\n")
                        
                print(f"Emergency battle log saved to {emergency_filename}")
                return emergency_filename
                
            except:
                print("Failed to save even emergency battle log!")
                return None
    
    def render(self, mode='human'):
        """Render the battlefield"""
        if mode != 'human':
            return
            
        # Create a grid for rendering
        grid = np.zeros((self.grid_size, self.grid_size, 3))
        
        # Add terrain colors
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                terrain = self.get_terrain_at([x, y])
                
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
        for obs_pos in self.obstacles:
            grid[obs_pos[0], obs_pos[1]] = [0, 0, 0]
            
        # Plot friendly units (blue)
        for unit in self.friendly_units:
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
                    
                plt.scatter(y, x, marker=marker, s=100, color='blue', label=unit.unit_type.name)
                
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
                plt.gca().add_patch(health_rect)
                
                # Attack range indicator
                attack_range = ATTACK_RANGES[unit.unit_type]
                range_circle = plt.Circle(
                    (y, x),
                    attack_range,
                    color='blue',
                    fill=False,
                    linestyle='--',
                    alpha=0.3
                )
                plt.gca().add_patch(range_circle)
        
        # Plot enemy units (red)
        for enemy in self.enemies:
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
                    
                plt.scatter(y, x, marker=marker, s=100, color='red', label=enemy.unit_type.name)
                
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
                plt.gca().add_patch(health_rect)
        
        # Set up grid
        plt.imshow(grid)
        plt.grid(True, color='black', linestyle='-', linewidth=0.5, alpha=0.2)
        
        # Add title with battle info
        plt.title(f"Step: {self.step_count} | Weather: {self.current_weather} | Terrain: {self.current_terrain}")
        
        # Show plot
        plt.tight_layout()
        plt.show()


def run_simulation(num_battles=10, max_steps=50, render_final=True, save_logs=True):
    """
    Run multiple battle simulations and collect data
    
    Parameters:
        num_battles: Number of battle simulations to run
        max_steps: Maximum steps per battle
        render_final: Whether to render the final state of battles
        save_logs: Whether to save battle logs
    """
    print(f"Starting enhanced battlefield simulation with {num_battles} battles, max {max_steps} steps each")
    
    # Track results
    results = {"victory": 0, "defeat": 0, "timeout": 0}
    log_files = []
    
    # Create battlefield environment
    env = BattlefieldEnv(max_steps=max_steps)
    
    # Run simulations
    for battle_num in range(num_battles):
        print(f"\n--- Battle {battle_num+1}/{num_battles} ---")
        
        # Reset environment
        obs = env.reset()
        
        # Report initial battle conditions
        print(f"Weather: {env.current_weather}, Terrain: {env.current_terrain}")
        print(f"Number of enemies: {len(env.enemies)}")
        
        # Print enemy types
        enemy_types = [enemy.unit_type.name for enemy in env.enemies]
        print(f"Enemy types: {enemy_types}")
        
        done = False
        total_reward = 0
        
        # Run battle until completion or max steps
        step = 0
        while not done and step < max_steps:
            # Choose actions for each friendly unit 
            # Either random or with a simple heuristic strategy
            actions = []
            
            for i, unit in enumerate(env.friendly_units):
                if not unit.is_alive():
                    actions.append(0)  # Dummy action for dead units
                    continue
                    
                # Get best action based on battlefield situation
                action = unit.select_best_action(
                    env.enemies, 
                    env.friendly_units,
                    env.terrain_data,
                    WEATHER_CONDITIONS[env.current_weather]
                ).value
                
                actions.append(action)
            
            # Execute actions
            obs, reward, done, info = env.step(actions)
            total_reward += reward
            
            # Print step summary
            step += 1
            if step % 5 == 0 or done:  # Print every 5 steps or at the end
                friendly_alive = sum(1 for unit in env.friendly_units if unit.is_alive())
                enemy_alive = sum(1 for enemy in env.enemies if enemy.is_alive())
                print(f"Step {step}: Reward: {reward:.2f}, Total: {total_reward:.2f}, " 
                      f"Friendly: {friendly_alive}/{len(env.friendly_units)}, "
                      f"Enemies: {enemy_alive}/{len(env.enemies)}")
        
        # Battle complete
        if 'result' in info:
            results[info['result']] += 1
            print(f"Battle {battle_num+1} result: {info['result']} - {info.get('message', '')}")
        
        # Save battle log
        if save_logs:
            log_file = env.save_battle_log()
            if log_file:
                log_files.append(log_file)
        
        # Render final state if requested
        if render_final:
            env.render()
    
    # Print overall results
    print("\n--- Simulation Results ---")
    print(f"Battles: {num_battles}")
    print(f"Victories: {results['victory']} ({results['victory']/num_battles*100:.1f}%)")
    print(f"Defeats: {results['defeat']} ({results['defeat']/num_battles*100:.1f}%)")
    print(f"Timeouts: {results['timeout']} ({results['timeout']/num_battles*100:.1f}%)")
    
    if save_logs and log_files:
        print(f"\nBattle logs saved to:")
        for log_file in log_files:
            print(f"- {log_file}")

    return results


if __name__ == "__main__":
    # Run a simulation with the enhanced battlefield
    results = run_simulation(num_battles=5, max_steps=30, render_final=True)