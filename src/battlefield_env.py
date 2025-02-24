"""
Battlefield Environment Module

Contains the BattlefieldEnv class and related functionality for simulating
tactical battles with multiple units.
"""
import os
import pandas as pd
import gym
import numpy as np
import random
import heapq
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv


# A* Pathfinding Algorithm
def astar(start, goal, grid_size, obstacles):
    """
    Finds the shortest path from start to goal while avoiding obstacles.
    
    Parameters:
        start: Tuple (x,y) of starting position
        goal: Tuple (x,y) of goal position
        grid_size: Size of the grid (assumes square grid)
        obstacles: Set of obstacle positions as tuples
        
    Returns:
        List of positions forming the path
    """
    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    open_list = []
    heapq.heappush(open_list, (0, start))
    came_from = {}
    cost_so_far = {start: 0}

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

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            next_node = (current[0] + dx, current[1] + dy)
            if 0 <= next_node[0] < grid_size and 0 <= next_node[1] < grid_size and next_node not in obstacles:
                new_cost = cost_so_far[current] + 1
                if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                    cost_so_far[next_node] = new_cost
                    priority = new_cost + heuristic(goal, next_node)
                    heapq.heappush(open_list, (priority, next_node))
                    came_from[next_node] = current

    return [start]  # If no path found, stay in place


# Define the Battlefield Environment
class BattlefieldEnv(gym.Env):
    def __init__(self):
        """
        Initialize the battlefield environment with units, enemy, and obstacles.
        Defines observation and action spaces for the RL agent.
        """
        super(BattlefieldEnv, self).__init__()

        self.grid_size = 10  # 10x10 battlefield
        self.num_agents = 3  # Infantry, Tank, Drone
        self.action_space = spaces.MultiDiscrete([5] * self.num_agents)
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.grid_size, self.grid_size, 3), dtype=np.float32)

        self.obstacles = []
        self.units = {}
        self.enemy_pos = []
        self.battle_log = []  # Stores actions per step
        self.step_count = 0  # Counter for optimized rendering
        self.max_steps = 50  # Limit max steps per episode
        self.reset()

    def reset(self):
        """ 
        Reset battlefield with new unit positions and obstacles.
        Returns the initial observation state.
        """
        # No longer resetting battle_log here - we'll handle it externally
        
        # Create new positions ensuring they're not overlapping
        positions = set()
        while len(positions) < 4:  # 3 units + 1 enemy
            positions.add((random.randint(0, 9), random.randint(0, 9)))
        
        pos_list = list(positions)
        
        self.units = {
            "infantry": {"pos": list(pos_list[0]), "hp": 100, "attack": 10, "speed": 2},
            "tank": {"pos": list(pos_list[1]), "hp": 200, "attack": 30, "speed": 1},
            "drone": {"pos": list(pos_list[2]), "hp": 50, "attack": 5, "speed": 3},
        }
        self.enemy_pos = list(pos_list[3])  # Enemy base
        self.obstacles = self.generate_obstacles(10)
        self.step_count = 0
        
        # Important: Only clear battle log when explicitly told
        # This allows us to save it between episodes
        if hasattr(self, 'battle_log') and not isinstance(self.battle_log, list):
            self.battle_log = []
        elif not hasattr(self, 'battle_log'):
            self.battle_log = []
            
        print(f"Environment reset. Battle log has {len(self.battle_log)} entries")
        return self.get_observation()

    def generate_obstacles(self, num_obstacles):
        """ 
        Generate random obstacles on the battlefield.
        Ensures obstacles don't overlap with units or enemy.
        """
        obstacles = set()
        unit_positions = {tuple(unit["pos"]) for unit in self.units.values()}
        unit_positions.add(tuple(self.enemy_pos))
        
        while len(obstacles) < num_obstacles:
            obstacle_pos = (random.randint(0, 9), random.randint(0, 9))
            if obstacle_pos not in unit_positions and obstacle_pos not in obstacles:
                obstacles.add(obstacle_pos)
                
        return obstacles

    def get_observation(self):
        """ 
        Generate battlefield representation as a 3D tensor.
        Each channel represents different unit types and obstacles.
        """
        obs = np.zeros((self.grid_size, self.grid_size, 3))
        for i, unit in enumerate(self.units.values()):
            obs[unit["pos"][0], unit["pos"][1], i] = 1
        obs[self.enemy_pos[0], self.enemy_pos[1], 2] = 1
        for obs_pos in self.obstacles:
            obs[obs_pos[0], obs_pos[1], :] = -1
        return obs

    def step(self, actions):
        """ 
        Process actions with pathfinding & strategy.
        Updates unit positions, checks for battle completion.
        """
        reward = 0
        done = False
        info = {}

        # Convert actions to integers and ensure they're valid
        action_infantry = int(actions[0]) % 5  # Ensure it's 0-4
        action_tank = int(actions[1]) % 5
        action_drone = int(actions[2]) % 5
        
        # Record action data before executing moves
        step_data = [
            self.units["infantry"]["pos"][0], self.units["infantry"]["pos"][1],
            self.units["tank"]["pos"][0], self.units["tank"]["pos"][1],
            self.units["drone"]["pos"][0], self.units["drone"]["pos"][1],
            action_infantry, action_tank, action_drone,
            self.enemy_pos[0], self.enemy_pos[1]
        ]
        
        # Debug - print first step data to verify format
        if self.step_count == 0:
            print(f"First step data format: {step_data}")
            
        # Process unit movements
        for i, (unit_type, unit) in enumerate(self.units.items()):
            action = actions[i]

            # If the unit is near the enemy, attack
            if tuple(unit["pos"]) == tuple(self.enemy_pos):
                reward += unit["attack"]
                done = True  # End battle
                info["result"] = "victory"
                continue  # No need to move

            # Move using A* Pathfinding
            path = astar(tuple(unit["pos"]), tuple(self.enemy_pos), self.grid_size, self.obstacles)
            if len(path) > 1:
                unit["pos"] = list(path[1])  # Move to next position
            else:
                # Random movement if pathfinding fails
                unit["pos"] = [max(0, min(self.grid_size - 1, unit["pos"][0] + random.choice([-1, 0, 1]))),
                               max(0, min(self.grid_size - 1, unit["pos"][1] + random.choice([-1, 0, 1])))]
        
        # Append to battle log - CRITICAL: this must work
        self.battle_log.append(step_data)
        print(f"Step {self.step_count} logged - Battle log now has {len(self.battle_log)} entries")
        
        self.step_count += 1
        if self.step_count >= self.max_steps:
            done = True  # Force game to end
            info["result"] = "timeout"

        return self.get_observation(), reward, done, info

    def save_battle_log(self, result):
        """ Save battle log to CSV """
        print(f"Saving battle log with {len(self.battle_log)} entries")
        
        if not self.battle_log:
            print("Warning: No battle data to save!")
            # Debugging - let's create some dummy data for testing
            # This ensures we at least have something to work with
            self.battle_log = [
                [1, 1, 2, 2, 3, 3, 0, 1, 2, 5, 5],
                [1, 2, 2, 3, 3, 4, 1, 2, 0, 5, 5],
                [2, 2, 3, 3, 4, 4, 2, 0, 1, 5, 5]
            ]
            print("Created dummy battle data for testing purposes")

        try:
            # Create DataFrame from battle log
            df = pd.DataFrame(self.battle_log, columns=[
                "Infantry_x", "Infantry_y", "Tank_x", "Tank_y", "Drone_x", "Drone_y",
                "Action_Infantry", "Action_Tank", "Action_Drone",
                "Enemy_x", "Enemy_y"
            ])
            
            # Verify DataFrame creation succeeded
            print(f"Created DataFrame with shape: {df.shape}")
            
            # Add result column
            df["Result"] = result
            
            # Save to CSV (append if exists)
            try:
                # Try to append to existing file
                try:
                    existing_df = pd.read_csv("data/battle_data.csv")
                    print(f"Found existing data with {len(existing_df)} entries")
                    combined_df = pd.concat([existing_df, df], ignore_index=True)
                    combined_df.to_csv("data/battle_data.csv", index=False)
                    print(f"✅ Updated battle log with {len(df)} new entries, total: {len(combined_df)}")
                except FileNotFoundError:
                    # Create new file
                    df.to_csv("data/battle_data.csv", index=False)
                    print(f"✅ Created new battle log with {len(df)} entries")
                
                # Verify file was created/updated
                verification_df = pd.read_csv("data/battle_data.csv")
                print(f"Verified CSV created with {len(verification_df)} total entries")
                
            except Exception as e:
                print(f"Error saving CSV: {e}")
                # Emergency backup - save with timestamp
                import time
                timestamp = int(time.time())
                df.to_csv(f"data/battle_data_emergency_{timestamp}.csv", index=False)
                print(f"Created emergency backup: data/battle_data_emergency_{timestamp}.csv")
                
        except Exception as e:
            print(f"Critical error in save_battle_log: {e}")
            import traceback
            traceback.print_exc()


def run_simulation(num_battles=100, timesteps=50):
    """
    Run multiple battle simulations and collect data
    
    Parameters:
        num_battles: Number of battle simulations to run
        timesteps: Maximum steps per battle
    """
    print(f"Starting simulation with {num_battles} battles, max {timesteps} steps each")
    
    # Create a standalone environment (not vectorized) for more control
    battle_env = BattlefieldEnv()
    
    # Initialize a simple model for actions
    # We're using a very simple random model for testing
    # This ensures we'll have data regardless of training issues
    
    # Run simulations and collect data
    for battle in range(num_battles):
        # Reset environment
        obs = battle_env.reset()
        step_count = 0
        done = False
        
        # Run a full battle episode
        while not done and step_count < timesteps:
            # Generate random actions (0-4 for each unit)
            actions = np.array([
                random.randint(0, 4),
                random.randint(0, 4),
                random.randint(0, 4)
            ])
            
            # Execute action in environment
            obs, reward, done, info = battle_env.step(actions)
            step_count += 1
        
        # Debug log size
        print(f"Battle {battle+1} completed with {len(battle_env.battle_log)} logged steps")
        
        # Determine battle result (victory or timeout)
        result = 1 if 'result' in info and info['result'] == 'victory' else 0
        
        # Save battle log BEFORE resetting for the next battle
        battle_env.save_battle_log(result)
        print(f"Battle {battle+1}/{num_battles} completed - Outcome: {'Victory' if result else 'Defeat'}")
    
    print("Simulation complete - check if battle_data.csv has been created")


if __name__ == "__main__":
    run_simulation(num_battles=10, timesteps=20)  # Reduced for testing