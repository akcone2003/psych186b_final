import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

GRID_SIZE = 10  # Match the BattlefieldEnv grid size

# Define terrain colors for the background
TERRAIN_COLORS = {
    "Plains": "lightgreen",
    "Mountains": "gray",
    "Forest": "darkgreen",
    "Urban": "lightgray",
    "Desert": "gold",
    "Unknown": "white"
}

# Define weather effects for overlaying text
WEATHER_EFFECTS = {
    "Clear": "",
    "Foggy": "🌫 Fog",
    "Rainy": "🌧 Rain",
    "Stormy": "⛈ Storm",
    "Snowy": "❄️ Snow"
}

def load_latest_battle_state(file_path="data/battle_data.csv"):
    """Loads the latest battlefield state, including terrain, weather, and obstacles from the simulation."""
    
    # Read the latest battle data
    df = pd.read_csv(file_path)

    if df.empty:
        print("⚠️ Warning: Battle data file is empty!")
        return None, None, None, "Unknown", "Unknown"

    # Get the last recorded time step
    last_row = df.iloc[-1]

    # Extract unit positions
    friendly_units = {
        "infantry": (last_row["Infantry_x"], last_row["Infantry_y"]),
        "tank": (last_row["Tank_x"], last_row["Tank_y"]),
        "drone": (last_row["Drone_x"], last_row["Drone_y"]),
    }
    enemy_pos = (last_row["Enemy_x"], last_row["Enemy_y"])

    # Extract terrain and weather from the simulation log
    terrain = last_row.get("Terrain", "Unknown")  # Use "Unknown" if missing
    weather = last_row.get("Weather", "Unknown")  # Use "Unknown" if missing

    # Extract obstacles (if present in the log)
    obstacles = []
    if "Obstacle_X" in df.columns and "Obstacle_Y" in df.columns:
        obstacles = list(zip(df["Obstacle_X"].dropna().astype(int), df["Obstacle_Y"].dropna().astype(int)))

    return friendly_units, enemy_pos, obstacles, terrain, weather

def show_latest_battlefield(file_path="data/battle_data.csv"):
    """Displays the latest battlefield state as a grid-based visualization with terrain & weather from the simulation."""
    
    # Load the latest battle state
    friendly_units, enemy_pos, obstacles, terrain, weather = load_latest_battle_state(file_path)

    # Handle cases where the data could not be loaded
    if friendly_units is None or enemy_pos is None:
        print("❌ Error: Could not load battle state!")
        return

    # Create a blank grid
    battlefield = np.zeros((GRID_SIZE, GRID_SIZE))

    # Mark positions (assign different values for different unit types)
    for unit, pos in friendly_units.items():
        battlefield[pos] = 1  # Friendly units (blue)
    battlefield[enemy_pos] = 2  # Enemy (red)

    # Create figure
    fig, ax = plt.subplots(figsize=(6, 6))

    # ✅ Ensure terrain color is applied correctly
    terrain_color = TERRAIN_COLORS.get(terrain, "white")
    ax.set_facecolor(terrain_color)

    # Custom colormap for visualization
    cmap = plt.cm.get_cmap("coolwarm", 3)  # Three categories: friendly, enemy, obstacles
    ax.imshow(battlefield, cmap=cmap, origin="upper", vmin=0, vmax=2)

    # ✅ Annotate unit positions explicitly
    for unit, pos in friendly_units.items():
        label = unit[0].upper()  # "I", "T", "D"
        ax.text(pos[1], pos[0], label, ha="center", va="center", fontsize=14, color="white", fontweight="bold")

    # ✅ Display enemy as "E"
    ax.text(enemy_pos[1], enemy_pos[0], "E", ha="center", va="center", fontsize=14, color="yellow", fontweight="bold")

    # ✅ Mark obstacles
    for obs in obstacles:
        ax.add_patch(plt.Rectangle((obs[1] - 0.5, obs[0] - 0.5), 1, 1, color="black", alpha=0.6))

    # ✅ Ensure weather always displays
    weather_text = WEATHER_EFFECTS.get(weather, weather)  # Ensure it doesn't return empty
    plt.title(f"Battlefield Visualization\nTerrain: {terrain} | Weather: {weather_text}")

    # Grid settings
    ax.set_xticks(np.arange(GRID_SIZE) - 0.5, minor=True)
    ax.set_yticks(np.arange(GRID_SIZE) - 0.5, minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=0.5)
    ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)

    plt.show()
