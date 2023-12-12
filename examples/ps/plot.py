import os
import glob
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from matplotlib.animation import FuncAnimation

from argparse import ArgumentParser

from aiaccel.config import load_config
from aiaccel.workspace import Workspace

from user import main as objective_function


parser = ArgumentParser()
parser.add_argument("--config", "-c", type=str, default="config.yaml")

args = parser.parse_args()
config = load_config(args.config)

workspace = Workspace(config.generic.workspace)
csv_directory = str(workspace.path / 'particle_swarm_optimizer')
output_file = './animation.gif'


# Set the x-axis limits
xlim = [None] * 2
xlim[0] = config.optimize.parameters[0].lower
xlim[1] = config.optimize.parameters[0].upper

# Set the y-axis limits
ylim = [None] * 2
ylim[0] = config.optimize.parameters[1].lower
ylim[1] = config.optimize.parameters[1].upper


def animate_and_save_trajectory(csv_directory, output_file):
    # Get CSV files in the directory
    csv_files = glob.glob(os.path.join(csv_directory, "*.csv"))

    # Create Figure and Axes for plotting
    fig, ax = plt.subplots()

    # Generate a grid for the search space
    x_grid = np.linspace(xlim[0], xlim[1], 100)
    y_grid = np.linspace(ylim[0], ylim[1], 100)
    X, Y = np.meshgrid(x_grid, y_grid)
    P = np.dstack((X, Y))
    Z = np.zeros_like(X)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            p = {"x1": X[i, j], "x2": Y[i, j]}
            Z[i, j] = objective_function(p)

    # Plot contours
    contour = ax.contour(X, Y, Z, levels=20, cmap='viridis')

    def animate(frame):
        ax.clear()
        contour = ax.contour(X, Y, Z, levels=20, cmap='viridis')
        for csv_file in csv_files:
            # Read data from CSV file
            df = pd.read_csv(csv_file, header=None)
            # Get coordinate columns from the data
            x = df.iloc[:frame+1, 1]
            y = df.iloc[:frame+1, 2]
            # Plot the trajectory
            ax.plot(x, y, marker='o', label=os.path.basename(csv_file))
        # Draw lines to indicate point movement
        for csv_file in csv_files:
            df = pd.read_csv(csv_file, header=None)
            x = df.iloc[:frame+1, 1]
            y = df.iloc[:frame+1, 2]
            ax.plot(x, y, linestyle='-', marker='o', markersize=5)

        #ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

    # Create the animation
    ani = FuncAnimation(fig, animate, frames=len(pd.read_csv(csv_files[0])), interval=500, repeat=False)
    # Save the animation to a file
    ani.save(output_file, writer='pillow')

if __name__ == "__main__":
    print("start")
    animate_and_save_trajectory(csv_directory, output_file)
