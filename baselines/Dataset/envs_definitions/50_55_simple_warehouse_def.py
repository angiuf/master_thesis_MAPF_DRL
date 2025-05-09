import numpy as np
import matplotlib.pyplot as plt

# --- PLACEHOLDER DATA --- 
# Actual obstacle data for the 50x55 simple warehouse map should be loaded or generated here.
# Example: obstacles_50_55_simple = np.load('path/to/simple_warehouse_50_55_obstacles.npy')
obstacles_50_55_simple = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                                   [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                                   [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                                   [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                                   [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                                   [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                                   [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                                   [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                                   [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                                   [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                                   [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                                   [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                                   [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                                   ]) # Placeholder: Represents no obstacles currently


# --- PLACEHOLDER DATA --- 
# Actual open list data for the 50x55 simple warehouse map should be loaded or generated here.
# This list should contain coordinates [r, c] of non-obstacle cells.
# Example: open_list_50_55_simple = np.load('path/to/simple_warehouse_50_55_openlist.npy').tolist()
# As a placeholder, assuming all cells are open if no obstacles are defined:
# open_list_50_55_simple = [[5,0], [6,0], [7,0], [8,0], [9,0], [10,0], [11,0], [12,0], [13,0], [13,0], [13,0], [13,0], [13,0], [14,0], [15,0], [16,0], [17,0], [18,0], [19,0], [20,0], [21,0], [22,0], [23,0], [24,0], [25,0], [26,0], [27,0], [28,0], [29,0], [30,0], [31,0], [32,0], [33,0], [34,0], [35,0], [36,0], [37,0], [38,0], [39,0], [40,0], [41,0], [42,0], [43,0], [44,0], [45,0],
#                           [5,54], [6,54], [7,54], [8,54], [9,54], [10,54], [11,54], [12,54], [13,54], [14,54], [15,54], [16,54], [17,54], [18,54], [19,54], [20,54], [21,54], [22,54], [23,54], [24,54], [25,54], [26,54], [27,54], [28,54], [29,54], [30,54], [31,54], [32,54], [33,54], [34,54], [35,54], [36,54], [37,54], [38,54], [39,54], [40,54], [41,54], [42,54], [43,54], [44,54], [45,54],
#                           ]

def generate_open_list_50_55_simple():
    """
    Generates a list of open cells for a 50x55 simple warehouse map.
    The open cells are represented as [row, column] coordinates.
    """
    # Define grid dimensions (although not strictly needed for coordinate generation)
    rows = 50
    cols = 55

    # Initialize the list to store coordinates
    green_squares = []

    # --- Define coordinate ranges for green squares ---
    # Coordinates are [column, row] or [x, y]
    # Rows are indexed from 0 (bottom) to 49 (top)
    # Columns are indexed from 0 (left) to 54 (right)

    left_row_ranges = range(5,46)

    for r in left_row_ranges:
        green_squares.append([r,0])
        green_squares.append([r,54])

    # 2. Middle repeating blocks
    # These occur in 5-row bands, starting at rows y = 8, 15, 22, 29, 36, 43
    middle_row_starts = [4,5,8,9,12,13,16,17,20,21,24,25,28,29,32,33,36,37,40,41,44,45]
    # Within each band, the columns are segmented
    middle_col_ranges = [
        range(5,16),   # x = 2 to 10
        range(22, 33),  # x = 14 to 22
        range(39, 50),  # x = 26 to 34   # x = 50 to 52
    ]

    # Loop through the starting row of each middle band
    for r in middle_row_starts:
        # Iterate through the 5 rows within the current band (e.g., 8 to 12)
        for c_range in middle_col_ranges:
            for c in c_range:
                green_squares.append([r,c])

    # The 'green_squares' list now contains all the coordinates.
    # You can uncomment the following lines to print the list or its length.
    print(green_squares)
    print(f"Total number of green squares: {len(green_squares)}")

    return green_squares

open_list_50_55_simple = [[5, 0], [5, 54], [6, 0], [6, 54], [7, 0], [7, 54], [8, 0], [8, 54], [9, 0], [9, 54], [10, 0], [10, 54], [11, 0], [11, 54], [12, 0], [12, 54], [13, 0], [13, 54], [14, 0], [14, 54], [15, 0], [15, 54], [16, 0], [16, 54], [17, 0], [17, 54], [18, 0], [18, 54], [19, 0], [19, 54], [20, 0], [20, 54], [21, 0], [21, 54], [22, 0], [22, 54], [23, 0], [23, 54], [24, 0], [24, 54], [25, 0], [25, 54], [26, 0], [26, 54], [27, 0], [27, 54], [28, 0], [28, 54], [29, 0], [29, 54], [30, 0], [30, 54], [31, 0], [31, 54], [32, 0], [32, 54], [33, 0], [33, 54], [34, 0], [34, 54], [35, 0], [35, 54], [36, 0], [36, 54], [37, 0], [37, 54], [38, 0], [38, 54], [39, 0], [39, 54], [40, 0], [40, 54], [41, 0], [41, 54], [42, 0], [42, 54], [43, 0], [43, 54], [44, 0], [44, 54], [45, 0], [45, 54], [4, 5], [4, 6], [4, 7], [4, 8], [4, 9], [4, 10], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 22], [4, 23], [4, 24], [4, 25], [4, 26], [4, 27], [4, 28], [4, 29], [4, 30], [4, 31], [4, 32], [4, 39], [4, 40], [4, 41], [4, 42], [4, 43], [4, 44], [4, 45], [4, 46], [4, 47], [4, 48], [4, 49], [5, 5], [5, 6], [5, 7], [5, 8], [5, 9], [5, 10], [5, 11], [5, 12], [5, 13], [5, 14], [5, 15], [5, 22], [5, 23], [5, 24], [5, 25], [5, 26], [5, 27], [5, 28], [5, 29], [5, 30], [5, 31], [5, 32], [5, 39], [5, 40], [5, 41], [5, 42], [5, 43], [5, 44], [5, 45], [5, 46], [5, 47], [5, 48], [5, 49], [8, 5], [8, 6], [8, 7], [8, 8], [8, 9], [8, 10], [8, 11], [8, 12], [8, 13], [8, 14], [8, 15], [8, 22], [8, 23], [8, 24], [8, 25], [8, 26], [8, 27], [8, 28], [8, 29], [8, 30], [8, 31], [8, 32], [8, 39], [8, 40], [8, 41], [8, 42], [8, 43], [8, 44], [8, 45], [8, 46], [8, 47], [8, 48], [8, 49], [9, 5], [9, 6], [9, 7], [9, 8], [9, 9], [9, 10], [9, 11], [9, 12], [9, 13], [9, 14], [9, 15], [9, 22], [9, 23], [9, 24], [9, 25], [9, 26], [9, 27], [9, 28], [9, 29], [9, 30], [9, 31], [9, 32], [9, 39], [9, 40], [9, 41], [9, 42], [9, 43], [9, 44], [9, 45], [9, 46], [9, 47], [9, 48], [9, 49], [12, 5], [12, 6], [12, 7], [12, 8], [12, 9], [12, 10], [12, 11], [12, 12], [12, 13], [12, 14], [12, 15], [12, 22], [12, 23], [12, 24], [12, 25], [12, 26], [12, 27], [12, 28], [12, 29], [12, 30], [12, 31], [12, 32], [12, 39], [12, 40], [12, 41], [12, 42], [12, 43], [12, 44], [12, 45], [12, 46], [12, 47], [12, 48], [12, 49], [13, 5], [13, 6], [13, 7], [13, 8], [13, 9], [13, 10], [13, 11], [13, 12], [13, 13], [13, 14], [13, 15], [13, 22], [13, 23], [13, 24], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [13, 30], [13, 31], [13, 32], [13, 39], [13, 40], [13, 41], [13, 42], [13, 43], [13, 44], [13, 45], [13, 46], [13, 47], [13, 48], [13, 49], [16, 5], [16, 6], [16, 7], [16, 8], [16, 9], [16, 10], [16, 11], [16, 12], [16, 13], [16, 14], [16, 15], [16, 22], [16, 23], [16, 24], [16, 25], [16, 26], [16, 27], [16, 28], [16, 29], [16, 30], [16, 31], [16, 32], [16, 39], [16, 40], [16, 41], [16, 42], [16, 43], [16, 44], [16, 45], [16, 46], [16, 47], [16, 48], [16, 49], [17, 5], [17, 6], [17, 7], [17, 8], [17, 9], [17, 10], [17, 11], [17, 12], [17, 13], [17, 14], [17, 15], [17, 22], [17, 23], [17, 24], [17, 25], [17, 26], [17, 27], [17, 28], [17, 29], [17, 30], [17, 31], [17, 32], [17, 39], [17, 40], [17, 41], [17, 42], [17, 43], [17, 44], [17, 45], [17, 46], [17, 47], [17, 48], [17, 49], [20, 5], [20, 6], [20, 7], [20, 8], [20, 9], [20, 10], [20, 11], [20, 12], [20, 13], [20, 14], [20, 15], [20, 22], [20, 23], [20, 24], [20, 25], [20, 26], [20, 27], [20, 28], [20, 29], [20, 30], [20, 31], [20, 32], [20, 39], [20, 40], [20, 41], [20, 42], [20, 43], [20, 44], [20, 45], [20, 46], [20, 47], [20, 48], [20, 49], [21, 5], [21, 6], [21, 7], [21, 8], [21, 9], [21, 10], [21, 11], [21, 12], [21, 13], [21, 14], [21, 15], [21, 22], [21, 23], [21, 24], [21, 25], [21, 26], [21, 27], [21, 28], [21, 29], [21, 30], [21, 31], [21, 32], [21, 39], [21, 40], [21, 41], [21, 42], [21, 43], [21, 44], [21, 45], [21, 46], [21, 47], [21, 48], [21, 49], [24, 5], [24, 6], [24, 7], [24, 8], [24, 9], [24, 10], [24, 11], [24, 12], [24, 13], [24, 14], [24, 15], [24, 22], [24, 23], [24, 24], [24, 25], [24, 26], [24, 27], [24, 28], [24, 29], [24, 30], [24, 31], [24, 32], [24, 39], [24, 40], [24, 41], [24, 42], [24, 43], [24, 44], [24, 45], [24, 46], [24, 47], [24, 48], [24, 49], [25, 5], [25, 6], [25, 7], [25, 8], [25, 9], [25, 10], [25, 11], [25, 12], [25, 13], [25, 14], [25, 15], [25, 22], [25, 23], [25, 24], [25, 25], [25, 26], [25, 27], [25, 28], [25, 29], [25, 30], [25, 31], [25, 32], [25, 39], [25, 40], [25, 41], [25, 42], [25, 43], [25, 44], [25, 45], [25, 46], [25, 47], [25, 48], [25, 49], [28, 5], [28, 6], [28, 7], [28, 8], [28, 9], [28, 10], [28, 11], [28, 12], [28, 13], [28, 14], [28, 15], [28, 22], [28, 23], [28, 24], [28, 25], [28, 26], [28, 27], [28, 28], [28, 29], [28, 30], [28, 31], [28, 32], [28, 39], [28, 40], [28, 41], [28, 42], [28, 43], [28, 44], [28, 45], [28, 46], [28, 47], [28, 48], [28, 49], [29, 5], [29, 6], [29, 7], [29, 8], [29, 9], [29, 10], [29, 11], [29, 12], [29, 13], [29, 14], [29, 15], [29, 22], [29, 23], [29, 24], [29, 25], [29, 26], [29, 27], [29, 28], [29, 29], [29, 30], [29, 31], [29, 32], [29, 39], [29, 40], [29, 41], [29, 42], [29, 43], [29, 44], [29, 45], [29, 46], [29, 47], [29, 48], [29, 49], [32, 5], [32, 6], [32, 7], [32, 8], [32, 9], [32, 10], [32, 11], [32, 12], [32, 13], [32, 14], [32, 15], [32, 22], [32, 23], [32, 24], [32, 25], [32, 26], [32, 27], [32, 28], [32, 29], [32, 30], [32, 31], [32, 32], [32, 39], [32, 40], [32, 41], [32, 42], [32, 43], [32, 44], [32, 45], [32, 46], [32, 47], [32, 48], [32, 49], [33, 5], [33, 6], [33, 7], [33, 8], [33, 9], [33, 10], [33, 11], [33, 12], [33, 13], [33, 14], [33, 15], [33, 22], [33, 23], [33, 24], [33, 25], [33, 26], [33, 27], [33, 28], [33, 29], [33, 30], [33, 31], [33, 32], [33, 39], [33, 40], [33, 41], [33, 42], [33, 43], [33, 44], [33, 45], [33, 46], [33, 47], [33, 48], [33, 49], [36, 5], [36, 6], [36, 7], [36, 8], [36, 9], [36, 10], [36, 11], [36, 12], [36, 13], [36, 14], [36, 15], [36, 22], [36, 23], [36, 24], [36, 25], [36, 26], [36, 27], [36, 28], [36, 29], [36, 30], [36, 31], [36, 32], [36, 39], [36, 40], [36, 41], [36, 42], [36, 43], [36, 44], [36, 45], [36, 46], [36, 47], [36, 48], [36, 49], [37, 5], [37, 6], [37, 7], [37, 8], [37, 9], [37, 10], [37, 11], [37, 12], [37, 13], [37, 14], [37, 15], [37, 22], [37, 23], [37, 24], [37, 25], [37, 26], [37, 27], [37, 28], [37, 29], [37, 30], [37, 31], [37, 32], [37, 39], [37, 40], [37, 41], [37, 42], [37, 43], [37, 44], [37, 45], [37, 46], [37, 47], [37, 48], [37, 49], [40, 5], [40, 6], [40, 7], [40, 8], [40, 9], [40, 10], [40, 11], [40, 12], [40, 13], [40, 14], [40, 15], [40, 22], [40, 23], [40, 24], [40, 25], [40, 26], [40, 27], [40, 28], [40, 29], [40, 30], [40, 31], [40, 32], [40, 39], [40, 40], [40, 41], [40, 42], [40, 43], [40, 44], [40, 45], [40, 46], [40, 47], [40, 48], [40, 49], [41, 5], [41, 6], [41, 7], [41, 8], [41, 9], [41, 10], [41, 11], [41, 12], [41, 13], [41, 14], [41, 15], [41, 22], [41, 23], [41, 24], [41, 25], [41, 26], [41, 27], [41, 28], [41, 29], [41, 30], [41, 31], [41, 32], [41, 39], [41, 40], [41, 41], [41, 42], [41, 43], [41, 44], [41, 45], [41, 46], [41, 47], [41, 48], [41, 49], [44, 5], [44, 6], [44, 7], [44, 8], [44, 9], [44, 10], [44, 11], [44, 12], [44, 13], [44, 14], [44, 15], [44, 22], [44, 23], [44, 24], [44, 25], [44, 26], [44, 27], [44, 28], [44, 29], [44, 30], [44, 31], [44, 32], [44, 39], [44, 40], [44, 41], [44, 42], [44, 43], [44, 44], [44, 45], [44, 46], [44, 47], [44, 48], [44, 49], [45, 5], [45, 6], [45, 7], [45, 8], [45, 9], [45, 10], [45, 11], [45, 12], [45, 13], [45, 14], [45, 15], [45, 22], [45, 23], [45, 24], [45, 25], [45, 26], [45, 27], [45, 28], [45, 29], [45, 30], [45, 31], [45, 32], [45, 39], [45, 40], [45, 41], [45, 42], [45, 43], [45, 44], [45, 45], [45, 46], [45, 47], [45, 48], [45, 49]]
                          
ENV_DEFINITION = {
    "grid_size": "50_55",
    "map_name": "simple_warehouse",
    "obstacles": obstacles_50_55_simple, # Using placeholder data
    "open_list": open_list_50_55_simple, # Using placeholder data
    "agent_counts": [4, 8, 16, 32, 64, 128, 256]
}

def plot_environment(obstacles, open_list, grid_size_str, map_name):
    """Plots the environment grid, marking obstacles and open list locations."""
    try:
        rows, cols = map(int, grid_size_str.split('_'))
    except ValueError:
        print(f"Error: Invalid grid_size format: {grid_size_str}. Expected 'rows_cols'.")
        return

    # Create a grid representing the environment
    # 0: open space, 1: obstacle, 2: open list location
    grid = np.zeros((rows, cols))

    # Mark obstacles
    # Assuming obstacles are represented by 1 in the input array
    # or directly by coordinates if the obstacles array structure changes.
    if obstacles.size > 0:
        # Check if obstacles is a 2D numpy array matching grid dimensions (like the 15x15 example)
        if len(obstacles.shape) == 2 and obstacles.shape == (rows, cols):
            grid[obstacles == 1] = 1 # Mark obstacles from the grid array
        # Check if obstacles is a list or array of coordinates
        elif len(obstacles.shape) == 2 and obstacles.shape[1] == 2:
             for r, c in obstacles:
                 if 0 <= r < rows and 0 <= c < cols:
                     grid[r, c] = 1
        else:
            print(f"Warning: Obstacle data format not recognized or empty. Shape: {obstacles.shape}")


    # Mark open list locations
    for r, c in open_list:
        if 0 <= r < rows and 0 <= c < cols:
            if grid[r, c] == 1:
                # Don't print warning if obstacles are just placeholders (empty)
                if obstacles.size > 0:
                    print(f"Warning: Open list location ({r},{c}) overlaps with an obstacle.")
            grid[r, c] = 2 # Mark open list locations

    # Plotting
    fig, ax = plt.subplots(figsize=(cols/5, rows/5)) # Adjust figure size based on grid
    # Use gray for obstacles, light green for open list, white for free space
    cmap = plt.cm.colors.ListedColormap(['white', 'black', 'lightgreen'])
    bounds = [-0.5, 0.5, 1.5, 2.5]
    norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)

    ax.imshow(grid, cmap=cmap, norm=norm)

    # Draw gridlines
    ax.set_xticks(np.arange(-.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-.5, rows, 1), minor=True)
    ax.grid(which='minor', color='grey', linestyle='-', linewidth=0.5)

    # Remove ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])

    plt.title(f"Environment: {map_name} ({grid_size_str})")
    # Create legend handles
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, fc="white", ec="black", label='Free Space'),
        plt.Rectangle((0, 0), 1, 1, fc="black", ec="black", label='Obstacle'),
        plt.Rectangle((0, 0), 1, 1, fc="lightgreen", ec="black", label='Open List')
    ]
    ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend
    plt.show()

if __name__ == "__main__":
    print(f"Plotting environment: {ENV_DEFINITION['map_name']} ({ENV_DEFINITION['grid_size']})")
    # Note: This will plot the placeholder data unless actual data is loaded.
    plot_environment(
        ENV_DEFINITION["obstacles"],
        ENV_DEFINITION["open_list"],
        ENV_DEFINITION["grid_size"],
        ENV_DEFINITION["map_name"]
    )
