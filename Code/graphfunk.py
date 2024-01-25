import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from adjustText import adjust_text


def plot_type(df, column_coordinates, type_to_plot, color):
    # Filter for the specified type
    type_df = df[df['type'] == type_to_plot]

    # Use only specified type twitter_names in column_coordinates
    column_coordinates_type = column_coordinates[column_coordinates.index.isin(type_df['twitter_name'])]

    # Sort column_coordinates_type by x-coordinate values
    column_coordinates_type_sorted = column_coordinates_type.sort_values(by=0)

    # Create a scatter plot
    plt.figure(figsize=(10, 10))

    # Assign each unique twitter_name a unique y-value based on sorted order
    y_values = np.linspace(0, 1, len(column_coordinates_type_sorted))

    scatter = plt.scatter(column_coordinates_type_sorted[0], y_values, c=color, alpha = 1)

    # Add labels
    texts = []
    for i, twitter_name in enumerate(column_coordinates_type_sorted.index):
        texts.append(plt.text(column_coordinates_type_sorted[0][i], y_values[i], twitter_name))

    # Adjust text to avoid overlaps
    adjust_text(texts)

    # Create a legend
    legend_elements = [Patch(facecolor=color, edgecolor=color, label=type_to_plot)]
    plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.yticks([])

    plt.show()