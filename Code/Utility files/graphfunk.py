import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from adjustText import adjust_text
import networkx as nx



def plot_type(df, column_coordinates, type_to_plot, color, dimension):
    # Filter for the specified type
    type_df = df[df['type2'] == type_to_plot]

    # Use only specified type twitter_names in column_coordinates
    column_coordinates_type = column_coordinates[column_coordinates.index.isin(type_df['twitter_name'])]

    # Sort column_coordinates_type by the specified dimension values
    column_coordinates_type_sorted = column_coordinates_type.sort_values(by=dimension)

    # Create a scatter plot
    plt.figure(figsize=(15, 10))

    # Assign each unique twitter_name a unique y-value based on sorted order
    y_values = np.linspace(0, 1, len(column_coordinates_type_sorted))

    scatter = plt.scatter(column_coordinates_type_sorted[dimension], y_values, c=color, alpha = 1)

    # Add labels
    texts = []
    for i, twitter_name in enumerate(column_coordinates_type_sorted.index):
        texts.append(plt.text(column_coordinates_type_sorted[dimension][i], y_values[i], twitter_name))

    # Adjust text to avoid overlaps
    adjust_text(texts, expand_points=(1.5, 1.5), expand_text=(1.2, 1.2), force_points=0.2)

    # Create a legend
    legend_elements = [Patch(facecolor=color, edgecolor=color, label=type_to_plot)]
    plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.yticks([])

    plt.show()

def plot_brands(df, dimension, fontsize=7):
    # Manually specify a color palette
    color_dict = {'consumption': 'blue', 'information': 'yellowgreen', 'football clubs': 'mediumvioletred', 'education': 'darkorange'}

    # Sort df by the specified dimension values
    df_sorted = df.sort_values(by=dimension)

    # Map 'type2' to colors
    df_sorted['color'] = df_sorted['type2'].map(color_dict)

    # Create a scatter plot
    plt.figure(figsize=(20, 10))

    # Assign each unique twitter_name a unique y-value based on sorted order
    y_values = np.linspace(0, 1, len(df_sorted))

    scatter = plt.scatter(df_sorted[dimension], y_values, c=df_sorted['color'], alpha = 1)

    # For each point, add a text label with an arrow
    for i in range(len(df_sorted)):
        twitter_name = df_sorted['twitter_name'].iloc[i]
        plt.annotate(twitter_name, 
                     (df_sorted[dimension].iloc[i], y_values[i]), 
                     textcoords="offset points", 
                     xytext=(-30,30), 
                     ha='center', 
                     fontsize=fontsize,  # Set font size here
                     arrowprops=dict(arrowstyle='->', lw=1.5))

    legend_elements = [Patch(facecolor=color, edgecolor=color, label=type2) for type2, color in color_dict.items()]
    plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.yticks([])

    plt.show()

def community_layout(g, partition):
    """
    Compute the layout for a modular graph.

    Arguments:
    g -- networkx.Graph or networkx.DiGraph instance
    partition -- dict mapping node -> community

    Returns:
    pos -- dict mapping node -> position
    """
    pos_communities = _position_communities(g, partition, scale=3.)
    pos_nodes = _position_nodes(g, partition, scale=1.)
    pos = dict()
    for node in g.nodes():
        pos[node] = pos_communities[node] + pos_nodes[node]
    return pos

def _position_communities(g, partition, **kwargs):
    """
    Compute the layout for communities.

    Arguments:
    g -- networkx.Graph or networkx.DiGraph instance
    partition -- dict mapping node -> community

    Returns:
    pos -- dict mapping node -> position
    """
    between_community_edges = _find_between_community_edges(g, partition)
    communities = set(partition.values())
    hypergraph = nx.DiGraph()
    hypergraph.add_nodes_from(communities)
    for (ci, cj), edges in between_community_edges.items():
        hypergraph.add_edge(ci, cj, weight=len(edges))
    pos_communities = nx.spring_layout(hypergraph, **kwargs)
    pos = dict()
    for node, community in partition.items():
        pos[node] = pos_communities[community]
    return pos

def _find_between_community_edges(g, partition):
    """
    Find edges that are between communities.

    Arguments:
    g -- networkx.Graph or networkx.DiGraph instance
    partition -- dict mapping node -> community

    Returns:
    edges -- dict mapping (community1, community2) -> [edges]
    """
    edges = dict()
    for (ni, nj) in g.edges():
        ci = partition[ni]
        cj = partition[nj]
        if ci != cj:
            try:
                edges[(ci, cj)] += [(ni, nj)]
            except KeyError:
                edges[(ci, cj)] = [(ni, nj)]
    return edges

def _position_nodes(g, partition, **kwargs):
    """
    Compute the layout for nodes.

    Arguments:
    g -- networkx.Graph or networkx.DiGraph instance
    partition -- dict mapping node -> community

    Returns:
    pos -- dict mapping node -> position
    """
    communities = dict()
    for node, community in partition.items():
        try:
            communities[community] += [node]
        except KeyError:
            communities[community] = [node]
    pos = dict()
    for ci, nodes in communities.items():
        subgraph = g.subgraph(nodes)
        pos_subgraph = nx.spring_layout(subgraph, **kwargs)
        pos.update(pos_subgraph)
    return pos



#Need to clean up this entire code, there are multiple functions for similar plots. Here are the ones developed via the CORG pipeline
def plot_all_brands_together(df, dimension, fontsize= 6):
    # Manually specify a color palette
    color_dict = {'consumption': 'blue', 'information': 'yellowgreen', 'football clubs': 'mediumvioletred', 'education': 'darkorange'}

    # Sort df by the specified dimension values
    df_sorted = df.sort_values(by=dimension)

    # Map 'type2' to colors
    df_sorted['color'] = df_sorted['type2'].map(color_dict)

    # Create a scatter plot
    plt.figure(figsize=(20, 10))

    # Assign each unique twitter_name a unique y-value based on sorted order
    y_values = np.linspace(0, 1, len(df_sorted))

    scatter = plt.scatter(df_sorted[dimension], y_values, c=df_sorted['color'], alpha = 0.6)

    # For each point, add a text label with an arrow
    for i in range(len(df_sorted)):
        twitter_name = df_sorted['twitter_name'].iloc[i]
        xytext = (-30,30) if i % 2 == 0 else (60,-30)  # Alternate label position based on index
        plt.annotate(twitter_name, 
                     (df_sorted[dimension].iloc[i], y_values[i]), 
                     textcoords="offset points", 
                     xytext=xytext, 
                     ha='center', 
                     fontsize=fontsize,  # Set font size here
                     arrowprops=dict(arrowstyle='->', lw=1.5))

    legend_elements = [Patch(facecolor=color, edgecolor=color, label=type2) for type2, color in color_dict.items()]
    plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.yticks([])

    plt.show()

def full_plot_function(df, dimension, types_to_plot, type_style, fontsize=6):
    """
    This function is to be used for plotting all, or some, brands along a dimension of desire.
    Creates a scatter plot of the data in the provided dataframe along the specified dimension. 
    The points are colored according to their type, and a legend is included that shows the color associated with each type.
    Option to plot the old or new_types type classification.
    Option to plot all or some brands only. 

    Parameters:
    df (DataFrame): The input dataframe.
    dimension (str): The column name in the dataframe to be used for the x-axis of the scatter plot.
    types_to_plot (list or str): A list of types to be included in the plot, or 'all' to include all types.
    type_style (str): Determines whether to use the old or new type classification. Expected values are 'old_type' or 'new_type'.
    fontsize (int, optional): Controls the font size of the text labels. Default is 6.
    """
    # Manually specify a color palette for 12 types
    color_dict_old = {'media': 'blue', 'clubs de football': 'yellowgreen', 'sport': 'mediumvioletred', 'grande distribution': 'darkorange',
                      'universities': 'red', 'commerce': 'purple', 'chain restaurants': 'brown', 'luxe vetements et malls': 'pink',
                      'magazine': 'gray', 'party': 'olive', 'ecoles de commerce': 'cyan', 'Lycées professionels': 'magenta'}
    color_dict_new = {'consumption': 'blue', 'information': 'yellowgreen', 'football clubs': 'mediumvioletred', 'education': 'darkorange'}

    # Choose the color dictionary and type column based on type_style
    if type_style == 'old':
        color_dict = color_dict_old
        type_column = 'type'
    elif type_style == 'new':
        color_dict = color_dict_new
        type_column = 'type2'
    else:
        raise ValueError('Invalid type_style. Expected "old_type" or "new_type".')

    # If types_to_plot is 'all', get all unique types from the data
    if types_to_plot == 'all':
        types_to_plot = df[type_column].unique()

    # Sort df by the specified dimension values and filter by type_to_plot
    df_sorted = df[df[type_column].isin(types_to_plot)].sort_values(by=dimension)

    # Map 'type' or 'type2' to colors
    df_sorted['color'] = df_sorted[type_column].map(color_dict)

    # Create a scatter plot
    plt.figure(figsize=(20, 10))

    # Assign each unique twitter_name a unique y-value based on sorted order
    y_values = np.linspace(0, 1, len(df_sorted))

    scatter = plt.scatter(df_sorted[dimension], y_values, c=df_sorted['color'], alpha = 0.6)

    # For each point, add a text label with an arrow
    for i in range(len(df_sorted)):
        twitter_name = df_sorted['twitter_name'].iloc[i]
        # If only one type is being plotted, set xytext to (50, -5)
        xytext = (50, -5) if len(df_sorted['type'].unique()) == 1 else (-30,30) if i % 2 == 0 else (60,-30)
        plt.annotate(twitter_name, 
                     (df_sorted[dimension].iloc[i], y_values[i]), 
                     textcoords="offset points", 
                     xytext=xytext, 
                     ha='center', 
                     fontsize=fontsize,  # Set font size here
                     arrowprops=dict(arrowstyle='->', lw=1.5))

    # Create legend elements based on the unique types in df_sorted
    legend_elements = [Patch(facecolor=color_dict[type2], edgecolor=color_dict[type2], label=type2) for type2 in df_sorted[type_column].unique()]

    plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.yticks([])

    plt.show()