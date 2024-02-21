import os
import sys
from collections import defaultdict

import community as community_louvain
import matplotlib.patches as mpatches
import networkx as nx
import pandas as pd
import prince
import numpy as np
from matplotlib import pyplot as plt
from networkx.algorithms import bipartite
from netgraph import Graph


class CA_pipeline:
    def __init__(self, data_subset, data_subset_name):
        if not isinstance(data_subset, pd.DataFrame):
            raise ValueError("data_subset must be a pandas DataFrame")

        required_columns = ['follower_id', 'twitter_name']
        if not all(column in data_subset.columns for column in required_columns):
            raise ValueError(f"data_subset must contain the following columns: {required_columns}")

        self.data_subset = data_subset
        self.edgelist_name = self.get_edgelist_name(data_subset_name)

    def get_edgelist_name(self, data_subset_name):
        # Extract the name from the DataFrame
        # This is a placeholder - replace this with your actual code
        return data_subset_name

    # Graph checks methods
    def create_bipartite_graph(self):
        # Create a new bipartite graph
        B = nx.DiGraph()  # initialize a new directed graph

        # Add nodes with the node attribute "bipartite"
        B.add_nodes_from(self.data_subset['follower_id'].unique(), bipartite=0)  # adding a node for each unique follower. Bipartite = 0 assigns followers to the first set in the graph.  Set 1 in the bipartite graph
        B.add_nodes_from(self.data_subset['twitter_name'].unique(), bipartite=1)  # adding a node for each unique marker. Set 2 in the bipartite graph

        # Add edges
        B.add_edges_from(list(zip(self.data_subset['follower_id'], self.data_subset['twitter_name'])))  # edges are directed from the first to the second element. So direction is; follower --> Marker

        self.B = B  # store the graph in an instance variable for later use
        
        #call the sanity check method here to avoid AttributeErrors 
        self.sanity_checks()
    
    def sanity_checks(self):
        if not hasattr(self, 'B'):
            raise AttributeError("Bipartite graph not created. Call create_bipartite_graph first.")
        num_nodes = self.B.number_of_nodes()
        num_edges = self.B.number_of_edges()
        num_rows = len(self.data_subset)

        print("Number of nodes:", num_nodes)
        if num_edges == num_rows:
            print("Edge number is sane - matches the number of rows in the inputted edgelist")

        B_undirected = self.B.to_undirected()
        print("Is the graph connected?", nx.is_connected(B_undirected))

    def connectedness(self):
        # Calculate weakly connected components
        connected_components = list(nx.weakly_connected_components(self.B))

        # Print the number of connected components
        print("Number of connected components:", len(connected_components))

        # Print the size of the largest connected component
        print("Size of largest connected component:", max(len(c) for c in connected_components))
    
    def plot_degree_cdf(self):
        followers = [n for n, d in self.B.nodes(data=True) if d['bipartite']==0]
        markers = [n for n, d in self.B.nodes(data=True) if d['bipartite']==1]

        in_degrees = [d for n, d in self.B.in_degree(markers)]
        out_degrees = [d for n, d in self.B.out_degree(followers)] #get out degrees for followers

        degrees_out = np.array(out_degrees)
        degrees_in = np.array(in_degrees)

        # Calculate the complementary cumulative distribution function (CCDF) for out_degrees
        counts, bin_edges = np.histogram(degrees_out, bins=range(1, max(degrees_out) + 1), density=True)
        cum_counts = np.cumsum(counts)
        ccdf = 1 - cum_counts

        # Plot the CCDF on a log-log scale
        plt.loglog(bin_edges[:-1], ccdf, marker='.')
        plt.xlabel('Out-Degree')
        plt.ylabel('CCDF')
        plt.title('CCDF of Out-Degrees on a log-log scale')
        plt.show()

        # Calculate the complementary cumulative distribution function (CCDF)for in_degrees 
        counts, bin_edges = np.histogram(degrees_in, bins=range(1, max(degrees_in) + 1), density=True)
        cum_counts = np.cumsum(counts)
        ccdf = 1 - cum_counts

        # Plot the CCDF on a log-log scale
        plt.loglog(bin_edges[:-1], ccdf, marker='.', color='green')
        plt.xlabel('In-Degree')
        plt.ylabel('CCDF')
        plt.title('CCDF of In-Degrees on a log-log scale')
        plt.show()
    
    def marker_projection(self):
        if not hasattr(self, 'B'):
            raise AttributeError("Bipartite graph not created. Call create_bipartite_graph first.")
        # Separate nodes into two sets
        followers, markers = bipartite.sets(self.B) #first set is followers (set 0) and second is markers (set 1)

        # Convert to undirected graph
        B_undirected = self.B.to_undirected()

        # Create the projection for markers
        G_markers = bipartite.projected_graph(B_undirected, markers) #unweighted projection

        # Weighted projection for markers
        G2_markers = bipartite.weighted_projected_graph(B_undirected, markers)

        # Store the projections for later use
        self.G_markers = G_markers
        self.G2_markers = G2_markers
    
    def plot_w_marker_relations(self):
        if not hasattr(self, 'G2_markers'):
            self.marker_projection()
        color_dict = {'consumption': 'blue', 'information': 'yellowgreen', 'football clubs': 'mediumvioletred', 'education': 'darkorange'}

        # Get the unique types and assign a color to each one
        unique_types = self.data_subset['type2'].unique()
        type_color = {utype: color_dict.get(utype, 'gray') for utype in unique_types}  # Use gray for missing types

        # Create a dictionary that maps each twitter_name to its type
        twitter_name_to_type = dict(zip(self.data_subset['twitter_name'], self.data_subset['type2']))

        # Create a list of colors for each node in the graph
        node_colors = [type_color[twitter_name_to_type.get(node, 'gray')] for node in self.G2_markers.nodes()]  # Use gray for missing types

        # Draw the graph with node colors
        plt.figure(figsize=(10, 10))  # Increase figure size
        pos = nx.spring_layout(self.G2_markers, weight='weight')  # Use spring layout

        # Draw edges with increased alpha
        nx.draw_networkx_edges(self.G2_markers, pos, alpha=0.09, width=0.1)

        # Draw nodes with original alpha and node colors
        nx.draw_networkx_nodes(self.G2_markers, pos, node_color=node_colors, node_size=40, alpha=0.5)

        # Calculate the center of the graph by averaging the positions of all nodes
        center = np.array([0.0, 0.0])  # Initialize as float array
        for coord in pos.values():
            center += np.array(coord)
        center /= len(pos)

        # Define the distance for nodes to be considered "outside the main cluster"
        distance = 0.7

        # Draw labels for nodes outside the main cluster
        for node, coord in pos.items():
            if np.linalg.norm(np.array(coord) - center) > distance:
                plt.text(coord[0] + 0.02, coord[1] + 0.02, node, fontsize=7)

        # Create legend
        patches = [mpatches.Patch(color=color, label=utype) for utype, color in type_color.items()]
        plt.legend(handles=patches, loc='center left', bbox_to_anchor=(0.95, 0.5), bbox_transform=plt.gcf().transFigure)

        # Remove axes
        plt.axis('off')

        plt.show()
    
    def calculate_communities(self):
        # Compute the best partition using the Louvain method
        partition = community_louvain.best_partition(self.G2_markers) #result is a dict where key = node and value = community

        # Get the number of unique communities
        num_communities = len(set(partition.values()))

        print(f"Number of communities: {num_communities}")

        # Store the partition for later use
        self.partition = partition
    
    
    # Main method to run all the graph checks
    def perform_graph_checks(self):
        self.create_bipartite_graph()
        self.sanity_checks()
        self.connectedness()
        self.plot_degree_cdf()
        self.marker_projection()
        self.plot_w_marker_relations()
        self.calculate_communities()

    
    # CA fitting methods
    def create_contingency_table(self):
        # Create the contingency table
        self.contingency_table = pd.crosstab(self.data_subset['follower_id'], self.data_subset['twitter_name'])

    
    def perform_ca_analysis(self, save_path, n_components=100, n_iter=100):
        # Initialize a CA object
        ca = prince.CA(
            n_components=n_components,  # Number of components to keep
            n_iter=n_iter,  # Number of iterations for the power method
            copy=True,  # Whether to overwrite the data matrix
            check_input=True,  # Whether to check the input for NaNs and Infs
            engine='sklearn',  # Whether to perform computations in C or Python
            random_state=42  # Random seed for reproducibility
        )

        # Fit the CA model on the contingency table
        self.ca = ca.fit(self.contingency_table)

        # Get the coordinates of the rows (followers) and columns (brands) 
        row_coordinates = ca.row_coordinates(self.contingency_table)
        column_coordinates = ca.column_coordinates(self.contingency_table)

        # Create a new directory with the name of the edgelist if it doesn't exist
        new_dir_path = os.path.join(save_path, f"{self.edgelist_name}_coords")
        if not os.path.exists(new_dir_path):
            os.makedirs(new_dir_path)

        # Save the row and column coordinates to CSV files in the new directory
        # If a file already exists, add a unique suffix to the filename
        row_file_path = os.path.join(new_dir_path, f'{self.edgelist_name}_row_coordinates.csv')
        column_file_path = os.path.join(new_dir_path, f'{self.edgelist_name}_column_coordinates.csv')
        row_file_path = self.get_unique_filepath(row_file_path)
        column_file_path = self.get_unique_filepath(column_file_path)
        row_coordinates.iloc[:, :4].to_csv(row_file_path)
        column_coordinates.iloc[:, :4].to_csv(column_file_path)

    def get_unique_filepath(self, filepath):
        # If the file doesn't exist, return the original filepath
        if not os.path.exists(filepath):
            return filepath

        # If the file exists, add a unique suffix to the filename
        base, ext = os.path.splitext(filepath)
        i = 1
        while os.path.exists(filepath):
            filepath = f"{base}_{i}{ext}"
            i += 1

        return filepath

    def plot_variance_per_dimension(self):
        # Assuming 'ca' is your prince.CA object
        percentage_of_variance = self.ca.explained_inertia_

        # Create a range of numbers for x axis
        dimensions = range(1, len(percentage_of_variance) + 1)

        # Create the plot
        plt.figure(figsize=(10, 7))
        plt.bar(dimensions, percentage_of_variance)
        plt.xlabel('Dimensions')
        plt.ylabel('Percentage of Variance')
        plt.title('Percentage of Total Variance per Dimension')
        plt.show()


    def perform_ca_pipeline(self, save_path):
        self.create_contingency_table()
        self.perform_ca_analysis(save_path, n_components=100, n_iter=100)
        self.plot_variance_per_dimension()

    # Run all
    def run_all(self):
        self.perform_graph_checks()
        self.perform_ca_pipeline()

