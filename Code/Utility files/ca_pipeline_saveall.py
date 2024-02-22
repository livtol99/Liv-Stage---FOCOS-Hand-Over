import os
import sys
from collections import defaultdict
import io
import community as community_louvain
import matplotlib.patches as mpatches
import networkx as nx
import pandas as pd
import prince
import numpy as np
from matplotlib import pyplot as plt
from networkx.algorithms import bipartite
from netgraph import Graph


class PipelineCorAnSave:
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
        try:
            # Create a new bipartite graph
            B = nx.DiGraph()  # initialize a new directed graph

            # Add nodes with the node attribute "bipartite"
            B.add_nodes_from(self.data_subset['follower_id'].unique(), bipartite=0)  # adding a node for each unique follower. Bipartite = 0 assigns followers to the first set in the graph.  Set 1 in the bipartite graph
            B.add_nodes_from(self.data_subset['twitter_name'].unique(), bipartite=1)  # adding a node for each unique marker. Set 2 in the bipartite graph

            # Add edges
            B.add_edges_from(list(zip(self.data_subset['follower_id'], self.data_subset['twitter_name'])))  # edges are directed from the first to the second element. So direction is; follower --> Marker

            self.B = B  # store the graph in an instance variable for later use
            
            #call the sanity check method here to avoid AttributeErrors 
            # self.sanity_checks()
            # self.connectedness()
            # self.plot_degree_cdf()
        except Exception as e:
            print(f"Error occurred while creating bipartite graph: {str(e)}")
    
    def sanity_checks(self):
        if not hasattr(self, 'B'):
            raise AttributeError("Bipartite graph not created. Call create_bipartite_graph first.")
        num_nodes = self.B.number_of_nodes()
        num_edges = self.B.number_of_edges()
        num_rows = len(self.data_subset)

        B_undirected = self.B.to_undirected()
        is_connected = nx.is_connected(B_undirected)

        return {
            'Number of nodes': num_nodes,
            'Edge number matches number of rows': num_edges == num_rows,
            'Is the graph connected': is_connected
        }

    def connectedness(self):
        # Calculate weakly connected components
        connected_components = list(nx.weakly_connected_components(self.B))

        # Get the number of connected components
        num_connected_components = len(connected_components)

        # Get the size of the largest connected component
        size_largest_connected_component = max(len(c) for c in connected_components)

        return {
            'Number of connected components': num_connected_components,
            'Size of largest connected component': size_largest_connected_component
        }
    
    import matplotlib.pyplot as plt
    import numpy as np

    def plot_degree_cdf(self):
        import io
        import matplotlib.pyplot as plt
        import pickle

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
        fig1 = plt.figure()
        plt.loglog(bin_edges[:-1], ccdf, marker='.')
        plt.xlabel('Out-Degree')
        plt.ylabel('CCDF')
        plt.title('CCDF of Out-Degrees on a log-log scale')

        # Calculate the complementary cumulative distribution function (CCDF)for in_degrees 
        counts, bin_edges = np.histogram(degrees_in, bins=range(1, max(degrees_in) + 1), density=True)
        cum_counts = np.cumsum(counts)
        ccdf = 1 - cum_counts

        # Plot the CCDF on a log-log scale
        fig2 = plt.figure()
        plt.loglog(bin_edges[:-1], ccdf, marker='.', color='green')
        plt.xlabel('In-Degree')
        plt.ylabel('CCDF')
        plt.title('CCDF of In-Degrees on a log-log scale')

        # Convert figures to bytes
        buf1 = io.BytesIO()
        fig1.savefig(buf1, format='png')
        buf1.seek(0)

        buf2 = io.BytesIO()
        fig2.savefig(buf2, format='png')
        buf2.seek(0)

        return buf1, buf2
    
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

        # Convert figure to bytes
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)

        return buf
    
    def calculate_communities(self):
        if not hasattr(self, 'G2_markers'):
            self.marker_projection()
        # Compute the best partition using the Louvain method
        partition = community_louvain.best_partition(self.G2_markers) #result is a dict where key = node and value = community

        # Get the number of unique communities
        num_communities = len(set(partition.values()))

        # Store the partition for later use
        self.partition = partition

        return {
            'Number of communities': num_communities,
            'Partition': partition
        }
    
    
    # Main method to run all the graph checks
    def perform_graph_checks(self):
        self.create_bipartite_graph()
        sanity_checks_output = self.sanity_checks()
        connectedness_output = self.connectedness()
        plot_degree_cdf_output = self.plot_degree_cdf()
        self.marker_projection()
        plot_w_marker_relations_output = self.plot_w_marker_relations()
        calculate_communities_output = self.calculate_communities()

        return {
            'sanity_checks': sanity_checks_output,
            'connectedness': connectedness_output,
            'plot_degree_cdf': plot_degree_cdf_output,
            'plot_w_marker_relations': plot_w_marker_relations_output,
            'calculate_communities': calculate_communities_output
        }

    
    # CA fitting methods
    def create_contingency_table(self):
        # Create the contingency table
        self.contingency_table = pd.crosstab(self.data_subset['follower_id'], self.data_subset['twitter_name'])

    
    def perform_ca_analysis(self, save_path, n_components=100, n_iter=100):
        try:
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
            ca.fit(self.contingency_table)
            self.ca = ca

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

        except Exception as e:
            print(f"Error occurred while performing CA analysis: {str(e)}")
 

    def plot_variance(self):
        # Get the percentage of variance
        percentage_of_variance = self.ca.percentage_of_variance_

        # Create a range of numbers for x axis
        dimensions = range(1, len(percentage_of_variance) + 1)

        # Create the plot
        plt.figure(figsize=(10, 7))
        plt.bar(dimensions, percentage_of_variance)
        plt.xlabel('Dimensions')
        plt.ylabel('Percentage of Variance')
        plt.title('Percentage of Total Variance per Dimension')

        # Convert figure to bytes
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)

        return buf

    def get_unique_filepath(self, filepath):
        import os

        # If the file doesn't exist, return the original filepath
        if not os.path.exists(filepath):
            return filepath

        # If the file exists, ask the user if they want to overwrite it
        overwrite = input(f"{filepath} already exists. Do you want to overwrite it? (yes/no): ")

        if overwrite.lower() == 'yes':
            return filepath

        # If the user doesn't want to overwrite, add a unique suffix to the filename
        base, ext = os.path.splitext(filepath)
        i = 1
        while os.path.exists(filepath):
            filepath = f"{base}_{i}{ext}"
            i += 1

        return filepath
    
    def save_outputs(self, outputs, filenames, formats, save_path):
        # Create a new directory 'outs' if it doesn't exist
        outs_dir_path = os.path.join(save_path, 'outs')
        if not os.path.exists(outs_dir_path):
            os.makedirs(outs_dir_path)

        # Loop over the outputs, filenames, and formats
        for output, filename, format in zip(outputs, filenames, formats):
            # Save the output to a file in the 'outs' directory
            # If a file already exists, add a unique suffix to the filename
            output_file_path = os.path.join(outs_dir_path, filename)
            output_file_path = self.get_unique_filepath(output_file_path)

            if format == 'csv':
                output.to_csv(output_file_path, index=False)
            elif format == 'png':
                output.savefig(output_file_path)
            elif format == 'txt':
                with open(output_file_path, 'w') as f:
                    f.write(output)
            else:
                raise ValueError(f"Unsupported format: {format}")
    

    def perform_ca_pipeline(self, save_path):
        print("Creating contingency table...")
        self.create_contingency_table()
        print("Performing CA analysis. Might take some time...")
        self.perform_ca_analysis(save_path, n_components=100, n_iter=100)
        print("Plotting variance...")
        plot_variance_output = self.plot_variance()

        return {
            'plot_variance': plot_variance_output
        }


    def run_all(self, save_path):
        print("Starting graph checks...")
        graph_checks_outputs = self.perform_graph_checks()
        print("Graph checks complete. Starting CA fitting pipeline...")
        ca_pipeline_outputs = self.perform_ca_pipeline(save_path)
        print("CA pipeline complete. Saving outputs...")

        # Save the outputs
        self.save_outputs(
            [graph_checks_outputs['sanity_checks'], graph_checks_outputs['connectedness'], graph_checks_outputs['plot_degree_cdf'], graph_checks_outputs['plot_w_marker_relations'], graph_checks_outputs['calculate_communities'], ca_pipeline_outputs['plot_variance']],
            ['sanity_checks.txt', 'connectedness.txt', 'plot_degree_cdf.png', 'plot_w_marker_relations.png', 'calculate_communities.json', 'plot_variance.png'],
            ['txt', 'txt', 'png', 'png', 'json', 'png'],
            save_path
        )

        print("All done!")


