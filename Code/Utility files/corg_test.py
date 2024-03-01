import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from mpl_toolkits.mplot3d import Axes3D
from corg import BenchmarkDimension, DiscoverDimension

class CorgPipeline:
    def __init__(self, file_number, n_dimensions):
        self.file_number = file_number
        self.n_dimensions = n_dimensions
        self.file_path = f"/home/livtollanes/NewData/coordinates/m{file_number}_coords/m{file_number}_column_coordinates.csv"
        print(f"Constructed file path: {self.file_path}") 
        self.df = pd.read_csv(self.file_path, index_col=0)
        self.models = {str(dim): BenchmarkDimension() for dim in range(n_dimensions) if str(dim) in self.df.columns}
        self.label_cols = {0: 'blue', 1: 'red'}

    def print_head_and_label_counts(self):
        print(self.df.head())
        print(self.df['label'].value_counts())

    def plot_data(self):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        for label in self.df['label'].unique():
            df = self.df[self.df['label'] == label]
            ax.scatter(df['0'], df['1'], df['2'], alpha=0.8, s=8, c=self.label_cols[label], label=label)

        ax.set_xlabel('d1')
        ax.set_ylabel('d2')
        ax.set_zlabel('d3')

        plt.legend()
        plt.show()

        fig = plt.figure(figsize=(20,4))
        ax = {i:fig.add_subplot(1,self.n_dimensions,i+1) for i in range(self.n_dimensions)}

        for i in range(self.n_dimensions):
            if str(i) in self.df.columns:
                sn.kdeplot(data=self.df, x=str(i), hue='label', ax=ax[i], palette=['red','blue'])
                ax[i].set_xlabel(f'dim {i+1}')

        plt.show()

    def func1_metrics(self):
        for dim, model in self.models.items():
            Y = self.df[['twitter_name','label']].rename(columns={'twitter_name': 'entity'})
            X = self.df[['twitter_name', dim]].rename(columns={'twitter_name': 'entity', dim: 'dimension'})
            model.fit(X, Y)
            print(f'Dimension {dim}: Precision={model.precision_train_:.3f}, Recall={model.recall_train_:.3f}, F1-score={model.f1_score_train_:.3f}')

    def drop_na(self):
        self.df = self.df.dropna(subset=['label'])

    def func1(self):
        self.drop_na()
        print(f"Outputs for CORG functionality 1. Model number {self.file_number}")
        self.print_head_and_label_counts()
        self.plot_data()
        self.func1_metrics()

    def fit_discover_model(self, print_output=True):
        self.discover_model = DiscoverDimension()
        Y_d = self.df[['twitter_name','label']].rename(columns={'twitter_name': 'entity'})
        X_d = self.df[['twitter_name'] + [str(i) for i in range(self.n_dimensions)]].rename(columns={'twitter_name': 'entity'})
        self.discover_model.fit(X_d, Y_d)
        if print_output:
            print("Decision boundary:")
            print(self.discover_model.model_decision_boundary_)
            print("Hyperplane Unit Normal (new found direction):")
            print(self.discover_model.decision_hyperplane_unit_normal)

    def plot_discover_model(self):
        normal = self.discover_model.decision_hyperplane_unit_normal

        # Calculate the number of subplots needed
        num_subplots = self.n_dimensions * (self.n_dimensions - 1) // 2

        # Create a figure with the appropriate number of subplots
        fig, axs = plt.subplots(num_subplots, figsize=(10, 4 * num_subplots))

        # Keep track of which subplot we're on
        subplot_index = 0

        # Loop over all pairs of dimensions
        for i in range(self.n_dimensions):
            for j in range(i + 1, self.n_dimensions):
                ax = axs[subplot_index]

                # Plot the data for this pair of dimensions
                ax.scatter(self.df.loc[self.df['label'] == 0, str(i)], self.df.loc[self.df['label'] == 0, str(j)], alpha=0.7, c='red')
                ax.scatter(self.df.loc[self.df['label'] == 1, str(i)], self.df.loc[self.df['label'] == 1, str(j)], alpha=0.7, c='blue')

                # Draw the decision boundary
                ax.arrow(0, 0, normal[i], normal[j], color='green', head_width=0.2, head_length=0.2)

                # Set the labels for this subplot
                ax.set_xlabel(str(i))
                ax.set_ylabel(str(j))

                # Move on to the next subplot
                subplot_index += 1

        plt.show()

    def plot_discover_model_3d(self):
        normal = self.discover_model.decision_hyperplane_unit_normal

        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(self.df.loc[self.df['label'] == 0,'0'],self.df.loc[self.df['label'] == 0,'1'],self.df.loc[self.df['label'] == 0,'2'],alpha=0.7,c='red')
        ax.scatter(self.df.loc[self.df['label'] == 1,'0'],self.df.loc[self.df['label'] == 1,'1'],self.df.loc[self.df['label'] == 1,'2'],alpha=0.7,c='blue')
        ax.quiver(0, 0, 0, normal[0], normal[1], normal[2],color='green',length=1,normalize=True)
        ax.set_xlabel('0'),ax.set_ylabel('1'),ax.set_zlabel('2')
    
    def func2_metrics(self):
        if not hasattr(self, 'discover_model'):
            self.fit_discover_model(print_output=False)
        print(f"Functionality 2 metrics for model number {self.file_number}")
        print('Precision=%.3f, Recall=%.3f, F1-score=%.3f, '%(self.discover_model.precision_train_,self.discover_model.recall_train_,self.discover_model.f1_score_train_))

    def func2(self):
        self.drop_na()
        print(f"Outputs for CORG functionality 2. Model number {self.file_number}")
        self.fit_discover_model()
        self.plot_discover_model()
        self.plot_discover_model_3d()
        self.func2_metrics()