import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from mpl_toolkits.mplot3d import Axes3D
from corg import BenchmarkDimension, DiscoverDimension


class CorgPipeline:
    def __init__(self, file_number):
        self.file_number = file_number
        self.file_path = f"/home/livtollanes/NewData/coordinates_labeled_subsets/m{file_number}_lab_coords/m{file_number}_lab_column_coordinates.csv"
        self.df = pd.read_csv(self.file_path, index_col=0)
        self.models = {dim: BenchmarkDimension() for dim in ['0', '1', '2', '3'] if dim in self.df.columns}
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
        ax = {1:fig.add_subplot(1,4,1), 2:fig.add_subplot(1,4,2), 3:fig.add_subplot(1,4,3), 4:fig.add_subplot(1,4,4)}

        for i in range(4):
            if str(i) in self.df.columns:
                sn.kdeplot(data=self.df, x=str(i), hue='label', ax=ax[i+1], palette=['red','blue'])
                ax[i+1].set_xlabel(f'dim {i+1}')

        plt.show()

    def func1_metrics(self):
        for dim, model in self.models.items():
            Y = self.df[['twitter_name','label']].rename(columns={'twitter_name': 'entity'})
            X = self.df[['twitter_name', dim]].rename(columns={'twitter_name': 'entity'})
            model.fit(X, Y)
            print(f'Dimension {dim}: Precision={model.precision_train_:.3f}, Recall={model.recall_train_:.3f}, F1-score={model.f1_score_train_:.3f}')

    def func1(self):
        print(f"Outputs for CORG functionality 1. Model number {self.file_number}")
        self.print_head_and_label_counts()
        self.plot_data()
        self.func1_metrics()
    
    
    def fit_discover_model(self, print_output=True):
        self.discover_model = DiscoverDimension()
        Y_d1 = self.df[['twitter_name','label']].rename(columns={'twitter_name': 'entity'})
        X_d1 = self.df[['twitter_name','0', '1', '2', '3']].rename(columns={'twitter_name': 'entity'})
        self.discover_model.fit(X_d1, Y_d1)
        if print_output:
            print("Decision boundary:")
            print(self.discover_model.model_decision_boundary_)
            print("Hyperplane Unit Normal:")
            print(self.discover_model.decision_hyperplane_unit_normal)

    def plot_discover_model(self):
        normal = self.discover_model.decision_hyperplane_unit_normal

        fig = plt.figure(figsize=(10,4))
        ax = {1:fig.add_subplot(1,2,1),2:fig.add_subplot(1,2,2)}
        ax[1].scatter(self.df.loc[self.df['label'] == 0,'0'],self.df.loc[self.df['label'] == 0,'1'],alpha=0.7,c='red')
        ax[1].scatter(self.df.loc[self.df['label'] == 1,'0'],self.df.loc[self.df['label'] == 1,'1'],alpha=0.7,c='blue')
        ax[1].arrow(0, 0, normal[0], normal[1],color='green',head_width=0.2,head_length=0.2)
        ax[1].set_xlabel('0'),ax[1].set_ylabel('1'),
        ax[2].scatter(self.df.loc[self.df['label'] == 0,'1'],self.df.loc[self.df['label'] == 0,'2'],alpha=0.7,c='red')
        ax[2].scatter(self.df.loc[self.df['label'] == 1,'1'],self.df.loc[self.df['label'] == 1,'2'],alpha=0.7,c='blue')
        ax[2].arrow(0, 0, normal[1], normal[2],color='green',head_width=0.2,head_length=0.2)
        ax[2].set_xlabel('1'),ax[2].set_ylabel('2')

    def func2_metrics(self):
        if not hasattr(self, 'discover_model'):
            self.fit_discover_model(print_output=False)
        print(f"Functionality 2 metrics for model number {self.file_number}")
        print('Precision=%.3f, Recall=%.3f, F1-score=%.3f, '%(self.discover_model.precision_train_,self.discover_model.recall_train_,self.discover_model.f1_score_train_))

    def func2(self):
        print(f"Outputs for CORG functionality 2. Model number {self.file_number}")
        self.fit_discover_model()
        self.plot_discover_model()
        self.func2_metrics()