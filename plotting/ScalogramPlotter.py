import matplotlib.pyplot as plt
import numpy as np
import os

# so this should be able to take in a data dir, and a save dir, get the data from the data dir, plot it, and then save it. 

class ScalogramPlotter:
    """
    This class is designed to handle the differnt kinds of scalogram plots that we want to generate. 
    It takes in the type of dimensions of the plot, the colormap to use, 
    the directory of the data, and the directory to save the plots.
    """
    def __init__(self, dimensions: tuple, cmap: str, data_dir: str, save_dir: str, index_list: list = list(range(247)), resolution: int = 224, average: bool = False): 
        self.dimensions = dimensions
        self.cmap = cmap
        self.data_dir = data_dir
        self.save_dir = save_dir
        self.index_list = index_list 
        self.average = average
        self.resolution = resolution

        self.figsize= (8, 8) # this is just the standard

        # Error handling
        if len(index_list) > dimensions[0] * dimensions[1]:
            raise ValueError("Number of indices in index_list should not exceed the number of subplots in the figure. Number of subplots MAY exceed the number of indices in index_list.")
        
        if not os.path.exists(save_dir):
            raise FileNotFoundError(f"Directory {save_dir} does not exist.")
        
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Directory {data_dir} does not exist.")
        

    def plot_many(self, coefficients: np.ndarray):
        """Plots scalograms when more than 1 are requested"""
        
        fig, ax = plt.subplots(
            self.dimensions[0], 
            self.dimensions[1], 
            figsize=(self.figsize))
        
        ax = np.array(ax, ndmin=2) # otherwise parser complains - might ahve to revisit this though
        
        for channel_index, channel in enumerate(coefficients):
            print(channel_index)
            print(channel.shape)
            # check if the channel index is in the index list in case you don't want to plot all of them.
            if channel_index in self.index_list:
                # plots them as a square grid
                r, c = divmod(channel_index, self.dimensions[0])
                ax[r, c].pcolormesh(channel, cmap=self.cmap)
                ax[r, c].set_xticks([])
                ax[r, c].set_yticks([])
            
        # in case there are not enough channels to fill the grid, fill with 0s
        if coefficients.shape[0] < self.dimensions[0]*self.dimensions[1]:
            for channel_index in range(coefficients.shape[0], self.dimensions[0]*self.dimensions[1]):
                print(f"plotting 0s for channel index: {channel_index}")
                r, c = divmod(channel_index, self.dimensions[0])
                ax[r, c].pcolormesh(np.zeros_like(channel), cmap=self.cmap)
                ax[r, c].set_xticks([])
                ax[r, c].set_yticks([])

        #formatting
        for axes in ax.flatten():
            axes.spines['top'].set_visible(False)
            axes.spines['right'].set_visible(False)
            axes.spines['left'].set_visible(False)
            axes.spines['bottom'].set_visible(False)

        # remove the space between the subplots
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0, wspace=0)
        fig.patch.set_visible(False)

        return fig


    def plot_average_scalogram(self, coefficients: np.ndarray):
        """Plots the average scalogram of all the coefficients"""
        fig, axes = plt.subplots(figsize=self.figsize)
        
        average = np.mean(np.abs(coefficients), axis=0)
        axes.pcolormesh(average)
        axes.set_xticks([])
        axes.set_yticks([])
        axes.spines['top'].set_visible(False)
        axes.spines['right'].set_visible(False)
        axes.spines['left'].set_visible(False)
        axes.spines['bottom'].set_visible(False)

        plt.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0, wspace=0)
        fig.patch.set_visible(False)

        return fig
    
    def save_plot(self, filename, fig):
        """Saves the plot in the save directory"""
        dpi = self.resolution / self.figsize[0]
        fig.savefig(os.path.join(self.save_dir, f"{filename}.png"), dpi=dpi)
        plt.close(fig)
        return None