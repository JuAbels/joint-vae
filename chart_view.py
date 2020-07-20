'''
Author of this file: Tillmann
'''

# imports
import matplotlib.pyplot as plt


def chart_viewer(path, title, xlabel, ylabel, x_data, y_data, y_lim=(0, 0.01)):
    '''
    View any data as a chart (e.g. one loss function)
    The chart will be saved as png with the given title
    '''
    # Add Data
    plt.plot(x_data, y_data, color='orange')
    # Add Title
    plt.title(title)
    # Add Labels
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # configure axis limits
    plt.ylim(y_lim)
    # Save Figure
    plt.savefig(path + title + ".png")
    # Show Figure
    # plt.show()
