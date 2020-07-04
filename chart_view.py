'''
Author of this file: Tillmann
'''

# imports
import matplotlib.pyplot as plt

def chart_viewer(title, xlabel, ylabel, x_data, y_data): 
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
    # Save Figure
    plt.savefig(title + ".png")
    # Show Figure
    plt.show()
