'''
Author of this file: Tillmann
'''

# imports
import matplotlib.pyplot as plt

def chart_viewer(title, xlabel, ylabel, x_data, y_data): 
    '''
    View any data as a chart (e.g. one loss function)
    '''
    # Add Data
    plt.plot(x_data, y_data, color='orange')
    # Add Title
    plt.title(title)
    # Add Labels
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # Add Limits
    plt.show()

def multiple_chart_viewer(title, xlabel, ylabel, x_data1, y_data1, x_data2, y_data2, x_data3, y_data3, x_data4, y_data4): 
    '''
    View any data as a chart (e.g. multiple loss functions)
    '''
    # Add Data
    plt.plot(x_data1, y_data1)
    plt.plot(x_data2, y_data2)
    plt.plot(x_data3, y_data3)
    plt.plot(x_data4, y_data4)
    # Add Title
    plt.title(title)
    # Add Labels
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
