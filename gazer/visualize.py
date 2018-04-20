import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class Visualizer():    
    
    def __init__(self, *args):
        pass    
    
    def show_performance(self, list_of_tuples, fig_size=(9,9), font_scale=1.1, file=''):
        """
        Parameters:     list_of_tuples: 
                            - list containing (clf_name, clf_performance) tuples for each 
                              classifier we wish to visualize
                        fig_size:
                            - set figure size (default: (9,9))
                        font_scale:
                            - text scale in seaborn plots (default: 1.1)
                        file:
                            - string containing a valid filename (default: '')

        Output:         f:  (matplotlib.pyplot.figure object)
        """
        if not (isinstance(list_of_tuples, list) and isinstance(list_of_tuples[0], tuple)):
            raise ValueError("Expecting a list of tuples")
        sns.set(font_scale=font_scale)
        sns.set_style("whitegrid") 
        data = list()
        for name, value in list_of_tuples: data.append([name, value])
        data = pd.DataFrame(data, columns=['classifier', 'performance'])
        data.sort_values('performance', inplace=True, ascending=False)
        """ 
        Close all figures (can close individual figure using plt.close(f) 
        where f is a matplotlib.pyplot.figure object) 
        """
        plt.close('all')
        
        f = plt.figure(figsize=fig_size)
        sns.barplot(x='performance', y='classifier', data=data)
        plt.xlabel('performance')
        if len(file)>1: 
            try: 
                plt.savefig(file)
            except: 
                pass
        return f


if __name__ == '__main__':
    sys.exit(-1)