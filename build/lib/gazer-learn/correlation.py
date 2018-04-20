import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


class CheckClassifierCorrelation():
    """
    Check correlation between classifier predictions using e.g. 
    Pearson's formula. Initially, a repository
    of classifiers is constructed from either a random, complete, 
    or chosen selection (set by the 'method' parameter).     
    """
    def __init__(self, pred_type=None):
        options = ('binaryclass', 'multiclass', 'regression')
        if pred_type not in options:
            raise ValueError("Valid options for prediction_type are: %s" % ", ".join(options))                    
        self.prediction_type = pred_type
        
    def compute_correlation_matrix(self, preds):       
        corr = np.zeros((len(preds), len(preds)), dtype=np.float32)
        names = []        
        # Note that this is a bit "dodgy" for binary variables
        if self.prediction_type == 'binaryclass' or prediction_type == 'multiclass': 
            for i, (nm1, y1) in enumerate(preds):
                names.append(nm1)
                for j, (nm2, y2) in enumerate(preds):
                    corr[i][j] = mcc(y1, y2)
                    
        elif self.prediction_type == 'regression':
            raise NotImplementedError("This method has not been implemented for regression")           
        
        return names, corr
        
    @staticmethod    
    def plot_correlation_matrix(names, corr, rot=0, fig_size=(9,9), font_scale=1.0, file=None):       
        fig = plt.figure(figsize=fig_size)        
        sns.set(font_scale=font_scale)
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="YlGnBu", 
                    xticklabels=names, yticklabels=names)        
        plt.xticks(rotation=90-rot)
        plt.tight_layout()        
        if file is not None and isinstance(file, str):
            try: 
                plt.savefig(file)
            except: 
                raise RunTimeWarning("Could not save figure to %s" % file)        
        return fig