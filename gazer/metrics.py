from sklearn.metrics import (
    f1_score, 
    precision_score, 
    recall_score, 
    roc_auc_score, 
    accuracy_score, 
    log_loss )

metrics = {
    'f1': f1_score, 
    'precision': precision_score,
    'recall': recall_score, 
    'auc': roc_auc_score,
    'accuracy': accuracy_score, 
    'log_loss': log_loss }

def get_scorer(scorer):
    """
    Get scikit-learn scorer.
    
    Parameters:
    ------------
        scorer : str
    
    Returns:
    ---------
        scorer : object of type sklearn.metrics    
    """
    available = ('f1', 'precision', 'recall', 'auc', 'accuracy', 'log_loss')
    if not scorer in available:
        raise ValueError("Invalid scorer type. Valid: %s" % ",".join(available))       
    return metrics[scorer]