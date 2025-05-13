from sklearn.metrics import precision_score, recall_score, f1_score, cohen_kappa_score

def compute_metrics(y_true, y_pred):
    return {
        'precision': precision_score(y_true, y_pred, average='macro'),
        'recall':    recall_score(y_true, y_pred, average='macro'),
        'f1':        f1_score(y_true, y_pred, average='macro'),
        'kappa':     cohen_kappa_score(y_true, y_pred)
    }
