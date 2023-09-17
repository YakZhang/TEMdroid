import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score,accuracy_score


def compute_metrics(pred):
    predictions, label_ids = pred
    if len(predictions) == 2: 
        predictions = predictions[0]
    predictions = np.argmax(predictions, axis=1)
    #pre,rec,f1,_ = precision_recall_fscore_support(label_ids,predictions)
    pre = precision_score(label_ids,predictions,average='binary')
    rec = recall_score(label_ids,predictions,average='binary')
    f1 = f1_score(label_ids,predictions,average='binary')
    #a = accuracy_score(label_ids,predictions)
    accuracy = (predictions == label_ids).astype(np.float32).mean().item()
    metric_dict = {
        "accuracy": accuracy,
        "precision":pre,
        "recall":rec,
        "f1":f1,

    }
    # return {"accuracy": (predictions == label_ids).astype(np.float32).mean().item()}
    return metric_dict

def compute_metrics_cosine(pred):
    threshold = 0.6 
    predictions, label_ids = pred
    predicted = (predictions > threshold).astype(np.float32) 
    pre = precision_score(label_ids,predicted,average='binary')
    rec = recall_score(label_ids,predicted,average='binary')
    f1 = f1_score(label_ids,predicted,average='binary')
    #a = accuracy_score(label_ids,predictions)
    accuracy = (predicted == label_ids).astype(np.float32).mean().item()
    metric_dict = {
        "accuracy": accuracy,
        "precision":pre,
        "recall":rec,
        "f1":f1,

    }
    # return {"accuracy": (predictions == label_ids).astype(np.float32).mean().item()}
    return metric_dict

def compute_metrics_class(pred):
    # for multi-class
    predictions, label_ids = pred
    predictions = np.argmax(predictions, axis=1)
    #pre,rec,f1,_ = precision_recall_fscore_support(label_ids,predictions)
    pre = precision_score(label_ids,predictions,average='weighted')
    rec = recall_score(label_ids,predictions,average='weighted')
    f1 = f1_score(label_ids,predictions,average='weighted')
    #a = accuracy_score(label_ids,predictions)
    accuracy = (predictions == label_ids).astype(np.float32).mean().item()
    metric_dict = {
        "accuracy": accuracy,
        "precision":pre,
        "recall":rec,
        "f1":f1,

    }
    # return {"accuracy": (predictions == label_ids).astype(np.float32).mean().item()}
    return metric_dict