import random  # For generating random numbers.
import dgl  # Deep Graph Library for graph data processing.
import numpy as np  # For numerical operations.
import torch  # PyTorch for tensor computations and neural networks.
from scipy.io import loadmat  # For loading MATLAB files.
from scipy.sparse import csc_matrix  # Compressed Sparse Column matrix for efficient arithmetic operations.
from sklearn.metrics import accuracy_score, roc_auc_score  # Metrics for evaluation.
# Define a tuple containing the label set.
label_set = tuple(range(9))  # Assuming there are 9 classes in total.


def evaluate(y_true, y_pred, y_score, is_target):
    """
    Evaluate the model's predictions against true labels.
    """
    ls, scores, accs = torch.unique(y_true), [], {}  # Get unique labels and initialize lists/dict for storing results.
    for label in ls:
        y_p, y_t = [], []
        for i in range(len(y_true)):
            if y_true[i] == label:
                y_p.append(y_pred[i])
                y_t.append(y_true[i])
        acc = accuracy_score(y_t, y_p)  # Calculate accuracy for each class.
        scores.append(acc)
        if is_target:
            if label < len(ls)-1:
                accs[label_set[label]] = acc  # Store accuracy for known classes only.
        else:
            accs[label_set[label]] = acc  # Store accuracy for all classes.
    if is_target:
        mask = y_true != ls[-1]
        acc_k = accuracy_score(y_true[mask], y_pred[mask])  # Known class accuracy.
        accs.update({
            'OS': np.mean(scores),  # Overall accuracy.
            'OS*': np.mean(scores[:-1]),  # Accuracy excluding the last (unknown) class.
            'UNK_AUC': roc_auc_score(y_true == ls[-1], y_score),  # ROC AUC score for detecting unknown class.
            'HS': 2*acc_k*scores[-1]/(acc_k+scores[-1])  # Harmonic mean of known and unknown accuracies.
        })
    return accs


def filter_unknown_labels(source_features, source_labels, source_adjacency_matrix, unknown_labels):
    """
    Filter out samples with unknown labels from the source dataset.
    """
    unknown_labels = {label_set.index(label)for label in unknown_labels}  # Convert to indices.
    n = source_features.shape[0]  # Number of samples.
    indices = np.array(range(n))
    mask = [label not in unknown_labels for label in source_labels]  # Mask for filtering.
    source_features, source_labels, indices = source_features[mask], source_labels[mask], indices[mask]
    invalid_indices = [i for i in range(n)if i not in indices]  # Indices that are filtered out.
    source_adjacency_matrix = source_adjacency_matrix.toarray()
    source_adjacency_matrix = np.delete(source_adjacency_matrix, invalid_indices, 0)  # Remove rows.
    source_adjacency_matrix = np.delete(source_adjacency_matrix, invalid_indices, 1)  # Remove columns.
    return source_features, source_labels, csc_matrix(source_adjacency_matrix)


def load_data(source, target, unknown_labels):
    """
    Load source and target datasets from .mat files and preprocess them.
    """
    s, t = loadmat(f'data/{source}.mat'), loadmat(f'data/{target}.mat')  # Load source and target datasets.
    source_features, source_labels, source_adjacency_matrix = s['features'], s['labels'][0], s['adjacency_matrix']
    target_features, target_labels, target_adjacency_matrix = t['features'], t['labels'][0], t['adjacency_matrix']
    source_features, source_labels, source_adjacency_matrix = filter_unknown_labels(
        source_features, source_labels, source_adjacency_matrix, unknown_labels)  # Preprocess source data.
    return(source_features, source_labels, source_adjacency_matrix, target_features, target_labels,
           target_adjacency_matrix)


def set_random_seed(gpu_available, seed):
    """
    Set seeds for various libraries to ensure reproducibility.
    """
    random.seed(seed)  # Seed for Python's built-in random module.
    dgl.seed(seed)  # Seed for DGL library.
    np.random.seed(seed)  # Seed for NumPy.
    torch.manual_seed(seed)  # Seed for PyTorch CPU operations.
    if gpu_available:
        torch.cuda.manual_seed_all(seed)  # Seed for PyTorch CUDA operations on all GPUs.
