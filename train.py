from itertools import permutations  # Import the permutations function for generating pairs of datasets.
from models import Network  # Import the Network class from a custom module 'models'.
from sklearn.cluster import KMeans  # Import KMeans clustering algorithm.
from torch import nn  # Import neural network module from PyTorch.
from torch.optim import Adam  # Import Adam optimizer from PyTorch.
from torch.utils.data.dataloader import DataLoader  # Import DataLoader for batch training.
import dgl  # Import Deep Graph Library for graph data processing.
import numpy as np  # Import NumPy for numerical operations.
import torch  # Import PyTorch for tensor computations and neural networks.
import utils  # Import utility functions from a custom module.


def batch(g):
    """
    Generate batches of node indices for mini-batch training on graph g.
    """
    num_nodes = g.num_nodes()
    node_indices = list(range(num_nodes))
    np.random.shuffle(node_indices)  # Shuffle node indices to create randomness in batches.
    node_indices = torch.tensor(((max_num_nodes//num_nodes+bool(max_num_nodes % num_nodes))*node_indices)[
                                :max_num_nodes], dtype=torch.long, device=g.device)
    collator = dgl.dataloading.NodeCollator(g, node_indices, dgl.dataloading.MultiLayerFullNeighborSampler(
        all_gat_layers))  # Create a NodeCollator for batching.
    for inputs, outputs, blocks in DataLoader(collator.dataset, batch_size, collate_fn=collator.collate):
        yield inputs, outputs, blocks


def _test0(g, x, y, is_target):
    """
    Evaluate the model on the given graph data during phase 0 (without domain adaptation).
    """
    network.eval()  # Set the network to evaluation mode.
    with torch.no_grad():  # Disable gradient calculation for inference.
        emb, logit = network(g, x, 1, is_target, 0)
    # noinspection DuplicatedCode
    prob_unk = logit.softmax(1)[:, -1]  # Calculate the probability of the unknown class.
    if not is_target:
        logit = logit[:, :-1]  # Exclude the unknown class for source domain.
    y_p = logit.argmax(1)  # Predicted classes.
    return emb, y_p, prob_unk, utils.evaluate(y.cpu(), y_p.cpu(), prob_unk.cpu(), is_target)


def _test1(g, x, y, is_target):
    """
    Evaluate the model on the given graph data during phase 1 (with domain adaptation).
    """
    network.eval()
    with torch.no_grad():
        if is_target:
            emb, logit, _ = network(g, x, grl_lambda_t, is_target, 1)
        else:
            emb, logit, _ = network(g, x, grl_lambda_s, is_target, 1)
    # noinspection DuplicatedCode
    prob_unk = logit.softmax(1)[:, -1]
    if not is_target:
        logit = logit[:, :-1]
    y_p = logit.argmax(1)
    return emb, y_p, prob_unk, utils.evaluate(y.cpu(), y_p.cpu(), prob_unk.cpu(), is_target)


def train0():
    """
    Train the model during phase 0 (without domain adaptation).
    """
    network.train()  # Set the network to training mode.
    for (i_s, o_s, b_s), (i_t, o_t, b_t) in zip(batch(G_s), batch(G_t)):
        ths = torch.tensor(o_s.shape[0]*[th], dtype=torch.float, device=G_s.device)
        _, logit_s = network(b_s, X_s[i_s], 1, False, 0)
        _, logit_t = network(b_t, X_t[i_t], 1, True, 0)
        loss_s = cross_entropy_loss_func(logit_s, y_s[o_s])  # Source domain classification loss.
        loss_adv = bce_loss_func(logit_t.softmax(1)[:, -1], ths)  # Adversarial loss on target domain.
        loss = loss_s+loss_adv  # Total loss.
        optimizer.zero_grad()  # Clear previous gradients.
        loss.backward()  # Perform backpropagation.
        optimizer.step()  # Update parameters.
    return _test0(G_s, X_s, y_s, False)[0]


def train1():
    """
    Train the model during phase 1 (with domain adaptation).
    """
    network.train()
    for (i_s, o_s, b_s), (i_t, o_t, b_t) in zip(batch(G_s), batch(G_t)):
        lis, lit = b_s[-1].srcdata['_ID'], b_t[-1].srcdata['_ID']
        b_grl_t = grl_lambda_t[lit].reshape(-1, 1)
        _, logit_s, d_prob_s = network(b_s, X_s[i_s], grl_lambda_s, False, 1)
        _, logit_t, d_prob_t = network(b_t, X_t[i_t], b_grl_t, True, 1)
        b_mask = pseudo_label[o_t] != -1
        if b_mask.any():
            loss_t = cross_entropy_loss_func(logit_t[b_mask], pseudo_label[o_t][b_mask])
        else:
            loss_t = 0
        loss_s = cross_entropy_loss_func(logit_s, y_s[o_s])
        loss_d = bce_loss_func(torch.cat((d_prob_s, d_prob_t)), torch.cat((d_s[lis], d_t[lit])))
        loss = loss_s+w_t*loss_t+loss_d
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return _test1(G_s, X_s, y_s, False)[0]


# Constants and hyperparameters setup
N_UNK = 4  # Number of unknown classes.
N_CLASS = 10-N_UNK  # Number of classes.
UNK_LABEL = N_CLASS-1  # Label index for the unknown class.
gpu_available = torch.cuda.is_available()  # Check if GPU is available.
gpu = torch.device('cuda')if gpu_available else None  # Define the device to use.
batch_size = 2048  # Batch size for training.
epochs0 = 30  # Epochs for phase 0 training.
epochs1 = 200  # Epochs for phase 1 training.
lr = 1e-3  # Learning rate.
num_heads = 8  # Number of attention heads.
num_hidden = 32  # Number of hidden units in each layer.
num_layers = 1  # Number of layers in the GAT.
num_out_heads = 2  # Number of output attention heads.
th = 0.5  # Threshold for adversarial loss.
grl_lambda_s = 1  # Gradient reversal lambda for source domain.
top_k = 2000  # Top K samples for pseudo-labeling.
w_t = 0.1  # Weight for target domain classification loss.
all_gat_layers, heads = num_layers+1, num_layers*[num_heads]+[num_out_heads]  # Layers configuration for GAT.
accs_t, feat_s, feat_t = {}, None, None  # Initialize accuracy dict and feature tensors.
bce_loss_func, cross_entropy_loss_func = nn.BCELoss(), nn.CrossEntropyLoss()  # Loss functions.
if gpu_available:
    bce_loss_func, cross_entropy_loss_func = bce_loss_func.cuda(), cross_entropy_loss_func.cuda()
file = open('UAGA.txt', 'w', encoding='UTF-8')
file.close()
for source, target in permutations(('citation-v1', 'dblp-v4', 'acm-v8'), 2):
    # Adjust hyperparameters based on the dataset pair.
    if(source, target)not in {('dblp-v4', 'acm-v8'), ('acm-v8', 'dblp-v4')}:
        weight_decay = 1e-4
    else:
        weight_decay = 1e-5
    if(source, target)not in {('citation-v1', 'dblp-v4'), ('acm-v8', 'citation-v1')}:
        feat_drop = attn_drop = 0.5
    elif(source, target) == ('citation-v1', 'dblp-v4'):
        feat_drop = attn_drop = 0.4
    else:
        feat_drop = attn_drop = 0.6
    os_ts, os_star_ts, unk_auc_ts, hs_ts = [], [], [], []  # Lists to store evaluation results.
    for seed in range(5):
        activation = nn.ReLU(True)  # Activation function.
        X_s, y_s, A_s, X_t, y_t, A_t = utils.load_data(source, target, utils.label_set[-N_UNK:])  # Load datasets.
        X_s, X_t = torch.FloatTensor(X_s.toarray()), torch.FloatTensor(X_t.toarray())  # Convert to float tensors.
        y_s, y_t = torch.LongTensor(y_s), torch.LongTensor(y_t)  # Convert labels to long tensors.
        d_s, d_t = torch.FloatTensor(X_s.shape[0]*[0]), torch.FloatTensor(X_t.shape[0]*[1])  # Domain labels.
        grl_lambda_t = torch.zeros(X_t.shape[0])  # Gradient reversal lambda for target domain.
        cluster_label = torch.LongTensor(X_t.shape[0]*[-1])  # Cluster labels for target domain.
        pred_label = torch.LongTensor(X_t.shape[0]*[-1])  # Predicted labels for target domain.
        pseudo_label = torch.LongTensor(X_t.shape[0]*[-1])  # Pseudo-labels for target domain.
        unk_score = torch.FloatTensor(X_t.shape[0]*[-1])  # Unknown score for target domain.
        utils.set_random_seed(gpu_available, seed)  # Set random seed for reproducibility.
        G_s = dgl.from_scipy(A_s)  # Create DGL graph for source domain.
        G_t = dgl.from_scipy(A_t)  # Create DGL graph for target domain.
        network = Network(num_layers, X_s.shape[1], num_hidden, heads, feat_drop, attn_drop, activation, N_CLASS)
        if gpu_available:
            G_s, G_t = G_s.to(gpu), G_t.to(gpu)
            d_s, d_t, grl_lambda_t, network = d_s.cuda(), d_t.cuda(), grl_lambda_t.cuda(), network.cuda()
            X_s, X_t, y_s, y_t, pred_label = X_s.cuda(), X_t.cuda(), y_s.cuda(), y_t.cuda(), pred_label.cuda()
            cluster_label, pseudo_label, unk_score = cluster_label.cuda(), pseudo_label.cuda(), unk_score.cuda()
        batch_size = min(batch_size, X_s.shape[0], X_t.shape[0])  # Adjust batch size.
        G_s, G_t = G_s.remove_self_loop().add_self_loop(), G_t.remove_self_loop().add_self_loop()  # Add self-loops.
        max_num_nodes = max(X_s.shape[0], X_t.shape[0])  # Maximum number of nodes between source and target graphs.
        y_t[y_t > UNK_LABEL] = UNK_LABEL  # Map unknown classes to the unknown label.
        optimizer = Adam(({'params': network.generator.parameters()}, {'params': network.classifier.parameters()}),
                         lr, weight_decay=weight_decay)  # Optimizer for phase 0.
        for epoch in range(1, epochs0+1):
            print(f'{source}->{target}: seed = {seed}, epoch0 = {epoch}')
            feat_s = train0()
            feat_t, pred_label, unk_score, accs_t = _test0(G_t, X_t, y_t, True)
            print('OS: {:.4f}, OS*: {:.4f}, UNK_AUC: {:.4f}, HS: {:.4f}'.format(accs_t['OS'], accs_t['OS*'], accs_t[
                'UNK_AUC'], accs_t['HS']))
        optimizer = Adam(network.parameters(), lr, weight_decay=weight_decay)  # Optimizer for phase 1.
        for epoch in range(1, epochs1+1):
            print(f'{source}->{target}: seed = {seed}, epoch1 = {epoch}')
            sorted_indices = unk_score.argsort()
            cluster_label = torch.tensor(KMeans(N_CLASS, init=np.array(torch.cat([feat_s[y_s == _].mean(
                0, True)for _ in range(UNK_LABEL)]+[feat_t[sorted_indices[-top_k:]].mean(0, True)]).tolist()),
                                                n_init=1).fit_predict(np.array(feat_t.tolist())), dtype=torch.long,
                                         device=X_s.device)
            pseudo_label[:] = -1
            mask = cluster_label == pred_label
            pseudo_label[mask] = cluster_label[mask]
            for _ in range(X_t.shape[0]):
                if _ in sorted_indices[-top_k:] and pseudo_label[_] == -1:
                    pseudo_label[_] = UNK_LABEL
            for _ in range(-1, N_CLASS):
                if _ < UNK_LABEL:
                    grl_lambda_t[pseudo_label == _] = 1
                else:
                    grl_lambda_t[pseudo_label == _] = -1
            feat_s = train1()
            feat_t, pred_label, unk_score, accs_t = _test1(G_t, X_t, y_t, True)
            print('OS: {:.4f}, OS*: {:.4f}, UNK_AUC: {:.4f}, HS: {:.4f}'.format(accs_t['OS'], accs_t['OS*'], accs_t[
                'UNK_AUC'], accs_t['HS']))
        os_ts.append(accs_t['OS'])
        os_star_ts.append(accs_t['OS*'])
        unk_auc_ts.append(accs_t['UNK_AUC'])
        hs_ts.append(accs_t['HS'])
    file = open('UAGA.txt', 'a', encoding='UTF-8')
    print(f'{source}->{target}', file=file)
    print('mean: (OS: {:.4f}±{:.4f}, OS*: {:.4f}±{:.4f}, UNK_AUC: {:.4f}±{:.4f}, HS: {:.4f}±{:.4f})'.format(np.mean(
        os_ts), np.std(os_ts), np.mean(os_star_ts), np.std(os_star_ts), np.mean(unk_auc_ts), np.std(unk_auc_ts),
        np.mean(hs_ts), np.std(hs_ts)), file=file)
    file.close()
