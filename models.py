from dgl.nn.pytorch import GATConv  # Import the Graph Attention Network layer from DGL library.
from torch import nn  # Import neural network module from PyTorch.
from torch.autograd import Function  # Import autograd function to create custom gradient functions.


class DomainDiscriminator(nn.Module):
    """
    A domain discriminator network used to distinguish between source and target domain features.
    """
    def __init__(self, in_features):
        super(DomainDiscriminator, self).__init__()
        self.linear0 = nn.Linear(in_features, 128)  # First fully connected layer.
        self.linear1 = nn.Linear(128, 128)  # Second fully connected layer.
        self.linear2 = nn.Linear(128, 1)  # Third fully connected layer with one output.
        self.relu = nn.ReLU(True)  # ReLU activation function.
        self.sigmoid = nn.Sigmoid()  # Sigmoid activation function for binary classification.

    def forward(self, x):
        """Forward pass through the domain discriminator."""
        x = self.relu(self.linear0(x))  # Apply first linear transformation followed by ReLU.
        x = self.relu(self.linear1(x))  # Apply second linear transformation followed by ReLU.
        return self.sigmoid(self.linear2(x)).reshape(-1)  # Apply third linear transformation and sigmoid.


class GAT(nn.Module):
    """
    Graph Attention Network model which applies multi-head self-attention on graph data.
    """
    def __init__(self, num_layers, in_feats, num_hidden, heads, feat_drop, attn_drop, activation):
        super(GAT, self).__init__()
        self.gat_layers = nn.ModuleList((GATConv(in_feats, num_hidden, heads[0], feat_drop, attn_drop,
                                                 activation=activation),))
        for _ in range(1, num_layers):
            self.gat_layers.append(GATConv(num_hidden*heads[_-1], num_hidden, heads[_], feat_drop, attn_drop,
                                           activation=activation))

    def forward(self, blocks_or_graph, x):
        """Forward pass through the GAT layers."""
        if self.training:
            for _ in range(len(self.gat_layers)):
                x = self.gat_layers[_](blocks_or_graph[_], x).flatten(1)
        else:
            for _ in range(len(self.gat_layers)):
                x = self.gat_layers[_](blocks_or_graph, x).flatten(1)
        return x


class GradReverse0(Function):
    """
    Gradient reversal layer. This layer is used to reverse the sign of gradients during backpropagation.
    It is particularly useful for domain adaptation tasks where one wants to train a domain-invariant feature extractor.
    """
    lambd = 0

    @staticmethod
    def forward(ctx, *args, **kwargs):
        """Forward pass returns the input as-is."""
        return args[0].view_as(args[0])

    @staticmethod
    def backward(ctx, *grad_outputs):
        """Backward pass reverses the sign of the incoming gradient."""
        return -GradReverse0.lambd*grad_outputs[0]


class GradReverse1S(Function):
    """
    Gradient reversal layer. This layer is used to reverse the sign of gradients during backpropagation.
    It is particularly useful for domain adaptation tasks where one wants to train a domain-invariant feature extractor.
    """
    lambd = 0

    @staticmethod
    def forward(ctx, *args, **kwargs):
        """Forward pass returns the input as-is."""
        return args[0].view_as(args[0])

    @staticmethod
    def backward(ctx, *grad_outputs):
        """Backward pass reverses the sign of the incoming gradient."""
        return -GradReverse1S.lambd*grad_outputs[0]


class GradReverse1T(Function):
    """
    Gradient reversal layer. This layer is used to reverse the sign of gradients during backpropagation.
    It is particularly useful for domain adaptation tasks where one wants to train a domain-invariant feature extractor.
    """
    lambd = 0

    @staticmethod
    def forward(ctx, *args, **kwargs):
        """Forward pass returns the input as-is."""
        return args[0].view_as(args[0])

    @staticmethod
    def backward(ctx, *grad_outputs):
        """Backward pass reverses the sign of the incoming gradient."""
        return -GradReverse1T.lambd*grad_outputs[0]


class Classifier(nn.Module):
    """
    Classification layer that performs a final prediction based on the extracted features.
    """
    def __init__(self, in_feats, np1class, head, feat_drop, attn_drop):
        super(Classifier, self).__init__()
        self.gat_conv = GATConv(in_feats, np1class, head, feat_drop, attn_drop)

    def forward(self, block_or_graph, x):
        """Forward pass through the classifier."""
        return self.gat_conv(block_or_graph, x).mean(1)


class Network(nn.Module):
    """
    Combined network including a generator (feature extractor), a classifier, and a domain discriminator.
    """
    def __init__(self, num_layers, in_feats, num_hidden, heads, feat_drop, attn_drop, activation, np1class):
        super(Network, self).__init__()
        hidden_size = num_hidden*heads[-2]
        self.classifier = Classifier(hidden_size, np1class, heads[-1], feat_drop, attn_drop)
        self.generator = GAT(num_layers, in_feats, num_hidden, heads, feat_drop, attn_drop, activation)
        self.domain_discriminator = DomainDiscriminator(hidden_size)

    def forward(self, blocks_or_graph, x, grl_lambda, is_target, step):
        """Forward pass through the combined network."""
        if self.training:
            if step == 0:
                x = self.generator(blocks_or_graph[:-1], x)
                if is_target:
                    x = grad_reverse0(x, grl_lambda)
                return x, self.classifier(blocks_or_graph[-1], x)
            elif step == 1:
                x = self.generator(blocks_or_graph[:-1], x)
                return x, self.classifier(blocks_or_graph[-1], x), self.domain_discriminator(grad_reverse1(
                    x, grl_lambda, is_target))
            else:
                raise NotImplementedError("Step not implemented.")
        else:
            if step == 0:
                x = self.generator(blocks_or_graph, x)
                if is_target:
                    x = grad_reverse0(x, grl_lambda)
                return x, self.classifier(blocks_or_graph, x)
            elif step == 1:
                x = self.generator(blocks_or_graph, x)
                return x, self.classifier(blocks_or_graph, x), self.domain_discriminator(grad_reverse1(
                    x, grl_lambda, is_target))
            else:
                raise NotImplementedError("Step not implemented.")


def grad_reverse0(x, grl_lambda):
    """Apply gradient reversal with lambda scaling factor."""
    GradReverse0.lambd = grl_lambda
    return GradReverse0.apply(x)


def grad_reverse1(x, grl_lambda, is_target):
    """Apply gradient reversal with lambda scaling factor for either source or target domain."""
    if is_target:
        GradReverse1T.lambd = grl_lambda
        return GradReverse1T.apply(x)
    else:
        GradReverse1S.lambd = grl_lambda
        return GradReverse1S.apply(x)
