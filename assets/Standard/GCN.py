import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.utils.data import DataLoader

class GCN(torch.nn.Module):
    def __init__(self, args, data):
        super(GCN, self).__init__()
        self.args = args
        self.num_layers = args.num_layers
        self.hidden_dim = args.dim_hidden
        self.input_dim = data.num_node_features
        self.output_dim = args.num_classes
        self.convs = torch.nn.ModuleList()
        self.data = data

        # First layer
        self.convs.append(GCNConv(self.input_dim, self.hidden_dim))

        # Intermediate layers
        for _ in range(self.num_layers - 2):  # For cases where num_layers > 2
            self.convs.append(GCNConv(self.hidden_dim, self.hidden_dim))

        # Last layer
        self.convs.append(GCNConv(self.hidden_dim, self.output_dim))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # Apply convolutional layers
        for conv in self.convs[:-1]:  # All but the last layer
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)

        # Last layer without activation
        x = self.convs[-1](x, edge_index)
        return F.log_softmax(x, dim=1)

    def train_net(self, input_dict):
        device = input_dict["device"]
        y = input_dict["y"].to(device)
        optimizer = input_dict["optimizer"]
        loss_op = input_dict["loss_op"]
        model = self.to(device)
        train_mask = input_dict['split_masks']['train']
        self.data = input_dict["data"].to(device)

        # Set model to training mode
        model.train()

        optimizer.zero_grad()
        out = model(self.data)
        loss = loss_op(out[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()

        # Calculate the number of correct predictions
        pred = out[train_mask].argmax(dim=1)
        correct = pred.eq(y[train_mask]).sum().item()

        # Calculate accuracy for the current epoch
        train_acc = correct / train_mask.sum().item()

        return loss.item(), train_acc

    def inference(self, input_dict):
        x_all = input_dict["x"]
        device = input_dict["device"]
        if self.args.batch_size == 0:
            # Perform full-batch inference without batching
            y_pred = self.forward(self.data.to(device))
            y_preds = y_pred.cpu()
        else:
            # Perform batched inference
            y_preds = []
            loader = DataLoader(range(x_all.size(0)), batch_size=self.args.batch_size)
            for perm in loader:
                y_pred = self.forward(self.data.to(device))
                y_preds.append(y_pred.cpu())
            y_preds = torch.cat(y_preds, dim=0)

        return y_preds