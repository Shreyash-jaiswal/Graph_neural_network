from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

dataset = Planetoid(root='data/Planetoid', name='Cora', transform=NormalizeFeatures())

import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv #GATConv

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        torch.manual_seed(42)

        # Initialize the layers
        self.conv1 = GCNConv(dataset.num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.out = Linear(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index):
        # First Message Passing Layer (Transformation)
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)

        # Second Message Passing Layer
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)

        # Output layer 
        x = F.softmax(self.out(x), dim=1)
        return x

# Initialize model
model = GCN(hidden_channels=16)
data = dataset[0]  

# Use GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
data = data.to(device)

# Initialize Optimizer
learning_rate = 0.01
decay = 5e-4
optimizer = torch.optim.Adam(model.parameters(), 
                             lr=learning_rate, 
                             weight_decay=decay)
# Define loss function (CrossEntropyLoss for Classification Problems with 
# probability distributions)
criterion = torch.nn.CrossEntropyLoss()

def train():
      model.train()
      optimizer.zero_grad() 
      # Use all data as input, because all nodes have node features
      out = model(data.x, data.edge_index)  
      # Only use nodes with labels available for loss calculation --> mask
      loss = criterion(out[data.train_mask], data.y[data.train_mask])  
      loss.backward() 
      optimizer.step()
      return loss

def test():
      model.eval()
      out = model(data.x, data.edge_index)
      # Use the class with highest probability.
      pred = out.argmax(dim=1)  
      # Check against ground-truth labels.
      test_correct = pred[data.test_mask] == data.y[data.test_mask]  
      # Derive ratio of correct predictions.
      test_acc = int(test_correct.sum()) / int(data.test_mask.sum())  
      return test_acc

losses = []
for epoch in range(0, 1001):
    loss = train()
    losses.append(loss)
    if epoch % 100 == 0:
      print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

# Visualize the training loss

# import seaborn as sns
# losses_float = [float(loss.cpu().detach().numpy()) for loss in losses] 
# loss_indices = [i for i,l in enumerate(losses_float)] 
# plt = sns.lineplot(loss_indices, losses_float)
# print(plt)

# #test accuracy
# test_acc = test()
# print(f'Test Accuracy: {test_acc:.4f}')

# #visualizing the embedding

# import seaborn as sns
# import numpy as np
# sample = 9
# sns.set_theme(style="whitegrid")
# print(model(data.x, data.edge_index).shape)
# pred = model(data.x, data.edge_index)
# sns.barplot(x=np.array(range(7)), y=pred[sample].detach().cpu().numpy())

# import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE
# import numpy as np

# def plt2arr(fig):
#     rgb_str = fig.canvas.tostring_rgb()
#     (w,h) = fig.canvas.get_width_height()
#     rgba_arr = np.fromstring(rgb_str, dtype=np.uint8, sep='').reshape((w,h,-1))
#     return rgba_arr


# def visualize(h, color, epoch):
#     fig = plt.figure(figsize=(5,5), frameon=False)
#     fig.suptitle(f'Epoch = {epoch}')
#     # Fit TSNE with 2 components
#     z = TSNE(n_components=2).fit_transform(out.detach().cpu().numpy())

#     # Create scatterplot from embeddings
#     plt.xticks([])
#     plt.yticks([])
#     plt.scatter(z[:, 0], 
#                 z[:, 1], 
#                 s=70, 
#                 c=color.detach().cpu().numpy(), 
#                 cmap="Set2")
#     fig.canvas.draw()

#     # Convert to numpy
#     return plt2arr(fig)


# # Reset the previously trained model weights
# for layer in model.children():
#    if hasattr(layer, 'reset_parameters'):
#        layer.reset_parameters()

# # Ignore deprecation warnings here
# import warnings
# warnings.filterwarnings('ignore')

# # Train the model and save visualizations
# images = []
# for epoch in range(0, 2000):
#     loss = train()
#     if epoch % 50 == 0:
#       out = model(data.x, data.edge_index)
#       images.append(visualize(out, color=data.y, epoch=epoch))
# print("TSNE Visualization finished.")

# # Building a GIF from this
# from moviepy.editor import ImageSequenceClip
# fps = 1
# filename = "/content/embeddings.gif"
# clip = ImageSequenceClip(images, fps=fps)
# clip.write_gif(filename, fps=fps)

# from IPython.display import Image
# with open('/content/embeddings.gif','rb') as f:
#     display(Image(data=f.read(), format='png'))
