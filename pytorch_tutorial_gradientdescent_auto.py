#pytorch training pipeline
# 1. Design Model (input size, output size, forward pass)
# 2. Construct Loss and Optimizer
# 3. Training Loop
#   - forward pass (compute prediction)
#   - backward pass (gradients)
#   - update weights
from pickletools import optimize
import torch
import torch.nn as nn

#f = x * w
#Step 1
x = torch.tensor([[1],[2],[3],[4]], dtype=torch.float32)
y = torch.tensor([[5],[10],[15],[20]], dtype=torch.float32)

n_samples, n_features = x.shape

print(n_samples, n_features)

input_size = n_features
output_size = n_features
#model = nn.Linear(input_size, output_size)

#custom model
class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        #define layers
        self.lin = nn.Linear(input_dim, output_dim)
    def forward(self,x):
        return self.lin(x)    

model = LinearRegression(input_size, output_size)

test_sample = torch.tensor([5], dtype=torch.float32)
print(f'Prediction before training: f(5) = {model(test_sample).item():.3f}')


learning_rate = 0.01

#loss - Step 2
loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

#Training - Step 3
n_iters = 10000
for epoch in range(n_iters):
    #prediction = forward pass
    y_pred = model(x)
    #loss 
    l = loss(y,y_pred)
    #gradients
    l.backward() #calculate gradient of loss with respect to w

    #update weight 
    optimizer.step()

    #zero gradients (whenever backward() is called)
    optimizer.zero_grad()

    if epoch % 1000 == 0:
        [w, b] = model.parameters()
        print(f'epoch {epoch + 1}: w = {w[0][0].item():.3f} loss = {l:.8f}')

print(f'Prediction after training: f(5) = {model(test_sample).item():.3f}')
