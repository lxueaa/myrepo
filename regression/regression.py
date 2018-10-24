import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim


x = torch.linspace(-1, 1, 100)
x = torch.unsqueeze(x, 1)
y = x.pow(3) + 0.2*torch.rand(x.size())

# plt.scatter(x.numpy(), y.numpy())
# plt.show()

class Net(nn.Module):
	def __init__(self, n_feature, n_h1, n_h2, n_output):
		super(Net, self).__init__()
		self.h1 = nn.Linear(n_feature, n_h1)
		self.h2 = nn.Linear(n_h1, n_h2)
		self.predict = nn.Linear(n_h2, n_output)

	def forward(self, x):
		x = F.sigmoid(self.h1(x))
		x = F.sigmoid(self.h2(x))
		x = self.predict(x)
		return x

net = Net(n_feature = 1, n_h1 = 10, n_h2= 10, n_output = 1)

print(net)

plt.ion()
plt.show()

optimizer = optim.SGD(net.parameters(), lr = 0.5)
criterion = nn.MSELoss()

for bach in range(1000):
	optimizer.zero_grad()
	prediction = net(x)
	loss = criterion(prediction, y)

	print('bach %d, ' % bach, 'loss = %.4f' % loss)

	loss.backward()
	optimizer.step()

	if bach % 100 == 0:
		plt.cla()
		plt.scatter(x.numpy(), y.numpy())
		plt.plot(x.numpy(), prediction.data.numpy(), 'r-', lw = 5)
		plt.text(0.5, 0, 'Loss = %.4f' % loss.data.numpy(), 
			fontdict={'size': 20, 'color': 'red'})
		plt.pause(0.1)






