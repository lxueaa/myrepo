import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class MNISTConvNet(nn.Module):

	def __init__(self):
		super(MNISTConvNet, self).__init__()
		self.conv1 = nn.Conv2d(1, 10, 5)
		self.pool1 = nn.MaxPool2d(2, 2)
		self.conv2 = nn.Conv2d(10, 20, 5)
		self.pool2 = nn.MaxPool2d(2, 2)
		self.fc1 = nn.Linear(320, 50)	
		self.fc2 = nn.Linear(50, 10)

	def forward(self, input):
		x = self.pool1(F.relu(self.conv1(input)))
		x = self.pool1(F.relu(self.conv2(x)))
		
		x = x.view(x.size(0), -1)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		return x

def printnorm(self, input, output):
	print('Inside '+ self.__class__.__name__ + ' forward')
	print('')
	print('input: ', type(input))
	print('input[0]', type(input[0]))
	print('output: ', type(output))	
	print('')
	print('input size: ', input[0].size())
	print('output size: ', output.data.size())
	print('output norm: ', output.data.norm())

def printgradnorm(self, grad_input, grad_output):
	print('Inside ' + self.__class__.__name__ + ' backward')
	print('Inside class:' + self.__class__.__name__)
	print('')
	print('grad_input: ', type(grad_input))
	print('grad_input[0]: ', type(grad_input[0]))
	print('grad_output: ', type(grad_output))
	print('grad_output[0]: ', type(grad_output[0]))
	print('')
	print('grad_input size: ', grad_input[0].size())
	print('grad_output size: ', grad_output[0].size())
	print('grad_intput norm: ', grad_input[0].data.norm())
	#print('gradPinput.size: ', grad_input.size())

net = MNISTConvNet()
print(net)

input = Variable(torch.randn(1, 1, 28, 28))
out = net(input)
print(out.size())
print("out", out)

target = Variable(torch.LongTensor([3]))
loss_fn = nn.CrossEntropyLoss()
err = loss_fn(out, target)
err.backward()

print("Target: ", target)
print("err: ", err)


print(net.conv1.weight.grad.size())
print(net.conv1.weight.data.norm())

print(net.conv1.weight.grad.data.norm())

net.conv2.register_forward_hook(printnorm)
out = net(input)

net.conv2.register_backward_hook(printgradnorm)

out = net(input)
err = loss_fn(out, target)
err.backward()


