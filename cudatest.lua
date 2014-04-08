require 'torch'
require 'nn'
require 'cutorch'
require 'cunn'

a = torch.Tensor(128,1,28,28)
a = a:cuda()
scr = nn.SpatialConvolutionRing(1,32,5,5)
scr = scr:cuda()
scb = nn.SpatialConvolutionBatch(32,64,5,5)
scb = scb:cuda()

b = scr:forward(a)
print(b:size())

c = scb:forward(b)
print(c:size())
