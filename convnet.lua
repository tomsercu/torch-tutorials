require 'nn'
require 'torch'
--gfx = require 'gfx.js'
require 'image'
require 'gnuplot'

print ('load data')
N = 1000
Nte = 5000
loaded = torch.load('2_supervised/train_32x32.t7', 'ascii')
tr = { X = loaded.X:transpose(3,4)[{{1,N}}], Y = loaded.y[{1,{1,N}}], size = N}
loaded = torch.load('2_supervised/test_32x32.t7', 'ascii')
te = {X = loaded.X:transpose(3,4)[{{1,Nte}}], Y = loaded.y[{1,{1,Nte}}], size = Nte}
tr.X = tr.X:float()
te.X = te.X:float()

print ('Transform to YUV space')
for i = 1,tr.size do
    tr.X[i] = image.rgb2yuv(tr.X[i])
end
for i = 1,te.size do
    te.X[i] = image.rgb2yuv(te.X[i])
end

print ('Pixel-wise normalizations')
mean_imgs = torch.mean(tr.X,1)[1]
std_imgs = torch.std(tr.X,1)[1]
for i = 1,tr.size do
    tr.X[i]:add(- mean_imgs)
    tr.X[i]:cdiv(std_imgs)
end
for i = 1,te.size do
    te.X[i] = te.X[i] - mean_imgs
    te.X[i]:cdiv(std_imgs)
end
-- Todo fix this to better normalization

--print ('Spatial contrastive normalization on all channels')
--scn_kernel = image.gaussian1D(13)
--normalizer = nn.SpatialContrastiveNormalization(1, scn_kernel, 1):float()
--for i = 1, tr.size do
    --for j = 1,3 do
        --tr.X[{i,{j}}] = normalizer:forward(tr.X[{i,{j}}])
    --end
--end
--for i = 1, te.size do 
    --for j = 1,3 do
        --te.X[{i,{j}}] = normalizer:forward(te.X[{i,{j}}])
    --end
--end

print ('Define convnet')
nin = 3*32*32
kernel = image.gaussian1D(7)
model = nn.Sequential()
--model:add(nn.Reshape(nin))
model:add(nn.SpatialConvolutionMM(3,32,5,5))
model:add(nn.Threshold(0,0))
model:add(nn.SpatialLPPooling(32,2,2,2,2,2))
model:add(nn.SpatialSubtractiveNormalization(32,kernel))
-- layer 2
model:add(nn.SpatialConvolutionMM(32,32,5,5))
model:add(nn.Threshold(0,0))
model:add(nn.SpatialLPPooling(32,2,2,2,2,2))
model:add(nn.SpatialSubtractiveNormalization(32,kernel))
-- fully connect
model:add(nn.Reshape(800))
model:add(nn.Linear(800,400))
model:add(nn.Threshold(0,0))
model:add(nn.Linear(400,10))

-- input dimensions
--nfeats = 3
--width = 32
--height = 32
--ninputs = nfeats*width*height

---- number of hidden units (for MLP only):
--nhiddens = ninputs / 2
--noutputs = 10

---- hidden units, filter sizes (for ConvNet only):
--nstates = {64,64,128}
--filtsize = 5
--poolsize = 2
--normkernel = image.gaussian1D(7)
   --model = nn.Sequential()

   ---- stage 1 : filter bank -> squashing -> L2 pooling -> normalization
   --model:add(nn.SpatialConvolutionMM(nfeats, nstates[1], filtsize, filtsize))
   --model:add(nn.Tanh())
   --model:add(nn.SpatialLPPooling(nstates[1],2,poolsize,poolsize,poolsize,poolsize))
   --model:add(nn.SpatialSubtractiveNormalization(nstates[1], normkernel))

   ---- stage 2 : filter bank -> squashing -> L2 pooling -> normalization
   --model:add(nn.SpatialConvolutionMM(nstates[1], nstates[2], filtsize, filtsize))
   --model:add(nn.Tanh())
   --model:add(nn.SpatialLPPooling(nstates[2],2,poolsize,poolsize,poolsize,poolsize))
   --model:add(nn.SpatialSubtractiveNormalization(nstates[2], normkernel))

   ---- stage 3 : standard 2-layer neural network
   --model:add(nn.Reshape(nstates[2]*filtsize*filtsize))
   --model:add(nn.Linear(nstates[2]*filtsize*filtsize, nstates[3]))
   --model:add(nn.Tanh())
   --model:add(nn.Linear(nstates[3], noutputs))

print ('Add loss and energy function')
model:add(nn.LogSoftMax())
crit = nn.ClassNLLCriterion()

print  ('Train')
lr = 0.01
decay = 0.01
losses = {}
err = {}
Terr = {}
for t = 1,30 do
    loss = 0
    alpha =  lr / (1 + decay * t)
    printper = tr.size/100
    for i = 1, tr.size do
        if i % printper == 0 then io.write('.') end
        input = tr.X[{i}]:double()
        mo = model:forward(input)
        loss = loss + crit:forward(mo,tr.Y[i])
        go = crit:backward(mo, tr.Y[i])
        model:zeroGradParameters()
        model:backward(input, go)
        model:updateParameters(alpha)
    --print (mo:mean())
    end
    print('')
    print (mo:mean())
    losses[#losses+1] = loss/tr.size
    -- train error
    corr = 0
    for i = 1, tr.size do
        probs = model:forward(tr.X[i]:double())
        _, ix = torch.max(probs, 1)
        if ix[1] == tr.Y[i] then
            corr = corr + 1
        end
    end
    err[#err+1] = 1 - corr / tr.size
    -- test error
    corr = 0
    for i = 1, te.size do
        probs = model:forward(te.X[i]:double())
        _, ix = torch.max(probs, 1)
        if ix[1] == te.Y[i] then
            corr = corr + 1
        end
    end
    Terr[#Terr+1] = 1 - corr / te.size
    print ('epoch ' .. t .. '   avLoss:  ' .. losses[#losses] .. '    / train error=' .. err[#err] .. ' / test error=' .. Terr[#Terr])
end

z = torch.Tensor(losses)
z:div(tr.size)
--gnuplot.plot({'Average loss', z, '+-'})
