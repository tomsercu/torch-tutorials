require 'nn'
require 'torch'
--gfx = require 'gfx.js'
require 'image'
require 'gnuplot'

print ('load data')
N = 10000
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

print ('Spatial contrastive normalization on all channels')
scn_kernel = image.gaussian1D(13)
normalizer = nn.SpatialContrastiveNormalization(1, scn_kernel, 1):float()
for i = 1, tr.size do
    for j = 1,3 do
        tr.X[{i,{j}}] = normalizer:forward(tr.X[{i,{j}}])
    end
end
for i = 1, te.size do 
    for j = 1,3 do
        te.X[{i,{j}}] = normalizer:forward(te.X[{i,{j}}])
    end
end

--print ('Define convnet')
print ('Define Logistic Regression')
nin = 3*32*32
model = nn.Sequential()
model:add(nn.Reshape(nin))
model:add(nn.Linear(nin, nin/2  ))
model:add(nn.Threshold(0,0))
model:add(nn.Linear(nin/2,10 ))


print ('Add loss and energy function')
model:add(nn.LogSoftMax())
crit = nn.ClassNLLCriterion()

print  ('Train')
lr = 0.01
decay = 0.2
losses = {}
err = {}
Terr = {}
for t = 1,60 do
    loss = 0
    alpha =  lr / (1 + decay * t)
    for i = 1, tr.size do
        input = tr.X[{i}]:double()
        mo = model:forward(input)
        loss = loss + crit:forward(mo,tr.Y[i])
        go = crit:backward(mo, tr.Y[i])
        model:zeroGradParameters()
        model:backward(input, go)
        model:updateParameters(alpha)
    end
    losses[#losses+1] = loss/tr.size
    -- train error
    probs = model:forward(tr.X:double())
    _, ix = torch.max(probs, 2)
    corr = torch.eq(tr.Y, ix:double()):sum()
    err[#err+1] = 1 - corr / tr.size
    -- test error
    probs = model:forward(te.X:double())
    _, ix = torch.max(probs, 2)
    corr = torch.eq(te.Y, ix:double()):sum()
    Terr[#Terr+1] = 1 - corr / te.size
    print ('epoch ' .. t .. '   avLoss:  ' .. losses[#losses] .. '    / train error=' .. err[#err] .. ' / test error=' .. Terr[#Terr])
end

z = torch.Tensor(losses)
z:div(tr.size)
gnuplot.plot({'Average loss', z, '+-'})
