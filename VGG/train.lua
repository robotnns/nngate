require 'xlua'
require 'optim'
require 'nn'
dofile './provider.lua'
local c = require 'trepl.colorize'

opt = lapp[[
   -s,--save                  (default "logs")      subdirectory to save logs
   -b,--batchSize             (default 128)          batch size
   -r,--learningRate          (default 1)        learning rate
   --learningRateDecay        (default 1e-7)      learning rate decay
   --weightDecay              (default 0.0005)      weightDecay
   -m,--momentum              (default 0.9)         momentum
   --epoch_step               (default 25)          epoch step
   --model                    (default vgg_bn_drop)     model name
   --max_epoch                (default 300)           maximum number of iterations
   --backend                  (default nn)            backend
   --type                     (default cuda)          cuda/float/cl
]]
-- LMX:split big dataset into smaller batches, with batchSize images in each batch
print(opt)

do -- data augmentation module
  local BatchFlip,parent = torch.class('nn.BatchFlip', 'nn.Module') -- LMX:create class named nn.BatchFlip, inherited from nn.Module

  function BatchFlip:__init()
    parent.__init(self)
    self.train = true
  end

  function BatchFlip:updateOutput(input)
    if self.train then
      local bs = input:size(1) --LMX: bs means batch size
      local flip_mask = torch.randperm(bs):le(bs/2) -- LMX:randperm: random permutation number 1 to bs, le means <=, flip_mask randomly set half of the images to flip
      for i=1,input:size(1) do
        if flip_mask[i] == 1 then image.hflip(input[i], input[i]) end -- LMX:flip image horizontally
      end
    end
    self.output:set(input)
    return self.output
  end
end

local function cast(t)
   if opt.type == 'cuda' then
      require 'cunn'
      return t:cuda()
   elseif opt.type == 'float' then
      return t:float()
   elseif opt.type == 'cl' then
      require 'clnn'
      return t:cl()
   else
      error('Unknown type '..opt.type)
   end
end

print(c.blue '==>' ..' configuring model')
local model = nn.Sequential()
model:add(nn.BatchFlip():float())
model:add(cast(nn.Copy('torch.FloatTensor', torch.type(cast(torch.Tensor()))))) -- LMX:nn.Copy is network layer. This layer copies the input to output with type casting from inputType to outputType
model:add(cast(dofile('models/'..opt.model..'.lua')))
model:get(2).updateGradInput = function(input) return end -- LMX: model:get(i) return the i-th layer added by model:add

if opt.backend == 'cudnn' then
   require 'cudnn'
   cudnn.benchmark=true -- LMX: if set true, uses the inbuilt cudnn auto-tuner to find the fastest convolution algorithms
   cudnn.convert(model:get(3), cudnn) -- LMX:convert the 3rd layer ('models/'..opt.model..'.lua') to a cudnn type
end

print(model)

print(c.blue '==>' ..' loading data')
provider = torch.load 'provider.0_4'
provider.trainData.data = provider.trainData.data:float()
provider.testData.data = provider.testData.data:float()
nb_classes = 2
confusion = optim.ConfusionMatrix(nb_classes) -- LMX:confusion matrix, or error matrix, Each column of the matrix represents the instances in a predicted class while each row represents the instances in an actual class (or vice versa)

print('Will save at '..opt.save)
paths.mkdir(opt.save)
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))
testLogger:setNames{'% mean class accuracy (train set)', '% mean class accuracy (test set)'}
testLogger.showPlot = false

parameters,gradParameters = model:getParameters()


print(c.blue'==>' ..' setting criterion')
criterion = cast(nn.CrossEntropyCriterion())


print(c.blue'==>' ..' configuring optimizer')
optimState = {
  learningRate = opt.learningRate,
  weightDecay = opt.weightDecay,
  momentum = opt.momentum,
  learningRateDecay = opt.learningRateDecay,
}


function train()
  model:training() -- LMX:This sets the mode of the Module (or sub-modules) to train=true. This is useful for modules like Dropout or BatchNormalization that have a different behaviour during training vs evaluation
  epoch = epoch or 1

  -- drop learning rate every "epoch_step" epochs
  if epoch % opt.epoch_step == 0 then optimState.learningRate = optimState.learningRate/2 end
  
  print(c.blue '==>'.." online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')

  local targets = cast(torch.FloatTensor(opt.batchSize))
  local indices = torch.randperm(provider.trainData.data:size(1)):long():split(opt.batchSize) -- LMX: [result] split([result,] tensor, size, [dim]) Splits Tensor tensor along dimension dim into a result table of Tensors of size size (a number) or less (in the case of the last Tensor), Argument dim defaults to 1.
  -- remove last element so that all the batches have equal size
  indices[#indices] = nil

  local tic = torch.tic() -- LMX: start timer
  for t,v in ipairs(indices) do -- LMX:The ipairs(table) function will allow iteration over index-value pairs
    xlua.progress(t, #indices)

    local inputs = provider.trainData.data:index(1,v) -- LMX:[Tensor] index (dim,index) returns a new Tensor which indexes the given tensor along dimension dim and using the entries in torch.LongTensor index. The returned tensor does not use the same storage as the original tensor
    targets:copy(provider.trainData.labels:index(1,v)) --LMX: for example y = x:index(1,torch.LongTensor{3,1}) returns y with 1st element = 3rd element in x, and 2nd element = 1st element in x

    -- LMX:In feval the cost and the gradient are computed per batch, no need to divide them by batch size and return the average cost and gradients, because nn.CrossEntropyCriterion divides the loss and the gradient that's propagated back by batch size
    local feval = function(x) -- LMX: evaluation function that returns cost and gradParameters, will be called by sgd solver
      if x ~= parameters then parameters:copy(x) end
      gradParameters:zero()
      
      local outputs = model:forward(inputs)
      local f = criterion:forward(outputs, targets)
      local df_do = criterion:backward(outputs, targets)
      model:backward(inputs, df_do)
      confusion:batchAdd(outputs, targets)
      return f,gradParameters 
    end
    optim.sgd(feval, parameters, optimState)
  end

  confusion:updateValids()
  print(('Train accuracy: '..c.cyan'%.2f'..' %%\t time: %.2f s'):format(
        confusion.totalValid * 100, torch.toc(tic)))

  train_acc = confusion.totalValid * 100

  confusion:zero()
  epoch = epoch + 1
end


function test()
  -- disable flips, dropouts and batch normalization
  model:evaluate() -- LMX:This sets the mode of the Module (or sub-modules) to train=false. This is useful for modules like Dropout or BatchNormalization that have a different behaviour during training vs evaluation.
  print(c.blue '==>'.." testing")
  local bs = 125

  for i=1,provider.testData.data:size(1),bs do -- LMX: for var=exp1,exp2,exp3 means for each value of var from exp1 to exp2, using exp3 as the step to increment var, exp3 = 1 by default
    local outputs = model:forward(provider.testData.data:narrow(1,i,bs)) -- LMX:[self] narrow(dim, index, size) Returns a new Tensor which is a narrowed version of the current one: narrow dimension 1 from index to index+size-1, take element bs, 2bs, ...
    confusion:batchAdd(outputs, provider.testData.labels:narrow(1,i,bs))
  end

  confusion:updateValids()
  print('Test accuracy:', confusion.totalValid * 100)
  
  if testLogger then
    paths.mkdir(opt.save)
    testLogger:add{train_acc, confusion.totalValid * 100}
    testLogger:style{'-','-'}
    testLogger:plot()

    if paths.filep(opt.save..'/test.log.eps') then
      local base64im
      do
        os.execute(('convert -density 200 %s/test.log.eps %s/test.png'):format(opt.save,opt.save))
        os.execute(('openssl base64 -in %s/test.png -out %s/test.base64'):format(opt.save,opt.save))
        local f = io.open(opt.save..'/test.base64')
        if f then base64im = f:read'*all' end
      end

      local file = io.open(opt.save..'/report.html','w')
      file:write(([[
      <!DOCTYPE html>
      <html>
      <body>
      <title>%s - %s</title>
      <img src="data:image/png;base64,%s">
      <h4>optimState:</h4>
      <table>
      ]]):format(opt.save,epoch,base64im))
      for k,v in pairs(optimState) do
        if torch.type(v) == 'number' then
          file:write('<tr><td>'..k..'</td><td>'..v..'</td></tr>\n')
        end
      end
      file:write'</table><pre>\n'
      file:write(tostring(confusion)..'\n')
      file:write(tostring(model)..'\n')
      file:write'</pre></body></html>'
      file:close()
    end
  end

  -- save model every 50 epochs
  if epoch % 50 == 0 then
    local filename = paths.concat(opt.save, 'model.net')
    print('==> saving model to '..filename)
    torch.save(filename, model:get(3):clearState()) -- LMX:clearState() clears intermediate module states as output, gradInput and others. Useful when serializing networks and running low on memory
  end

  confusion:zero()
end


for i=1,opt.max_epoch do
  train()
  test()
end


