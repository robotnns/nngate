require 'nn'
require 'image'
require 'xlua'
dofile './loader/torchloader.lua'

local Provider = torch.class 'Provider'

function Provider:__init(full)
	dataset_1= load_pixdb('./data/0.pdb')
	dataset_2= load_pixdb('./data/4.pdb')

--	size_G = dataset_G.label:size()[1]
--	size_H = dataset_H.label:size()[1]

    dataset_1.label:fill(1)
    dataset_2.label:fill(2)
	--local coef = 0.8
	--size_G_tr = size_G * coef
	--size_H_tr = size_H * coef
    --size_G_tr = math.floor(size_G_tr)
    --size_H_tr = math.floor(size_H_tr)

	--size_G_te = size_G - size_G_tr
	--size_H_te = size_H - size_H_tr
    size_1_tr = 5000
    size_2_tr = 5000
    size_1_te = 1000
    size_2_te = 1000
    size_1 = size_1_tr + size_1_te
    size_2 = size_2_tr + size_2_te
    local trsize = size_1_tr + size_2_tr 
    local tesize = size_1_te + size_2_te
 
  
  -- load dataset
  imgSize=128*128
  self.trainData = {
     data = torch.Tensor(trsize, imgSize),
     labels = torch.Tensor(trsize),
     size = function() return trsize end
  }
  local trainData = self.trainData
  
print('train data class 1')
  trainData.data[{ {1, size_1_tr} }] = dataset_1.data[{{1,size_1_tr}}]
  trainData.labels[{ {1, size_1_tr} }] = dataset_1.label[{{1,size_1_tr}}]
print('train data class 2')  
  trainData.data[{ {size_1_tr+1, trsize} }] = dataset_2.data[{{1,size_2_tr}}]
  trainData.labels[{ {size_1_tr+1, trsize} }] = dataset_2.label[{{1,size_2_tr}}]
  
    self.testData = {
     data = torch.Tensor(tesize, imgSize),
     labels = torch.Tensor(tesize),
     size = function() return tesize end
  }
  local testData = self.testData
  
print('test data class 1')
  testData.data[{ {1, size_1_te} }] = dataset_1.data[{{1+size_1_tr,size_1}}]
  testData.labels[{ {1, size_1_te} }] = dataset_1.label[{{1+size_1_tr,size_1}}] 
print('test data class 2')
  testData.data[{ {size_1_te+1, tesize} }] = dataset_2.data[{{1+size_2_tr,size_2}}]
  testData.labels[{ {size_1_te+1, tesize} }] = dataset_2.label[{{1+size_2_tr,size_2}}] 
print('resize')
  -- resize dataset (if using small version)
  --trainData.data = trainData.data[{ {1,trsize} }]
  --trainData.labels = trainData.labels[{ {1,trsize} }]

  --testData.data = testData.data[{ {1,tesize} }]
  --testData.labels = testData.labels[{ {1,tesize} }]

  -- reshape data
  trainData.data = trainData.data:reshape(trsize,1,128,128)
  testData.data = testData.data:reshape(tesize,1,128,128)
end

function Provider:normalize()
  ----------------------------------------------------------------------
  -- preprocess/normalize train/test sets
  --
  local trainData = self.trainData
  local testData = self.testData

  print '<trainer> preprocessing data (normalization)'
  collectgarbage()

  -- preprocess trainSet
  local normalization = nn.SpatialContrastiveNormalization(1, image.gaussian1D(7))
  local mean = trainData.data:mean()
  local std = trainData.data:std()
  --for i = 1,trainData:size() do
  --   xlua.progress(i, trainData:size())
  --   trainData.data[i] = normalization(trainData.data[{{i}}])
  --end
  trainData.data:add(-mean)
  trainData.data:div(std)
  
  -- preprocess testSet
  local mean = testData.data:mean()
  local std = testData.data:std()
  --for i = 1,testData:size() do
  --   xlua.progress(i, testData:size())
   --  testData.data[i] = normalization(testData.data[{{i}}])
  --end
  testData.data:add(-mean)
  testData.data:div(std)
end
