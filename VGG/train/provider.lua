require 'nn'
require 'image'
require 'xlua'
dofile './loader/torchloader.lua'

local Provider = torch.class 'Provider'

function Provider:__init(full)
  local trsize = 10000
  local tesize = 2000

--	dataset_E= load_pixdb('../E.pdb')
--	dataset_F= load_pixdb('../F.pdb')
	dataset_G= load_pixdb('../0.pdb')
	dataset_H= load_pixdb('../4.pdb')
--[[
_ff=0
_00=0
local enreg = 1
local image = dataset_G.data[enreg]
imgSize = 128*128
for i=1,imgSize do

	if image[i]  > 0 then 
 	_ff = _ff + 1
	end 

	if image[i] == 0 then 
 	_00 = _00 + 1 

	end
end
print (enreg.." 1 => " .. _ff .. " 0 => " .. _00 )
--]]
--	size_E = dataset_E.label:size()[1]
--	size_F = dataset_F.label:size()[1]
--	size_G = dataset_G.label:size()[1]
--	size_H = dataset_H.label:size()[1]
    size_G = 6000
    size_H = 6000
    dataset_G.label:fill(1)
    dataset_H.label:fill(2)
	local coef = 0.8
	
--	size_E_tr = size_E * coef
--	size_F_tr = size_F * coef
--size_E_tr = math.floor(size_E_tr)
--size_F_tr = math.floor(size_F_tr)
--	size_E_te = size_E - size_E_tr
--	size_F_te = size_F - size_F_tr

	--size_G_tr = size_G * coef
	--size_H_tr = size_H * coef
--size_G_tr = math.floor(size_G_tr)
--size_H_tr = math.floor(size_H_tr)

	--size_G_te = size_G - size_G_tr
	--size_H_te = size_H - size_H_tr
    size_G_tr = 5000
    size_H_tr = 5000
    size_G_te = 1000
    size_H_te = 1000
  --trsize = size_G_tr + size_H_tr 
  --tesize = size_G_te + size_H_te
  

  print(trsize)
  print(size_G_tr)
  print(size_H_tr)
  print(tesize)
  print(size_G_te)
  print(size_H_te)
  
  -- load dataset
  imgSize=128*128
  self.trainData = {
     data = torch.Tensor(trsize, imgSize),
     labels = torch.Tensor(trsize),
     size = function() return trsize end
  }
  local trainData = self.trainData
  
print('G')
  trainData.data[{ {1, size_G_tr} }] = dataset_G.data[{{1,size_G_tr}}]
  trainData.labels[{ {1, size_G_tr} }] = dataset_G.label[{{1,size_G_tr}}]
print('H')  
  trainData.data[{ {size_G_tr+1, trsize} }] = dataset_H.data[{{1,size_H_tr}}]
  trainData.labels[{ {size_G_tr+1, trsize} }] = dataset_H.label[{{1,size_H_tr}}]
  
    self.testData = {
     data = torch.Tensor(tesize, imgSize),
     labels = torch.Tensor(tesize),
     size = function() return tesize end
  }
  local testData = self.testData
  
print('G')
  testData.data[{ {1, size_G_te} }] = dataset_G.data[{{1+size_G_tr,size_G}}]
  testData.labels[{ {1, size_G_te} }] = dataset_G.label[{{1+size_G_tr,size_G}}] 
print('H')
  testData.data[{ {size_G_te+1, tesize} }] = dataset_H.data[{{1+size_H_tr,size_H}}]
  testData.labels[{ {size_G_te+1, tesize} }] = dataset_H.label[{{1+size_H_tr,size_H}}] 
print('resize')
  -- resize dataset (if using small version)
  --trainData.data = trainData.data[{ {1,trsize} }]
  --trainData.labels = trainData.labels[{ {1,trsize} }]

  --testData.data = testData.data[{ {1,tesize} }]
  --testData.labels = testData.labels[{ {1,tesize} }]

  -- reshape data
  --[[
local enreg = 1
local image = trainData.data[enreg]
imgSize = 128*128
_ff = 0
_00 = 0
for i=1,imgSize do

	if image[i]  > 0 then 
 	_ff = _ff + 1
	end 

	if image[i] == 0 then 
 	_00 = _00 + 1 

	end
end
print (enreg.." 1 => " .. _ff .. " 0 => " .. _00 )
]]--
  trainData.data = trainData.data:reshape(trsize,1,128,128)
  testData.data = testData.data:reshape(tesize,1,128,128)
  --print(trainData.data[1])
  --print(trainData.data[6000])

local enreg = 1

imgSize = 128*128
_ff = 0
_00 = 0
for i=1,128 do
    for j=1,128 do
	if testData.data[enreg][1][i][j]  > 0 then 
 	_ff = _ff + 1
	end 

	if testData.data[enreg][1][i][j]  == 0 then 
 	_00 = _00 + 1 

	end
    end
end
print (enreg.." 1 => " .. _ff .. " 0 => " .. _00 )

 -- trainData.data = trainData.data[{{},{},{35,98},{35,98}}]
 -- testData.data = testData.data[{{},{},{35,98},{35,98}}]
  
 -- trainData.data = trainData.data:reshape(trsize,1,64,64)
 -- testData.data = testData.data:reshape(tesize,1,64,64)
 self.trainData = trainData
 self.testData = testData
end

function Provider:normalize()
  ----------------------------------------------------------------------
  -- preprocess/normalize train/test sets
  --
  local trainData = self.trainData
  local testData = self.testData

  print '<trainer> preprocessing data (color space + normalization)'
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
