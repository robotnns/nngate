 require 'torch'
--s = torch.ByteStorage(10):fill(1)
--x = torch.Tensor(s, 1, torch.LongStorage{2,5})
--x = torch.ByteTensor(s):type('torch.FloatTensor')
--print(x)
function load_pixdb(filename)

imgSize=128*128
labelSize=1
--print("image size:" .. imgSize)
labelArray = {}
imgArray = {}
dataset ={}
rawdata = torch.ByteStorage(filename)
nbrimage = rawdata:size() / (imgSize+labelSize)
--for i=1,rawdata:size() do
--print ("nbr img " .. nbrimage)
labelTensor = torch.ByteTensor(nbrimage)
imgTensor = torch.ByteTensor(nbrimage,imgSize)
--
for i=1,nbrimage do
--print ("process img " .. i )
label = rawdata[imgSize*(i-1)+i]

--img = torch.ByteStorage(rawdata,imgSize*(i-1)+i+labelSize,imgSize)
--labelArray[i]=label
labelTensor[i] =label
--convert byte to float
tensorImga=torch.ByteTensor(rawdata,imgSize*(i-1)+i+labelSize,imgSize):type('torch.FloatTensor') 
-- x:type('torch.IntTensor')
imgTensor[i]=tensorImga
--print (tensorImga:type())

end


dataset.label = labelTensor:type('torch.FloatTensor') 
dataset.data = imgTensor

--print (labelTensor )
--print (imgArray)
--print (imgArray[3][imgSize])
--print (imgArray[3])
_ff=0
_00=0
enreg = 40000
--for i=1,imgSize do

--	if imgTensor[enreg][i]  > 0 then 
-- 	_ff = _ff + 1
--	end 

--	if imgTensor[enreg][i] == 0 then 
-- 	_00 = _00 + 1 

--	end
--end
print (" 1 => " .. _ff .. " 0 => " .. _00 )


return dataset
end

--dataset= load_pixdb('0.pdb')
--print(dataset.data[2]);
--print(dataset.data);
