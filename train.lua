require 'xlua'
require 'optim'
require 'cunn'
require 'image'
require 'batchflip'

local Trainer = torch.class('Trainer')

-- opt is dict of:
--  backend
--  model
--  weightDecay
--  momentum
function Trainer.__init(self, opt)
  self.opt = opt
  local model = nn.Sequential()
  model:add(nn.BatchFlip():float())
  model:add(nn.Copy('torch.FloatTensor','torch.CudaTensor'):cuda())
  model:add(dofile('models/'..opt.model..'.lua'):cuda())
  model:get(2).updateGradInput = function(input) return end

  if opt.backend == 'cudnn' then
     print('using cudnn')
     require 'cudnn'
     if opt.cudnnfastest then
       print('Using cudnn \'fastest\' mode')
       cudnn.fastest = true
     end
     cudnn.convert(model:get(3), cudnn)
  end
  self.model = model

  print(model)

  self.parameters, self.gradParameters = model:getParameters()
  self.criterion = nn.CrossEntropyCriterion():cuda()
  self.optimState = {
    learningRate = opt.learningRate,
    weightDecay = opt.weightDecay,
    momentum = opt.momentum,
    learningRateDecay = opt.learningRateDecay,
  }
end

function Trainer.trainBatch(self, learningRate, inputs, targets)
  local opt = self.opt

  local loss = nil
  self.optimState.learningRate = learningRate
  self.model:training()
  self.cutargets = self.cutargets or torch.CudaTensor(targets:size())
  self.cutargets:resize(targets:size())
  self.cutargets:copy(targets)
  local feval = function(x)
    if x ~= self.parameters then self.parameters:copy(x) end
    self.gradParameters:zero()
    
    local outputs = self.model:forward(inputs)
    loss = self.criterion:forward(outputs, self.cutargets)
    local df_do = self.criterion:backward(outputs, self.cutargets)
    self.model:backward(inputs, df_do)

    return loss, self.gradParameters
  end
  optim.sgd(feval, self.parameters, self.optimState)
  return loss
end

function Trainer.predict(self, inputs)
  -- disable flips, dropouts and batch normalization
  self.model:evaluate()
  local outputs = self.model:forward(inputs)
  return outputs:byte()
end

function Trainer.save(self, filepath)
  torch.save(filepath, self.model:get(3):clearState())
end

