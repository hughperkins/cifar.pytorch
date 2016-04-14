require 'xlua'
require 'optim'
require 'cunn'
require 'batchflip'

local Train = torch.class('Train')

-- opt is dict of:
--  backend
--  model
--  weightDecay
--  momentum
function Train.__init(self, opt)
  self.opt = opt

  local model = nn.Sequential()
  model:add(nn.BatchFlip():float())
  model:add(nn.Copy('torch.FloatTensor','torch.CudaTensor'):cuda())
  model:add(dofile('models/'..opt.model..'.lua'):cuda())
  model:get(2).updateGradInput = function(input) return end

  if opt.backend == 'cudnn' then
     require 'cudnn'
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

function Train.trainBatch(self, learningRate, inputs, targets)
  local opt = self.opt
  local model = self.model
  local optimState = self.optimState

  optimState.learningRate = learningRate
  model:training()

  local feval = function(x)
    if x ~= self.parameters then self.parameters:copy(x) end
    self.gradParameters:zero()
    
    local outputs = model:forward(inputs)
    local f = criterion:forward(outputs, targets)
    local df_do = criterion:backward(outputs, targets)
    model:backward(inputs, df_do)

    confusion:batchAdd(outputs, targets)

    return f, self.gradParameters
  end
  optim.sgd(feval, self.parameters, optimState)
end

function Train.testBatch(self, inputs)
  local model = self.model

  -- disable flips, dropouts and batch normalization
  model:evaluate()
  local outputs = model:forward(inputs)
  return outputs
end

function Train.save(self, filepath)
  torch.save(filepath, self.model:get(3):clearState())
end

