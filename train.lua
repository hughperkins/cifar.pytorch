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
  self.backend = opt.backend
  local model = nn.Sequential()
  model:add(nn.BatchFlip():float())
  model:add(self:cast(nn.Copy('torch.FloatTensor', torch.type(self:cast(torch.Tensor())))))
  model:add(self:cast(dofile('models/'..opt.model..'.lua')))
  model:get(2).updateGradInput = function(input) return end

  if opt.backend == 'cudnn' then
    print('using cudnn')
    require 'cudnn'
    if opt.cudnnfastest then
      print('Using cudnn \'fastest\' mode')
      cudnn.fastest = true
      cudnn.benchmark = true
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

function Trainer.cast(self, t)
  local backend = self.backend
  if backend == 'cuda' then
    require 'cunn'
    return t:cuda()
  elseif backend == 'float' then
    return t:float()
  elseif backend == 'cl' then
    require 'clnn'
    return t:cl()
  else
    error('Unknown type '..opt.type)
  end
end

function Trainer.trainBatch(self, learningRate, inputs, targets)
  local opt = self.opt

  local loss = nil
  self.optimState.learningRate = learningRate
  self.model:training()
  self.cutargets = self.cutargets or self:cast(torch.Tensor(target:size()))
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
  local _, predictions = outputs:max(2)
  return predictions:byte()
end

function Trainer.save(self, filepath)
  torch.save(filepath, self.model:get(3):clearState())
end

