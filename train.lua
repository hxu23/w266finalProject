
require 'torch'
require 'nn'
require 'nngraph'
-- exotic things
require 'loadcaffe'
-- local imports
local utils = require 'misc.utils'
require 'misc.DataLoader'
require 'misc.LanguageModel'
local net_utils = require 'misc.net_utils'
require 'misc.optim_updates'
require 'misc.LayoutEncoder'
require 'misc.LayoutEncoderLocation'
require 'misc.LayoutEncoderAttention'
require 'misc.LanguageModelAttention'
require 'misc.LayoutEncoderLocationAttention'
require 'paths'

-------------------------------------------------------------------------------
-- Input arguments and options
-------------------------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('Train an Image Captioning model')
cmd:text()
cmd:text('Options')

cmd:option('-generate_from', 'objname_location_image', 'based on which descriptions are generated')
cmd:option('-attention', 'without_attention', 'whether or not use attention')
cmd:option('-report_test_after', 2500 * 50, 'after so many iterations eval the test set')
cmd:option('-test_images_use', 5000, 'number of test images for evaluation')
cmd:option('-start_iter', 0, '')


-- Data input settings
cmd:option('-input_h5', 'coco/data.h5', 'path to the h5file containing the preprocessed dataset')
cmd:option('-input_json', 'coco/data.json', 'path to the json file containing additional info and vocab')
cmd:option('-cnn_proto', 'model/VGG_ILSVRC_16_layers_deploy.prototxt', 'path to CNN prototxt file in Caffe format. Note this MUST be a VGGNet-16 right now.')
cmd:option('-cnn_model', 'model/VGG_ILSVRC_16_layers.caffemodel', 'path to CNN model file containing the weights, Caffe format. Note this MUST be a VGGNet-16 right now.')
cmd:option('-start_from', '', 'path to a model checkpoint to initialize model weights from. Empty = don\'t')

-- Model settings
cmd:option('-rnn_size', 512, 'size of the rnn in number of hidden nodes in each layer')
cmd:option('-input_encoding_size', 512, 'the encoding size of each token in the vocabulary, and the image.')

-- Optimization: General
cmd:option('-max_iters', -1, 'max number of iterations to run for (-1 = run forever)')
cmd:option('-batch_size', 16, 'what is the batch size in number of images per batch? (there will be x seq_per_img sentences)')
cmd:option('-grad_clip', 0.1, 'clip gradients at this value (note should be lower than usual 5 because we normalize grads by both batch and seq_length)')
cmd:option('-drop_prob_lm', 0.5, 'strength of dropout in the Language Model RNN')
cmd:option('-finetune_cnn_after', 250000, 'After what iteration do we start finetuning the CNN? (-1 = disable; never finetune, 0 = finetune from start)')
cmd:option('-seq_per_img', 5, 'number of captions to sample for each image during training. Done for efficiency since CNN forward pass is expensive. E.g. coco has 5 sents/image')
-- Optimization: for the Language Model
cmd:option('-optim', 'adam', 'what update to use? rmsprop|sgd|sgdmom|adagrad|adam')
cmd:option('-learning_rate', 4e-4, 'learning rate')
cmd:option('-learning_rate_decay_start', -1, 'at what iteration to start decaying learning rate? (-1 = dont)')
cmd:option('-learning_rate_decay_every', 50000, 'every how many iterations thereafter to drop LR by half?')
cmd:option('-optim_alpha', 0.8, 'alpha for adagrad/rmsprop/momentum/adam')
cmd:option('-optim_beta', 0.999, 'beta used for adam')
cmd:option('-optim_epsilon', 1e-8, 'epsilon that goes into denominator for smoothing')
-- Optimization: for the CNN
cmd:option('-cnn_optim', 'adam', 'optimization to use for CNN')
cmd:option('-cnn_optim_alpha', 0.8, 'alpha for momentum of CNN')
cmd:option('-cnn_optim_beta', 0.999, 'alpha for momentum of CNN')
cmd:option('-cnn_learning_rate', 1e-5, 'learning rate for the CNN')
cmd:option('-cnn_weight_decay', 0, 'L2 weight decay just for the CNN')

-- Evaluation/Checkpointing
cmd:option('-val_images_use', 3200, 'how many images to use when periodically evaluating the validation loss? (-1 = all)')
cmd:option('-save_checkpoint_every', 2500, 'how often to save a model checkpoint?')
cmd:option('-checkpoint_path', '', 'folder to save checkpoints into (empty = this folder)')
cmd:option('-language_eval', 0, 'Evaluate language as well (1 = yes, 0 = no)? BLEU/CIDEr/METEOR/ROUGE_L? requires coco-caption code from Github.')
cmd:option('-losses_log_every', 25, 'How often do we snapshot losses, for inclusion in the progress dump? (0 = disable)')

-- misc
cmd:option('-backend', 'cudnn', 'nn|cudnn')
cmd:option('-id', '', 'an id identifying this run/job. used in cross-val and appended when writing progress files')
cmd:option('-seed', 123, 'random number generator seed to use')
cmd:option('-gpuid', 0, 'which gpu to use. -1 = use CPU')

cmd:text()

-------------------------------------------------------------------------------
-- Basic Torch initializations
-------------------------------------------------------------------------------
local opt = cmd:parse(arg)
torch.manualSeed(opt.seed)
torch.setdefaulttensortype('torch.FloatTensor') -- for CPU

if opt.gpuid >= 0 then
  require 'cutorch'
  require 'cunn'
  if opt.backend == 'cudnn' then require 'cudnn' end
  cutorch.manualSeed(opt.seed)
  cutorch.setDevice(opt.gpuid + 1) -- note +1 because lua is 1-indexed
end

-- the rnn package provide nn.LinearNoBias for implementing attention
if opt.attention == 'with_attention' then require 'rnn' end

local use_cnn, use_layout_encoder = false, false
if opt.generate_from == 'image' or opt.generate_from == 'objname_location_image' then
  use_cnn = true
end
if opt.generate_from ~= 'image' then
  use_layout_encoder = true
end

if opt.attention == 'with_attention' and use_cnn then
  error('generating from image with attention not implemented')
end

-------------------------------------------------------------------------------
-- Create the Data Loader instance
-------------------------------------------------------------------------------
local loader = DataLoader{h5_file = opt.input_h5, json_file = opt.input_json}

-------------------------------------------------------------------------------
-- Initialize the networks
-------------------------------------------------------------------------------
local protos = {}

if string.len(opt.start_from) > 0 then
  -- load protos from file
  print('initializing weights from ' .. opt.start_from)
  local loaded_checkpoint = torch.load(opt.start_from)
  protos = loaded_checkpoint.protos
  net_utils.unsanitize_gradients(protos.cnn)
  local lm_modules = protos.lm:getModulesList()
  for k,v in pairs(lm_modules) do net_utils.unsanitize_gradients(v) end
  local layout_encoder_modules = protos.layout_encoder:getModulesList()
  for k, v in pairs(layout_encoder_modules) do net_utils.unsanitize_gradients(v) end
  protos.crit = nn.LanguageModelCriterion() -- not in checkpoints, create manually

  -- protos.category_expander = nn.FeatExpander(opt.seq_per_img)
  opt.start_iter = loaded_checkpoint.iter - 1
else
  -- create protos from scratch
  -- intialize language model
  local lmOpt = {}
  lmOpt.vocab_size = loader:getVocabSize()
  lmOpt.input_encoding_size = opt.input_encoding_size
  lmOpt.rnn_size = opt.rnn_size
  lmOpt.num_layers = 1
  lmOpt.dropout = opt.drop_prob_lm
  lmOpt.seq_length = loader:getSeqLength()
  lmOpt.batch_size = opt.batch_size * opt.seq_per_img

  if opt.generate_from == 'objclass' then
    lmOpt.layout_encoder_seq_length = lmOpt.seq_length
  else
    lmOpt.layout_encoder_seq_length = lmOpt.seq_length * 6
  end

  if opt.attention == 'with_attention' then
    protos.lm = nn.LanguageModelAttention(lmOpt)
  else
    protos.lm = nn.LanguageModel(lmOpt)
  end

  if opt.generate_from == 'objname_location_image' or opt.generate_from == 'image' then
    protos.layout_encoder = nn.LayoutEncoderLocation(lmOpt)
  elseif opt.generate_from == 'objname_location' then
    if opt.attention == 'with_attention' then
      protos.layout_encoder = nn.LayoutEncoderLocationAttention(lmOpt)
    else
      protos.layout_encoder = nn.LayoutEncoderLocation(lmOpt)
    end
  elseif opt.generate_from == 'objclass' or opt.generate_from == 'objname' then
    if opt.attention == 'with_attention' then
      protos.layout_encoder = nn.LayoutEncoderAttention(lmOpt)
    else
      protos.layout_encoder = nn.LayoutEncoder(lmOpt)
    end
  else
    error('bad option for generate_from')
  end

  -- initialize the ConvNet
  local cnn_backend = opt.backend
  if opt.gpuid == -1 then cnn_backend = 'nn' end -- override to nn if gpu is disabled
  local cnn_raw = loadcaffe.load(opt.cnn_proto, opt.cnn_model, cnn_backend)
  protos.cnn = net_utils.build_cnn(cnn_raw, {encoding_size = opt.input_encoding_size, backend = cnn_backend})
  -- initialize a special FeatExpander module that "corrects" for the batch number discrepancy 
  -- where we have multiple captions per one image in a batch. This is done for efficiency
  -- because doing a CNN forward pass is expensive. We expand out the CNN features for each sentence
  -- protos.expander = nn.FeatExpander(opt.seq_per_img)
  -- criterion for the language model
  protos.crit = nn.LanguageModelCriterion()
end

if opt.attention == 'with_attention' then
  -- batch_size x seq_length x feat_size expanded to
  -- batch_size*seq_per_img x seq_length x feat_size
  -- seq_length is not known until computed
  protos.expander = nn.Sequential()
  protos.expander:add(nn.Replicate(opt.seq_per_img, 2)):add(nn.Contiguous())
  protos.expander:add(nn.View(opt.batch_size*opt.seq_per_img, -1, opt.rnn_size))
else
  protos.expander = nn.FeatExpander(opt.seq_per_img) -- not in checkpoints, create manually
end

-- ship everything to GPU, maybe
if opt.gpuid >= 0 then
  for k,v in pairs(protos) do v:cuda() end
end

-- flatten and prepare all model parameters to a single vector. 
-- Keep CNN params separate in case we want to try to get fancy with different optims on LM/CNN
local params, grad_params = protos.lm:getParameters()
local layout_encoder_params, layout_encoder_grad_params = protos.layout_encoder:getParameters()
local cnn_params, cnn_grad_params = protos.cnn:getParameters()
print('total number of parameters in LM: ', params:nElement())
print('total number of parameters in CNN: ', cnn_params:nElement())
print('total number of parameters in Layout Encoder: ', layout_encoder_params:nElement())
assert(params:nElement() == grad_params:nElement())
assert(cnn_params:nElement() == cnn_grad_params:nElement())
assert(layout_encoder_params:nElement() == layout_encoder_grad_params:nElement())

-- construct thin module clones that share parameters with the actual
-- modules. These thin module will have no intermediates and will be used
-- for checkpointing to write significantly smaller checkpoint files
local thin_lm = protos.lm:clone()
thin_lm.core:share(protos.lm.core, 'weight', 'bias') -- TODO: we are assuming that LM has specific members! figure out clean way to get rid of, not modular.
thin_lm.lookup_table:share(protos.lm.lookup_table, 'weight', 'bias')
local thin_cnn = protos.cnn:clone('weight', 'bias')
local thin_layout_encoder = protos.layout_encoder:clone()
thin_layout_encoder.core:share(protos.layout_encoder.core, 'weight', 'bias')
thin_layout_encoder.lookup_table:share(protos.layout_encoder.lookup_table, 'weight', 'bias')
-- sanitize all modules of gradient storage so that we dont save big checkpoints
net_utils.sanitize_gradients(thin_cnn)
local lm_modules = thin_lm:getModulesList()
for k,v in pairs(lm_modules) do net_utils.sanitize_gradients(v) end
local layout_encoder_modules = thin_layout_encoder:getModulesList()
for k,v in pairs(layout_encoder_modules) do net_utils.sanitize_gradients(v) end


-- create clones and ensure parameter sharing. we have to do this 
-- all the way here at the end because calls such as :cuda() and
-- :getParameters() reshuffle memory around.
print("RNN creating clones for parameter sharing")
protos.lm:createClones()
protos.layout_encoder:createClones()
collectgarbage() -- "yeah, sure why not"

local function bboxencode2coords(bbox_encode)
  local x = torch.div(bbox_encode, 1e9)
  local y = torch.div(bbox_encode % 1e9, 1e6)
  local width = torch.div(bbox_encode % 1e6, 1e3)
  local height = bbox_encode % 1e3
  local bbox_coords = torch.Tensor(bbox_encode:size(1), bbox_encode:size(2), 4):typeAs(x):zero()
  bbox_coords[{ {}, {}, 1 }] = x
  bbox_coords[{ {}, {}, 2 }] = y
  bbox_coords[{ {}, {}, 3 }] = width
  bbox_coords[{ {}, {}, 4 }] = height
  bbox_coords = bbox_coords:float():div(608)
  return bbox_coords
end

local function compute_forward_feats(data, protos, opt)
  local forward_feats = {}

  -- bbox_coords shape batch_size x seq_length x 4
  local bbox_coords = bboxencode2coords(data.bbox)
  local bbox_coords_T = bbox_coords:transpose(1, 2)
  if opt.gpuid >= 0 then bbox_coords_T = bbox_coords_T:cuda() end
  -- data.full_category is of shape batch_size x seq_length
  local data_full_category_T = data.full_category:transpose(1, 2)
  if opt.gpuid >= 0 then data_full_category_T = data_full_category_T:cuda() end
  local data_category_T = data.category:transpose(1, 2)

  if opt.generate_from == 'image' then
    -- forward the ConvNet on images (most work happens here)
    forward_feats.im_feats = protos.cnn:forward(data.images)
    -- we have to expand out image features, once for each sentence
    forward_feats.expanded_im_feats = protos.expander:forward(forward_feats.im_feats)
    forward_feats.lm_input_feats = forward_feats.expanded_im_feats
    forward_feats.lm_input_feats_no_expand = forward_feats.im_feats
  elseif opt.generate_from == 'objclass' then
    forward_feats.layout_feats = protos.layout_encoder:forward(data_category_T)
    forward_feats.expanded_layout_feats = protos.expander:forward(forward_feats.layout_feats)
    forward_feats.lm_input_feats = forward_feats.expanded_layout_feats
    forward_feats.lm_input_feats_no_expand = forward_feats.layout_feats
  elseif opt.generate_from == 'objname' then
    forward_feats.layout_feats = protos.layout_encoder:forward(data_full_category_T)
    forward_feats.expanded_layout_feats = protos.expander:forward(forward_feats.layout_feats)
    forward_feats.lm_input_feats = forward_feats.expanded_layout_feats
    forward_feats.lm_input_feats_no_expand = forward_feats.layout_feats
  elseif opt.generate_from == 'objname_location' then
    forward_feats.layout_feats = protos.layout_encoder:forward({ data_full_category_T, bbox_coords_T })
    forward_feats.expanded_layout_feats = protos.expander:forward(forward_feats.layout_feats)
    forward_feats.lm_input_feats = forward_feats.expanded_layout_feats
    forward_feats.lm_input_feats_no_expand = forward_feats.layout_feats
  elseif opt.generate_from == 'objname_location_image' then
    forward_feats.im_feats = protos.cnn:forward(data.images)
    forward_feats.layout_feats = protos.layout_encoder:forward({ data_full_category_T, bbox_coords_T })
    -- forward_feats.comb_feats = nn.CAddTable():cuda():forward({ forward_feats.layout_feats, forward_feats.im_feats })
    forward_feats.comb_feats = forward_feats.layout_feats + forward_feats.im_feats
    forward_feats.expanded_comb_feats = protos.expander:forward(forward_feats.comb_feats)
    forward_feats.lm_input_feats = forward_feats.expanded_comb_feats
    forward_feats.lm_input_feats_no_expand = forward_feats.comb_feats
  else
    error('bad option for generate_from')
  end
  return forward_feats
end

-------------------------------------------------------------------------------
-- Validation evaluation
-------------------------------------------------------------------------------
local function eval_split(split, evalopt)
  local verbose = utils.getopt(evalopt, 'verbose', true)
  local val_images_use = utils.getopt(evalopt, 'val_images_use', true)

  protos.cnn:evaluate()
  protos.lm:evaluate()
  protos.layout_encoder:evaluate()
  loader:resetIterator(split) -- rewind iteator back to first datapoint in the split
  local n = 0
  local loss_sum = 0
  local loss_evals = 0
  local predictions = {}
  local vocab = loader:getVocab()
  local max_samples = val_images_use
  local samples_processed = 0
  while true do

    -- fetch a batch of data
    local data = loader:getBatch { batch_size = math.min(opt.batch_size, (max_samples - samples_processed)),
                                   split = split, seq_per_img = opt.seq_per_img }
    data.images = net_utils.prepro(data.images, false, opt.gpuid >= 0) -- preprocess in place, and don't augment
    n = n + data.images:size(1)

    local forward_feats = compute_forward_feats(data, protos, opt)
    local logprobs = protos.lm:forward{forward_feats.lm_input_feats, data.labels}
    local loss = protos.crit:forward(logprobs, data.labels)
    loss_sum = loss_sum + loss
    loss_evals = loss_evals + 1

    local sample_opts = { sample_max = 1, beam_size = 0, temperature = 1.0 }
    local seq = protos.lm:sample(forward_feats.lm_input_feats_no_expand, sample_opts)

    local sents = net_utils.decode_sequence(vocab, seq)
    local gt_sents = net_utils.decode_sequence(vocab, data.labels)
    for k = 1, #sents do
      local entry = { image_id = data.infos[k].id, caption = sents[k] }
      table.insert(predictions, entry)
      if verbose then
        print(string.format('image coco/images/%s : %s', entry.image_id, entry.caption))
        for seq_idx = 1, opt.seq_per_img do
          print(string.format('gt %d: %s', seq_idx, gt_sents[(k - 1) * opt.seq_per_img + seq_idx]))
        end
        print('------------------------------------------------------------------------')
      end
    end

    -- if we wrapped around the split or used up val imgs budget then bail
    local ix0 = data.bounds.it_pos_now
    local ix1 = math.min(data.bounds.it_max, val_images_use)
    max_samples = math.min(data.bounds.it_max, val_images_use)
    if verbose then
      print(string.format('evaluating validation performance... %d/%d (%f)', ix0 - 1, ix1, loss))
    end
    samples_processed = ix0 - 1
    if loss_evals % 10 == 0 then collectgarbage() end
    if data.bounds.wrapped then break end -- the split ran out of data, lets break out
    if n >= val_images_use then break end -- we've used enough images
  end

  local lang_stats
  if opt.language_eval == 1 then
    lang_stats = net_utils.language_eval(predictions, opt.id)
  end

  return loss_sum/loss_evals, predictions, lang_stats
end

-------------------------------------------------------------------------------
-- Loss function
-------------------------------------------------------------------------------
local iter = opt.start_iter
local function lossFun()
  if use_cnn then protos.cnn:training() end
  protos.lm:training()
  grad_params:zero()
  if use_cnn and opt.finetune_cnn_after >= 0 and iter >= opt.finetune_cnn_after then
    cnn_grad_params:zero()
  end
  if use_layout_encoder then
    protos.layout_encoder:training()
    layout_encoder_grad_params:zero()
  end

  -----------------------------------------------------------------------------
  -- Forward pass
  -----------------------------------------------------------------------------
  -- get batch of data  
  local data = loader:getBatch{batch_size = opt.batch_size, split = 'train', seq_per_img = opt.seq_per_img}
  data.images = net_utils.prepro(data.images, true, opt.gpuid >= 0) -- preprocess in place, do data augmentation
  -- data.images: Nx3x224x224 
  -- data.seq: LxM where L is sequence length upper bound, and M = N*seq_per_img

  local forward_feats = compute_forward_feats(data, protos, opt)
  local logprobs = protos.lm:forward{forward_feats.lm_input_feats, data.labels}
  -- forward the language model criterion
  local loss = protos.crit:forward(logprobs, data.labels)

  -----------------------------------------------------------------------------
  -- Backward pass
  -----------------------------------------------------------------------------
  -- backprop criterion
  local dlogprobs = protos.crit:backward(logprobs, data.labels)
  -- backprop language model
  forward_feats.dlm_input_feats, ddummy = unpack(protos.lm:backward({forward_feats.lm_input_feats, data.labels}, dlogprobs))

  if opt.generate_from == 'image' then
    -- backprop the CNN, but only if we are finetuning
    if opt.finetune_cnn_after >= 0 and iter >= opt.finetune_cnn_after then
      local dexpanded_im_feats = forward_feats.dlm_input_feats
      local dim_feats = protos.expander:backward(forward_feats.im_feats, dexpanded_im_feats)
      local dx = protos.cnn:backward(data.images, dim_feats)
    end
  elseif opt.generate_from == 'objclass' then
    local dexpanded_layout_feats = forward_feats.dlm_input_feats
    local dlayout_feats = protos.expander:backward(forward_feats.layout_feats, dexpanded_layout_feats)
    protos.layout_encoder:backward(data_category_T, dlayout_feats)
  elseif opt.generate_from == 'objname' then
    local dexpanded_layout_feats = forward_feats.dlm_input_feats
    local dlayout_feats = protos.expander:backward(forward_feats.layout_feats, dexpanded_layout_feats)
    protos.layout_encoder:backward(data_full_category_T, dlayout_feats)
  elseif opt.generate_from == 'objname_location' then
    local dexpanded_layout_feats = forward_feats.dlm_input_feats
    local dlayout_feats = protos.expander:backward(forward_feats.layout_feats, dexpanded_layout_feats)
    protos.layout_encoder:backward({ data_full_category_T, bbox_coords_T }, dlayout_feats)
  elseif opt.generate_from == 'objname_location_image' then
    local dexpanded_comb_feats = forward_feats.dlm_input_feats
    local dcomb_feats = protos.expander:backward(forward_feats.comb_feats, dexpanded_comb_feats)
    local dlayout_feats, dim_feats = dcomb_feats, dcomb_feats
    protos.layout_encoder:backward({ data_full_category_T, bbox_coords_T }, dlayout_feats)
    if opt.finetune_cnn_after >= 0 and iter >= opt.finetune_cnn_after then
      local dx = protos.cnn:backward(data.images, dim_feats)
    end
  else
    error('bad option for generate_from')
  end

  -- clip gradients
  -- print(string.format('claming %f%% of gradients', 100*torch.mean(torch.gt(torch.abs(grad_params), opt.grad_clip))))
  grad_params:clamp(-opt.grad_clip, opt.grad_clip)
  
  if use_layout_encoder then
    -- layout_encoder_grad_params:add(opt.cnn_weight_decay, layout_encoder_params)
    layout_encoder_grad_params:clamp(-opt.grad_clip, opt.grad_clip)
  end
  -- apply L2 regularization
  if use_cnn and opt.cnn_weight_decay > 0 then
    cnn_grad_params:add(opt.cnn_weight_decay, cnn_params)
    -- note: we don't bother adding the l2 loss to the total loss, meh.
    cnn_grad_params:clamp(-opt.grad_clip, opt.grad_clip)
  end

  -----------------------------------------------------------------------------

  -- and lets get out!
  local losses = { total_loss = loss }
  return losses
end

-------------------------------------------------------------------------------
-- Main loop
-------------------------------------------------------------------------------
local loss0
local optim_state = {}
local cnn_optim_state = {}
local layout_encoder_optim_state = {}
local loss_history = {}
local val_lang_stats_history = {}
local test_lang_stats_history = {}
local val_loss_history = {}
local best_score

if string.len(opt.start_from) > 0 then
  local json_file = opt.start_from:match("(.+)%..+") .. '.json'
  print(json_file)
  local json_data = utils.read_json(json_file)
  loss_history = json_data.loss_history
  val_lang_stats_history = json_data.val_lang_stats_history
  test_lang_stats_history = json_data.test_lang_stats_history
  val_loss_history = json_data.val_loss_history
end

while true do

  -- eval loss/gradient
  local losses = lossFun()
  if iter % opt.losses_log_every == 0 then loss_history[iter] = losses.total_loss end
  print(string.format('iter %d: %f', iter, losses.total_loss))

  -- save checkpoint once in a while (or on final iteration)
  if (iter % opt.save_checkpoint_every == 0 or iter == opt.max_iters) then

    -- evaluate the validation performance
    local val_loss, val_predictions, lang_stats = eval_split('val', { val_images_use = opt.val_images_use })
    print('validation loss: ', val_loss)
    print(lang_stats)
    val_loss_history[iter] = val_loss
    if lang_stats then
      val_lang_stats_history[iter] = lang_stats
    end

    local checkpoint_path = path.join(opt.checkpoint_path, 'model_id' .. opt.id)

    if iter >= opt.report_test_after then
      local test_loss, test_predictions, test_lang_stats = eval_split('test', { val_images_use = opt.test_images_use })
      test_lang_stats_history[iter] = test_lang_stats
    end

    -- write a (thin) json report
    local checkpoint = {}
    checkpoint.opt = opt
    checkpoint.iter = iter
    checkpoint.loss_history = loss_history
    checkpoint.val_loss_history = val_loss_history
    checkpoint.val_predictions = val_predictions -- save these too for CIDEr/METEOR/etc eval
    checkpoint.val_lang_stats_history = val_lang_stats_history
    checkpoint.test_lang_stats_history = test_lang_stats_history

    utils.write_json(checkpoint_path .. '.json', checkpoint)
    print('wrote json checkpoint to ' .. checkpoint_path .. '.json')

    -- write the full model checkpoint as well if we did better than ever
    local current_score
    if lang_stats then
      -- use CIDEr score for deciding how well we did
      current_score = lang_stats['CIDEr']
    else
      -- use the (negative) validation loss as a score
      current_score = -val_loss
    end
    if best_score == nil or current_score > best_score then
      best_score = current_score
      if iter > 0 then -- dont save on very first iteration
        -- include the protos (which have weights) and save to file
        local save_protos = {}
        save_protos.lm = thin_lm -- these are shared clones, and point to correct param storage
        save_protos.cnn = thin_cnn
        save_protos.layout_encoder = thin_layout_encoder
        checkpoint.protos = save_protos
        checkpoint.iter = iter + 1
        -- also include the vocabulary mapping so that we can use the checkpoint 
        -- alone to run on arbitrary images without the data loader
        checkpoint.vocab = loader:getVocab()
        torch.save(checkpoint_path .. '.t7', checkpoint)
        print('wrote checkpoint to ' .. checkpoint_path .. '.t7')
      end
    end
  end

  -- decay the learning rate for both LM and CNN
  local learning_rate = opt.learning_rate
  local cnn_learning_rate = opt.cnn_learning_rate
  if iter > opt.learning_rate_decay_start and opt.learning_rate_decay_start >= 0 then
    local frac = (iter - opt.learning_rate_decay_start) / opt.learning_rate_decay_every
    local decay_factor = math.pow(0.5, frac)
    learning_rate = learning_rate * decay_factor -- set the decayed rate
    cnn_learning_rate = cnn_learning_rate * decay_factor
  end

  -- perform a parameter update
  if opt.optim == 'rmsprop' then
    rmsprop(params, grad_params, learning_rate, opt.optim_alpha, opt.optim_epsilon, optim_state)
  elseif opt.optim == 'adagrad' then
    adagrad(params, grad_params, learning_rate, opt.optim_epsilon, optim_state)
  elseif opt.optim == 'sgd' then
    sgd(params, grad_params, opt.learning_rate)
  elseif opt.optim == 'sgdm' then
    sgdm(params, grad_params, learning_rate, opt.optim_alpha, optim_state)
  elseif opt.optim == 'sgdmom' then
    sgdmom(params, grad_params, learning_rate, opt.optim_alpha, optim_state)
  elseif opt.optim == 'adam' then
    adam(params, grad_params, learning_rate, opt.optim_alpha, opt.optim_beta, opt.optim_epsilon, optim_state)
  else
    error('bad option opt.optim')
  end

  -- do a cnn update (if finetuning, and if rnn above us is not warming up right now)
  if use_cnn and opt.finetune_cnn_after >= 0 and iter >= opt.finetune_cnn_after then
    if opt.cnn_optim == 'sgd' then
      sgd(cnn_params, cnn_grad_params, cnn_learning_rate)
    elseif opt.cnn_optim == 'sgdm' then
      sgdm(cnn_params, cnn_grad_params, cnn_learning_rate, opt.cnn_optim_alpha, cnn_optim_state)
    elseif opt.cnn_optim == 'adam' then
      adam(cnn_params, cnn_grad_params, cnn_learning_rate, opt.cnn_optim_alpha, opt.cnn_optim_beta, opt.optim_epsilon, cnn_optim_state)
    else
      error('bad option for opt.cnn_optim')
    end
  end

  if use_layout_encoder then
    -- perform a parameter update
    if opt.optim == 'rmsprop' then
      rmsprop(layout_encoder_params, layout_encoder_grad_params, learning_rate, opt.optim_alpha, opt.optim_epsilon, layout_encoder_optim_state)
    elseif opt.optim == 'adagrad' then
      adagrad(layout_encoder_params, layout_encoder_grad_params, learning_rate, opt.optim_epsilon, layout_encoder_optim_state)
    elseif opt.optim == 'sgd' then
      sgd(layout_encoder_params, layout_encoder_grad_params, opt.learning_rate)
    elseif opt.optim == 'sgdm' then
      sgdm(layout_encoder_params, layout_encoder_grad_params, learning_rate, opt.optim_alpha, layout_encoder_optim_state)
    elseif opt.optim == 'sgdmom' then
      sgdmom(layout_encoder_params, layout_encoder_grad_params, learning_rate, opt.optim_alpha, layout_encoder_optim_state)
    elseif opt.optim == 'adam' then
      adam(layout_encoder_params, layout_encoder_grad_params, learning_rate, opt.optim_alpha, opt.optim_beta, opt.optim_epsilon, layout_encoder_optim_state)
    else
      error('bad option opt.optim')
    end
  end


  -- stopping criterions
  iter = iter + 1
  if iter % 10 == 0 then collectgarbage() end -- good idea to do this once in a while, i think
  if loss0 == nil then loss0 = losses.total_loss end
  if losses.total_loss > loss0 * 20 then
    print('loss seems to be exploding, quitting.')
    break
  end
  if opt.max_iters > 0 and iter >= opt.max_iters then break end -- stopping criterion

end
