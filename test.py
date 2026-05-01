import argparse
import random

import yaml
import torch
from torch import amp
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

import datasets
import models
import utils
import utils.optimizers as optimizers

def main(config):
  random.seed(0)
  np.random.seed(0)
  torch.manual_seed(0)
  torch.cuda.manual_seed(0)
  # torch.backends.cudnn.deterministic = True
  # torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.benchmark = True
  ##### Dataset #####

  dataset = datasets.make(config['dataset'], **config['test'])
  utils.log('meta-test set: {} (x{}), {}'.format(
    dataset[0][0].shape, len(dataset), dataset.n_classes))
  loader = DataLoader(dataset, config['test']['n_episode'],
    collate_fn=datasets.collate_fn, num_workers=8, pin_memory=True,prefetch_factor=4,persistent_workers=True)

  ##### Model #####

  ckpt = torch.load(config['load'])
  inner_args = utils.config_inner_args(config.get('inner_args'))
  transport_mode = ckpt.get('transport_mode', config.get('transport_mode', 'none'))
  low_rank_rank = int(ckpt.get('low_rank_rank', config.get('low_rank_rank', 4)))
  low_rank_init_scale = float(
    ckpt.get('low_rank_init_scale', config.get('low_rank_init_scale', 1e-3))
  )
  use_shift_aware_residual_gate = ckpt.get(
    'use_shift_aware_residual_gate',
    False
  )
  residual_gate_eps = float(
    ckpt.get('residual_gate_eps', config.get('residual_gate_eps', 0.05))
  )
  residual_gate_hidden_dim = int(
    ckpt.get('residual_gate_hidden_dim', config.get('residual_gate_hidden_dim', 64))
  )
  shift_gate_detach_context = ckpt.get(
    'shift_gate_detach_context',
    config.get('shift_gate_detach_context', True)
  )
  if use_shift_aware_residual_gate and transport_mode != 'scalar_gate':
    raise ValueError(
      'use_shift_aware_residual_gate requires transport_mode="scalar_gate"'
    )
  model = models.load(
    ckpt,
    load_clf=(not inner_args['reset_classifier']),
    transport_mode=transport_mode,
    low_rank_rank=low_rank_rank,
    low_rank_init_scale=low_rank_init_scale,
    use_shift_aware_residual_gate=use_shift_aware_residual_gate,
    residual_gate_eps=residual_gate_eps,
    residual_gate_hidden_dim=residual_gate_hidden_dim,
    shift_gate_detach_context=shift_gate_detach_context
  )

  if args.efficient:
    model.go_efficient()

  if config.get('_parallel'):
    model = nn.DataParallel(model)

  utils.log('num params: {}'.format(utils.compute_n_params(model)))

  ##### Evaluation #####

  model.eval()
  aves_va = utils.AverageMeter()
  va_lst = []

  for epoch in range(1, config['epoch'] + 1):
    for data in tqdm(loader, leave=False):
      x_shot, x_query, y_shot, y_query = data
      x_shot = x_shot.cuda(non_blocking=True)
      y_shot = y_shot.cuda(non_blocking=True)
      x_query = x_query.cuda(non_blocking=True)
      y_query = y_query.cuda(non_blocking=True)


      if inner_args['reset_classifier']:
        if config.get('_parallel'):
          model.module.reset_classifier()
        else:
          model.reset_classifier()

      with amp.autocast('cuda'):  # NEW
          logits = model(
              x_shot,
              x_query,
              y_shot,
              inner_args,
              meta_train=False,
              transport_mode=transport_mode,
              low_rank_rank=low_rank_rank,
              low_rank_init_scale=low_rank_init_scale,
              use_shift_aware_residual_gate=use_shift_aware_residual_gate,
              residual_gate_eps=residual_gate_eps,
              shift_gate_detach_context=shift_gate_detach_context
          )
          logits = logits.view(-1, config['test']['n_way'])
          labels = y_query.view(-1)

          pred = torch.argmax(logits, dim=1)
          acc = utils.compute_acc(pred, labels)
      # logits = model(x_shot, x_query, y_shot, inner_args, meta_train=False)
      # logits = logits.view(-1, config['test']['n_way'])
      # labels = y_query.view(-1)
      #
      # pred = torch.argmax(logits, dim=1)
      # acc = utils.compute_acc(pred, labels)
      aves_va.update(acc, 1)
      va_lst.append(acc)

    print('test epoch {}: acc={:.2f} +- {:.2f} (%)'.format(
      epoch, aves_va.item() * 100, 
      utils.mean_confidence_interval(va_lst) * 100))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--config', 
                      help='configuration file')
  parser.add_argument('--gpu', 
                      help='gpu device number', 
                      type=str, default='0')
  parser.add_argument('--efficient', 
                      help='if True, enables gradient checkpointing',
                      action='store_true')
  args = parser.parse_args()
  config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
  
  if len(args.gpu.split(',')) > 1:
    config['_parallel'] = True
    config['_gpu'] = args.gpu

  utils.set_gpu(args.gpu)
  # ✅ GPU varsa kullan, yoksa CPU'da devam et
  if torch.cuda.is_available() and args.gpu != '-1':
      utils.set_gpu(args.gpu)
  else:
      print("⚠️ CUDA not available — running on CPU mode.")
  main(config)
