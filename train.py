import argparse
import os
import random
from collections import OrderedDict

import yaml
import torch
#from torch.cuda.amp import autocast, GradScaler
from torch import amp
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torchviz import make_dot
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
  ckpt_name = args.name
  if ckpt_name is None:
    ckpt_name = config['encoder']
    ckpt_name += '_' + config['dataset'].replace('meta-', '')
    ckpt_name += '_{}_way_{}_shot'.format(
      config['train']['n_way'], config['train']['n_shot'])
  if args.tag is not None:
    ckpt_name += '_' + args.tag

  ckpt_path = os.path.join('./save', ckpt_name)
  utils.ensure_path(ckpt_path)
  utils.set_log_path(ckpt_path)
  writer = SummaryWriter(os.path.join(ckpt_path, 'tensorboard'))
  yaml.dump(config, open(os.path.join(ckpt_path, 'config.yaml'), 'w'))
  # Alignment log açık mı?
  log_alignment = config.get('log_alignment', False)

  # Pre-alignment loss kullanılacak mı?
  use_alignment_pre_loss = config.get('use_alignment_pre_loss', False)

  # Post-alignment loss kullanılacak mı?
  use_alignment_post_loss = config.get('use_alignment_post_loss', False)

  # Pre-alignment loss ağırlığı (eta)
  alignment_pre_weight = config.get('alignment_pre_weight', 0.0)

  # Post-alignment loss ağırlığı (eta)
  alignment_post_weight = config.get('alignment_post_weight', 0.0)
  # Modelden alignment çıktısı almamız gerekiyor mu?
  # Log açık olabilir, pre-loss açık olabilir, post-loss açık olabilir.
  need_alignment_outputs = (
      log_alignment or
      use_alignment_pre_loss or
      use_alignment_post_loss
  )

  transport_mode = config.get('transport_mode', 'none')
  low_rank_rank = int(config.get('low_rank_rank', 4))
  low_rank_init_scale = float(config.get('low_rank_init_scale', 1e-3))

  valid_transport_modes = ['none', 'scalar_gate', 'low_rank']
  if transport_mode not in valid_transport_modes:
      raise ValueError(f"Unknown transport_mode: {transport_mode}")
  ##### Dataset #####

  # meta-train
  train_set = datasets.make(config['dataset'], **config['train'])
  utils.log('meta-train set: {} (x{}), {}'.format(
    train_set[0][0].shape, len(train_set), train_set.n_classes))
  train_loader = DataLoader(
    train_set, config['train']['n_episode'],
    collate_fn=datasets.collate_fn, num_workers=8, pin_memory=True,prefetch_factor=4,persistent_workers=True)

  # meta-val
  eval_val = False
  if config.get('val'):
    eval_val = True
    val_set = datasets.make(config['dataset'], **config['val'])
    utils.log('meta-val set: {} (x{}), {}'.format(
      val_set[0][0].shape, len(val_set), val_set.n_classes))
    val_loader = DataLoader(
      val_set, config['val']['n_episode'],
      collate_fn=datasets.collate_fn, num_workers=4, pin_memory=True,prefetch_factor=4,persistent_workers=True)

  ##### Model and Optimizer #####

  inner_args = utils.config_inner_args(config.get('inner_args'))
  if config.get('load'):
    ckpt = torch.load(config['load'])
    config['encoder'] = ckpt['encoder']
    config['encoder_args'] = ckpt['encoder_args']
    config['classifier'] = ckpt['classifier']
    config['classifier_args'] = ckpt['classifier_args']
    transport_mode = ckpt.get('transport_mode', transport_mode)
    low_rank_rank = int(ckpt.get('low_rank_rank', low_rank_rank))
    low_rank_init_scale = float(ckpt.get('low_rank_init_scale', low_rank_init_scale))
    model = models.load(
        ckpt,
        load_clf=(not inner_args['reset_classifier']),
        transport_mode=transport_mode,
        low_rank_rank=low_rank_rank,
        low_rank_init_scale=low_rank_init_scale
    )
    optimizer, lr_scheduler = optimizers.load(ckpt, model.parameters())
    start_epoch = ckpt['training']['epoch'] + 1
    max_va = ckpt['training']['max_va']
  else:
    config['encoder_args'] = config.get('encoder_args') or dict()
    config['classifier_args'] = config.get('classifier_args') or dict()
    config['encoder_args']['bn_args']['n_episode'] = config['train']['n_episode']
    config['classifier_args']['n_way'] = config['train']['n_way']
    model = models.make(
        config['encoder'],
        config['encoder_args'],
        config['classifier'],
        config['classifier_args'],
        transport_mode=transport_mode,
        low_rank_rank=low_rank_rank,
        low_rank_init_scale=low_rank_init_scale
    )
    optimizer, lr_scheduler = optimizers.make(
      config['optimizer'], model.parameters(), **config['optimizer_args'])
    start_epoch = 1
    max_va = 0.

  if args.efficient:
    model.go_efficient()

  if config.get('_parallel'):
    model = nn.DataParallel(model)

  utils.log('num params: {}'.format(utils.compute_n_params(model)))
  timer_elapsed, timer_epoch = utils.Timer(), utils.Timer()

  scaler = amp.GradScaler('cuda')  # NEW (AMP scaler)
  ##### Training and evaluation #####

  # 'tl': meta-train loss
  # 'ta': meta-train accuracy
  # 'vl': meta-val loss
  # 'va': meta-val accuracy
  aves_keys = ['tl', 'ta', 'vl', 'va']
  if log_alignment:
      aves_keys += ['align_pre', 'align_post']
  trlog = dict()
  for k in aves_keys:
    trlog[k] = []

  for epoch in range(start_epoch, config['epoch'] + 1):
    timer_epoch.start()
    aves = {k: utils.AverageMeter() for k in aves_keys}

    # meta-train
    model.train()
    writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
    np.random.seed(epoch)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    did_viz = False
    for data in tqdm(train_loader, desc='meta-train', leave=False):
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

      optimizer.zero_grad(set_to_none=True)  # NEW (daha verimli)
      with amp.autocast('cuda'):  # NEW
          # Alignment ile ilgili herhangi bir şey açıksa
          # modeli metrics dönecek şekilde çağırıyoruz.
          if need_alignment_outputs:
              logits, metrics = model(
                  x_shot,
                  x_query,
                  y_shot,
                  inner_args,
                  meta_train=True,
                  y_query=y_query,
                  return_metrics=True,
                  use_alignment_pre_loss=use_alignment_pre_loss,
                  use_alignment_post_loss=use_alignment_post_loss,
                  alignment_pre_weight=alignment_pre_weight,
                  alignment_post_weight=alignment_post_weight,
                  transport_mode=transport_mode,
                  low_rank_rank=low_rank_rank,
                  low_rank_init_scale=low_rank_init_scale
              )
          else:
              logits = model(
                  x_shot,
                  x_query,
                  y_shot,
                  inner_args,
                  meta_train=True,
                  transport_mode=transport_mode,
                  low_rank_rank=low_rank_rank,
                  low_rank_init_scale=low_rank_init_scale
              )
              metrics = None
          if log_alignment and metrics is not None:
              if metrics['align_pre_mean'] is not None:
                  aves['align_pre'].update(metrics['align_pre_mean'], 1)
              if metrics['align_post_mean'] is not None:
                  aves['align_post'].update(metrics['align_post_mean'], 1)
          logits = logits.flatten(0, 1)
          labels = y_query.flatten()

          pred = torch.argmax(logits, dim=-1)
          acc = utils.compute_acc(pred, labels)
          loss = F.cross_entropy(logits, labels)
          # Toplam alignment cezasını burada ana loss'a ekliyoruz.
          # Başlangıçta sıfır kabul ediyoruz.
          align_loss_total = 0.0

          # Pre-alignment loss aktifse ekle
          if metrics is not None and metrics['align_pre_loss_mean'] is not None:
              align_loss_total = align_loss_total + metrics['align_pre_loss_mean']

          # Post-alignment loss aktifse ekle
          if metrics is not None and metrics['align_post_loss_mean'] is not None:
              align_loss_total = align_loss_total + metrics['align_post_loss_mean']

          # Final outer loss = query CE loss + alignment cezası
          loss = loss + align_loss_total

      # if not did_viz:
      #     from torchviz import make_dot
      #     # AMP bloğunun DIŞINDA çağır
      #     dot = make_dot(loss, params=dict(model.named_parameters()))
      #     dot.save("maml_graph.dot")
      #     did_viz = True

      ##AGAG Burada theta'yı güncelliyoruz.
      ##AGAG backward ile varolan önceki güncellemeleri de hesaba katarak theta güncellenir.
      scaler.scale(loss).backward()  # NEW
      scaler.unscale_(optimizer)  # NEW (clip öncesi gerekli)
      for param in optimizer.param_groups[0]['params']:
          if param.grad is not None:
              nn.utils.clip_grad_value_(param, 10)
      scaler.step(optimizer)  # NEW
      scaler.update()  # NEW


      aves['tl'].update(loss.item(), 1)
      aves['ta'].update(acc, 1)


    # meta-val
    if eval_val:
      model.eval()
      np.random.seed(0)

      for data in tqdm(val_loader, desc='meta-val', leave=False):
        x_shot, x_query, y_shot, y_query = data
        x_shot, y_shot = x_shot.cuda(), y_shot.cuda()
        x_query, y_query = x_query.cuda(), y_query.cuda()

        if inner_args['reset_classifier']:
          if config.get('_parallel'):
            model.module.reset_classifier()
          else:
            model.reset_classifier()

        with torch.no_grad(), amp.autocast('cuda'):  # NEW (val’de de AMP aç)
            logits = model(x_shot, x_query, y_shot, inner_args, meta_train=False, transport_mode=transport_mode,
    low_rank_rank=low_rank_rank,
    low_rank_init_scale=low_rank_init_scale)
            logits = logits.flatten(0, 1)
            labels = y_query.flatten()

            pred = torch.argmax(logits, dim=-1)
            acc = utils.compute_acc(pred, labels)
            loss = F.cross_entropy(logits, labels)
        aves['vl'].update(loss.item(), 1)
        aves['va'].update(acc, 1)

    if lr_scheduler is not None:
      lr_scheduler.step()

    for k, avg in aves.items():
      aves[k] = avg.item()
      trlog[k].append(aves[k])


    t_epoch = utils.time_str(timer_epoch.end())
    t_elapsed = utils.time_str(timer_elapsed.end())
    t_estimate = utils.time_str(timer_elapsed.end() /
      (epoch - start_epoch + 1) * (config['epoch'] - start_epoch + 1))

    # formats output
    log_str = 'epoch {}, meta-train {:.4f}|{:.4f}'.format(
      str(epoch), aves['tl'], aves['ta'])
    writer.add_scalars('loss', {'meta-train': aves['tl']}, epoch)
    writer.add_scalars('acc', {'meta-train': aves['ta']}, epoch)
    if log_alignment:
        writer.add_scalar('alignment/align_pre', aves['align_pre'], epoch)
        writer.add_scalar('alignment/align_post', aves['align_post'], epoch)
    if eval_val:
      if log_alignment:
          log_str += ', meta-val {:.4f}|{:.4f}, align {:.4f}|{:.4f}'.format(aves['vl'], aves['va'], aves['align_pre'], aves['align_post'])
      else:
          log_str += ', meta-val {:.4f}|{:.4f}'.format(aves['vl'], aves['va'])
      writer.add_scalars('loss', {'meta-val': aves['vl']}, epoch)
      writer.add_scalars('acc', {'meta-val': aves['va']}, epoch)

    log_str += ', {} {}/{}'.format(t_epoch, t_elapsed, t_estimate)
    utils.log(log_str)

    # saves model and meta-data
    if config.get('_parallel'):
      model_ = model.module
    else:
      model_ = model

    training = {
      'epoch': epoch,
      'max_va': max(max_va, aves['va']),

      'optimizer': config['optimizer'],
      'optimizer_args': config['optimizer_args'],
      'optimizer_state_dict': optimizer.state_dict(),
      'lr_scheduler_state_dict': lr_scheduler.state_dict()
        if lr_scheduler is not None else None,
    }
    ckpt = {
      'file': __file__,
      'config': config,

      'encoder': config['encoder'],
      'encoder_args': config['encoder_args'],
      'encoder_state_dict': model_.encoder.state_dict(),

      'classifier': config['classifier'],
      'classifier_args': config['classifier_args'],
      'classifier_state_dict': model_.classifier.state_dict(),

      'transport_mode': transport_mode,
      'low_rank_rank': low_rank_rank,
      'low_rank_init_scale': low_rank_init_scale,
      'scalar_transport_logits_state_dict': model_.scalar_transport_logits.state_dict(),
      'low_rank_U_state_dict': model_.low_rank_U.state_dict(),
      'low_rank_V_state_dict': model_.low_rank_V.state_dict(),

      'training': training,

    }

    # 'epoch-last.pth': saved at the latest epoch
    # 'max-va.pth': saved when validation accuracy is at its maximum
    torch.save(ckpt, os.path.join(ckpt_path, 'epoch-last.pth'))
    torch.save(trlog, os.path.join(ckpt_path, 'trlog.pth'))

    if aves['va'] > max_va:
      max_va = aves['va']
      torch.save(ckpt, os.path.join(ckpt_path, 'max-va.pth'))

    writer.flush()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--config', 
                      help='configuration file')
  parser.add_argument('--name', 
                      help='model name', 
                      type=str, default=None)
  parser.add_argument('--tag', 
                      help='auxiliary information', 
                      type=str, default=None)
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

    # ✅ GPU varsa kullan, yoksa CPU'da devam et
    if torch.cuda.is_available() and args.gpu != '-1':
        utils.set_gpu(args.gpu)
    else:
        print("⚠️ CUDA not available — running on CPU mode.")
  main(config)