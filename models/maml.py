from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.utils.checkpoint as cp

from . import encoders
from . import classifiers
from .modules import get_child_dict, Module, BatchNorm2d


def make(enc_name, enc_args, clf_name, clf_args,transport_mode='none',
    low_rank_rank=4,
    low_rank_init_scale=1e-3):
  """
  Initializes a random meta model.

  Args:
    enc_name (str): name of the encoder (e.g., 'resnet12').
    enc_args (dict): arguments for the encoder.
    clf_name (str): name of the classifier (e.g., 'meta-nn').
    clf_args (dict): arguments for the classifier.

  Returns:
    model (MAML): a meta classifier with a random encoder.
  """
  enc = encoders.make(enc_name, **enc_args)
  clf_args['in_dim'] = enc.get_out_dim()
  clf = classifiers.make(clf_name, **clf_args)
  model = MAML(
      enc, clf,
      transport_mode=transport_mode,
      low_rank_rank=low_rank_rank,
      low_rank_init_scale=low_rank_init_scale
  )
  return model


def load(ckpt, load_clf=False, clf_name=None, clf_args=None,transport_mode=None,
    low_rank_rank=None,
    low_rank_init_scale=None):
  """
  Initializes a meta model with a pre-trained encoder.

  Args:
    ckpt (dict): a checkpoint from which a pre-trained encoder is restored.
    load_clf (bool, optional): if True, loads a pre-trained classifier.
      Default: False (in which case the classifier is randomly initialized)
    clf_name (str, optional): name of the classifier (e.g., 'meta-nn')
    clf_args (dict, optional): arguments for the classifier.
    (The last two arguments are ignored if load_clf=True.)

  Returns:
    model (MAML): a meta model with a pre-trained encoder.
  """
  if transport_mode is None:
    transport_mode = ckpt.get('transport_mode', 'none')
  if low_rank_rank is None:
    low_rank_rank = int(ckpt.get('low_rank_rank', 4))
  if low_rank_init_scale is None:
    low_rank_init_scale = float(ckpt.get('low_rank_init_scale', 1e-3))

  enc = encoders.load(ckpt)
  if load_clf:
    clf = classifiers.load(ckpt)
  else:
    if clf_name is None and clf_args is None:
      clf = classifiers.make(ckpt['classifier'], **ckpt['classifier_args'])
    else:
      clf_args['in_dim'] = enc.get_out_dim()
      clf = classifiers.make(clf_name, **clf_args)
  model = MAML(
      enc, clf,
      transport_mode=transport_mode,
      low_rank_rank=low_rank_rank,
      low_rank_init_scale=low_rank_init_scale
  )
  if 'scalar_transport_logits_state_dict' in ckpt:
      model.scalar_transport_logits.load_state_dict(
          ckpt['scalar_transport_logits_state_dict']
      )

  if 'low_rank_U_state_dict' in ckpt:
      model.low_rank_U.load_state_dict(
          ckpt['low_rank_U_state_dict']
      )

  if 'low_rank_V_state_dict' in ckpt:
      model.low_rank_V.load_state_dict(
          ckpt['low_rank_V_state_dict']
      )
  return model


class MAML(Module):
  def __init__(self, encoder, classifier,transport_mode='none',
    low_rank_rank=4,
    low_rank_init_scale=1e-3):
    super(MAML, self).__init__()
    self.encoder = encoder
    self.classifier = classifier

    self.transport_mode = transport_mode
    self.low_rank_rank = int(low_rank_rank)
    self.low_rank_init_scale = float(low_rank_init_scale)

    self.scalar_transport_logits = nn.ParameterDict()
    self.low_rank_U = nn.ParameterDict()
    self.low_rank_V = nn.ParameterDict()
    self._build_transport_params()

  def reset_classifier(self):
    self.classifier.reset_parameters()

  def _build_transport_params(self):
      for prefix, module in [('encoder', self.encoder), ('classifier', self.classifier)]:
          for name, param in module.named_parameters():
              key = self._transport_key(f'{prefix}.{name}')

              self.scalar_transport_logits[key] = nn.Parameter(
                  torch.tensor(4.0, device=param.device, dtype=param.dtype)
              )

              n = param.numel()
              r = min(self.low_rank_rank, n)

              self.low_rank_U[key] = nn.Parameter(
                  torch.randn(n, r, device=param.device, dtype=param.dtype) * self.low_rank_init_scale
              )
              self.low_rank_V[key] = nn.Parameter(
                  torch.randn(n, r, device=param.device, dtype=param.dtype) * self.low_rank_init_scale
              )

  def _transport_key(self, name):
      return name.replace('.', '__')

  def _apply_transport_to_grad(
          self,
          name,
          grad,
          transport_mode='none',
  ):
      if transport_mode == 'none':
          return grad

      key = self._transport_key(name)

      if transport_mode == 'scalar_gate':
          gate = torch.sigmoid(self.scalar_transport_logits[key]).to(dtype=grad.dtype)
          return gate * grad

      elif transport_mode == 'low_rank':
          g_flat = grad.reshape(-1).float()

          U = self.low_rank_U[key].float()
          V = self.low_rank_V[key].float()

          correction = U @ (V.t() @ g_flat)
          transported_flat = g_flat + 0.01 * correction
          transported_grad = transported_flat.view_as(grad).to(dtype=grad.dtype)
          return transported_grad

      else:
          raise ValueError(f"Unknown transport_mode: {transport_mode}")

  def _inner_forward(self, x, params, episode):
    """ Forward pass for the inner loop. """
    feat = self.encoder(x, get_child_dict(params, 'encoder'), episode)
    logits = self.classifier(feat, get_child_dict(params, 'classifier'))
    return logits

  def _inner_iter(
          self,
          x,
          y,
          params,
          mom_buffer,
          episode,
          inner_args,
          detach,
          transport_mode='none',
          low_rank_rank=4,
          low_rank_init_scale=1e-3
  ):
    """ 
    Performs one inner-loop iteration of MAML including the forward and 
    backward passes and the parameter update.

    Args:
      x (float tensor, [n_way * n_shot, C, H, W]): per-episode support set.
      y (int tensor, [n_way * n_shot]): per-episode support set labels.
      params (dict): the model parameters BEFORE the update.
      mom_buffer (dict): the momentum buffer BEFORE the update.
      episode (int): the current episode index.
      inner_args (dict): inner-loop optimization hyperparameters.
      detach (bool): if True, detachs the graph for the current iteration.

    Returns:
      updated_params (dict): the model parameters AFTER the update.
      mom_buffer (dict): the momentum buffer AFTER the update.
    """
    with torch.enable_grad():
      # forward pass
      # AGAG Loss'ların bulunduğu yer.
      logits = self._inner_forward(x, params, episode)
      loss = F.cross_entropy(logits, y)
      # backward pass
      grads = autograd.grad(loss, params.values(), 
        create_graph=(not detach and not inner_args['first_order']),
        only_inputs=True, allow_unused=True)
      # parameter update
      updated_params = OrderedDict()
      for (name, param), grad in zip(params.items(), grads):
        if grad is None:
          updated_param = param
        else:
          if inner_args['weight_decay'] > 0:
            grad = grad + inner_args['weight_decay'] * param
          if inner_args['momentum'] > 0:
            grad = grad + inner_args['momentum'] * mom_buffer[name]
            mom_buffer[name] = grad
          if 'encoder' in name:
            lr = inner_args['encoder_lr']
          elif 'classifier' in name:
            lr = inner_args['classifier_lr']
          else:
            raise ValueError('invalid parameter name')
          transported_grad = self._apply_transport_to_grad(
              name=name,
              grad=grad,
              transport_mode=transport_mode,
          )

          updated_param = param - lr * transported_grad #AGAG θi′=θi−α∇θiLsupport(θ)
          #updated_param = param - lr * grad  #AGAG θi′=θi−α∇θiLsupport(θ)
        if detach:
          updated_param = updated_param.detach().requires_grad_(True)
        updated_params[name] = updated_param

    return updated_params, mom_buffer

  def _adapt(
          self,
          x,
          y,
          params,
          episode,
          inner_args,
          meta_train,
          transport_mode='none',
          low_rank_rank=4,
          low_rank_init_scale=1e-3
  ):
    """
    Performs inner-loop adaptation in MAML.

    Args:
      x (float tensor, [n_way * n_shot, C, H, W]): per-episode support set.
        (T: transforms, C: channels, H: height, W: width)
      y (int tensor, [n_way * n_shot]): per-episode support set labels.
      params (dict): a dictionary of parameters at meta-initialization.
      episode (int): the current episode index.
      inner_args (dict): inner-loop optimization hyperparameters.
      meta_train (bool): if True, the model is in meta-training.
      
    Returns:
      params (dict): model paramters AFTER inner-loop adaptation.
    """
    assert x.dim() == 4 and y.dim() == 1
    assert x.size(0) == y.size(0)  #AGAG ilgili epizotta örnek sayısıyla etiket sayısı eşit mi kontrolü

    # Initializes a dictionary of momentum buffer for gradient descent in the 
    # inner loop. It has the same set of keys as the parameter dictionary.
    mom_buffer = OrderedDict()
    if inner_args['momentum'] > 0:  #AGAG Klasik gradient descent yerine momentum gradient descent. Normal MAML'da yok, ufak bir ekleme. Çok önemli değil
      for name, param in params.items():
        mom_buffer[name] = torch.zeros_like(param)
    params_keys = tuple(params.keys())
    mom_buffer_keys = tuple(mom_buffer.keys())

    for m in self.modules():
      if isinstance(m, BatchNorm2d) and m.is_episodic():
        m.reset_episodic_running_stats(episode)

    #AGAG aşağıdaki self.efficient true ise daha az ram kullanmak amacıyla checkpoint mantığının çalıştırılması için kullanılan bir şey. Normalde kullanılmıyor.
    def _inner_iter_cp(episode, *state):
      """ 
      Performs one inner-loop iteration when checkpointing is enabled. 
      The code is executed twice:
        - 1st time with torch.no_grad() for creating checkpoints.
        - 2nd time with torch.enable_grad() for computing gradients.
      """
      params = OrderedDict(zip(params_keys, state[:len(params_keys)]))
      mom_buffer = OrderedDict(
        zip(mom_buffer_keys, state[-len(mom_buffer_keys):]))

      detach = not torch.is_grad_enabled()  # detach graph in the first pass
      self.is_first_pass(detach)
      params, mom_buffer = self._inner_iter(
          x,
          y,
          params,
          mom_buffer,
          int(episode),
          inner_args,
          detach,
          transport_mode=transport_mode,
          low_rank_rank=low_rank_rank,
          low_rank_init_scale=low_rank_init_scale
      )
      state = tuple(t if t.requires_grad else t.clone().requires_grad_(True)
        for t in tuple(params.values()) + tuple(mom_buffer.values()))
      return state

    for step in range(inner_args['n_step']): #AGAG buradaki step, bir taskteki support set üzerindeki verileri kaç kere işleyip kaç kere gradient'i güncelleyeceğimizi belirler.
      if self.efficient:  # checkpointing
        state = tuple(params.values()) + tuple(mom_buffer.values())
        state = cp.checkpoint(_inner_iter_cp, torch.as_tensor(episode), *state)
        params = OrderedDict(zip(params_keys, state[:len(params_keys)]))
        mom_buffer = OrderedDict(
          zip(mom_buffer_keys, state[-len(mom_buffer_keys):]))
      else:
          params, mom_buffer = self._inner_iter(
              x,
              y,
              params,
              mom_buffer,
              episode,
              inner_args,
              not meta_train,
              transport_mode=transport_mode,
              low_rank_rank=low_rank_rank,
              low_rank_init_scale=low_rank_init_scale
          )
        
    return params

  def forward(
          self,
          x_shot,
          x_query,
          y_shot,
          inner_args,
          meta_train,
          y_query=None,
          return_metrics=False,
          use_alignment_pre_loss=False,  # pre-alignment loss aktif mi?
          use_alignment_post_loss=False,  # post-alignment loss aktif mi?
          alignment_pre_weight=0.0,  # pre-alignment loss katsayısı (eta)
          alignment_post_weight=0.0,  # post-alignment loss katsayısı (eta)
          transport_mode=None,
          low_rank_rank=None,
          low_rank_init_scale=None,
          use_rank1_jacobian_proxy=False,
          rank1_jacobian_fd_eps=1e-3,
          rank1_jacobian_normalize=True,
          use_jacobian_proxy=False,
          jacobian_proxy_rank=1,
          jacobian_proxy_fd_eps=1e-3,
          jacobian_proxy_normalize=True,
          jacobian_proxy_power_iters=1
  ):
    """
    Args:
      x_shot (float tensor, [n_episode, n_way * n_shot, C, H, W]): support sets.
      x_query (float tensor, [n_episode, n_way * n_query, C, H, W]): query sets.
        (T: transforms, C: channels, H: height, W: width)
      y_shot (int tensor, [n_episode, n_way * n_shot]): support set labels.
      inner_args (dict, optional): inner-loop hyperparameters.
      meta_train (bool): if True, the model is in meta-training.
      
    Returns:
      logits (float tensor, [n_episode, n_way * n_shot, n_way]): predicted logits.
    """
    assert self.encoder is not None
    assert self.classifier is not None
    assert x_shot.dim() == 5 and x_query.dim() == 5
    assert x_shot.size(0) == x_query.size(0)

    if transport_mode is None:
      transport_mode = self.transport_mode
    if low_rank_rank is None:
      low_rank_rank = self.low_rank_rank
    if low_rank_init_scale is None:
      low_rank_init_scale = self.low_rank_init_scale

    if (not use_jacobian_proxy) and use_rank1_jacobian_proxy:
      use_jacobian_proxy = True
      jacobian_proxy_rank = 1
      jacobian_proxy_fd_eps = rank1_jacobian_fd_eps
      jacobian_proxy_normalize = rank1_jacobian_normalize

    jacobian_proxy_rank = max(1, int(jacobian_proxy_rank))
    jacobian_proxy_power_iters = max(0, int(jacobian_proxy_power_iters))
    enable_transport_proxy_grads = (
        meta_train and use_jacobian_proxy and (transport_mode != 'none')
    )

    # Alignment metrik/log hesaplaması yapılacak mı?
    # Bunun için hem return_metrics açık olmalı hem de query etiketleri gelmiş olmalı.
    do_alignment_log = return_metrics and (y_query is not None)

    # Pre-alignment loss gerçekten train loss'una eklenecek mi?
    # Sadece meta-train sırasında anlamlı.
    do_alignment_pre_loss = use_alignment_pre_loss and meta_train and (y_query is not None)

    # Post-alignment loss gerçekten train loss'una eklenecek mi?
    # Sadece meta-train sırasında anlamlı.
    do_alignment_post_loss = use_alignment_post_loss and meta_train and (y_query is not None)

    # Epoch boyunca episode bazlı pre-alignment loss değerlerini tutacağız.
    align_pre_loss_list = []

    # Epoch boyunca episode bazlı post-alignment loss değerlerini tutacağız.
    align_post_loss_list = []

    align_pre_list = []
    align_post_list = []

    jacobian_proxy_loss_list = []
    # a dictionary of parameters that will be updated in the inner loop
    #AGAG Gereksiz yani gradyanı hesaplanmayacak parametrelerin çıkartılması
    params = OrderedDict(self.named_parameters())
    for name in list(params.keys()):
      if not params[name].requires_grad or \
              any(s in name for s in inner_args['frozen'] + [
                  'temp',
                  'scalar_transport_logits',
                  'low_rank_U',
                  'low_rank_V'
              ]):
        params.pop(name)

    logits = []
    for ep in range(x_shot.size(0)): #AGAG x_shot.size(0) -> n_episode yani o batch içerisinde kaç tane task olduğu.
      # inner-loop training
      ##AGAG train moduna alınması dropout, batch normalization gibi şeylerin.
      self.train()
      if not meta_train:
        for m in self.modules():
          if isinstance(m, BatchNorm2d) and not m.is_episodic():
            m.eval()

      g_sup_pre = None
      jacobian_proxy_state = None
      # PRE alignment bloğuna gerçekten ihtiyaç var mı?
      # - log alacaksak lazım
      # - pre-loss kullanacaksak lazım
      # - post-loss kullanacaksak support gradient'ini daha sonra da kullanacağız
      need_pre_block = do_alignment_log or do_alignment_pre_loss or do_alignment_post_loss
      if need_pre_block:
          with torch.enable_grad():
              # Support ve query loss'larını başlangıç parametresi (theta) üzerinde hesaplıyoruz.
              logits_sup_pre = self._inner_forward(x_shot[ep], params, ep)
              loss_sup_pre = F.cross_entropy(logits_sup_pre, y_shot[ep])

              logits_qry_pre = self._inner_forward(x_query[ep], params, ep)
              loss_qry_pre = F.cross_entropy(logits_qry_pre, y_query[ep])

              param_list = list(params.values())

              # Support gradient'i:
              # Eğer pre-loss veya post-loss kullanacaksak graph korunmalı.
              keep_sup_grad_graph = do_alignment_pre_loss or do_alignment_post_loss
              g_sup_pre = self._get_grads_from_loss(
                  loss_sup_pre,
                  param_list,
                  retain_graph=keep_sup_grad_graph,
                  create_graph=keep_sup_grad_graph,
                  detach_grads=(not keep_sup_grad_graph)
              )

              # Query-pre gradient'i:
              # Sadece pre-loss aktifse graph'lı tutmamız gerekir.
              g_qry_pre = self._get_grads_from_loss(
                  loss_qry_pre,
                  param_list,
                  retain_graph=do_alignment_pre_loss,
                  create_graph=do_alignment_pre_loss,
                  detach_grads=(not do_alignment_pre_loss)
              )

              # PRE alignment cosine değeri
              align_pre = self._cosine_between_grad_lists(g_sup_pre, g_qry_pre)

              # Sadece log açıksa metriği kaydet
              if do_alignment_log:
                  align_pre_list.append(align_pre.detach().item())

              # Sadece pre-loss açıksa ceza terimini oluştur
              # Omega_align_pre = eta * (1 - cos)
              if do_alignment_pre_loss:
                  align_pre_loss = alignment_pre_weight * (1.0 - align_pre)
                  align_pre_loss_list.append(align_pre_loss)
      if use_jacobian_proxy and meta_train:
          with torch.enable_grad():
              logits_sup_proxy = self._inner_forward(x_shot[ep], params, ep)
              loss_sup_proxy = F.cross_entropy(logits_sup_proxy, y_shot[ep])

              raw_sup_grads = autograd.grad(
                  loss_sup_proxy,
                  params.values(),
                  create_graph=False,
                  retain_graph=False,
                  allow_unused=True
              )

              g_sup_proxy = self._apply_inner_update_rule_to_grads(
                  params,
                  raw_sup_grads,
                  transport_mode=transport_mode,
                  weight_decay=inner_args['weight_decay']
              )
              if not enable_transport_proxy_grads:
                  g_sup_proxy = [g.detach() for g in g_sup_proxy]

                      # Rank-1 Jacobian tarafında, inner stepte gerçekten kullanılan yönü esas alıyoruz.

              jacobian_proxy_state = self._prepare_jacobian_proxy_state(
                  x_shot[ep],
                  y_shot[ep],
                  params,
                  ep,
                  g_sup_proxy,
                  inner_args,
                  jacobian_proxy_rank=jacobian_proxy_rank,
                  fd_eps=jacobian_proxy_fd_eps,
                  normalize=jacobian_proxy_normalize,
                  power_iters=jacobian_proxy_power_iters,
                  track_gradients=enable_transport_proxy_grads,
                  transport_mode=transport_mode,
                  weight_decay=inner_args['weight_decay']
              )
      updated_params = self._adapt(  #AGAG Modelin inner loop'u -> θ′=θ−α∇θLsupport
        x_shot[ep], y_shot[ep], params, ep, inner_args, meta_train, transport_mode=transport_mode,
    low_rank_rank=low_rank_rank,
    low_rank_init_scale=low_rank_init_scale)
      # inner-loop validation
      # Query tarafında gradient gerekip gerekmediğini belirliyoruz.
      # - log alacaksak lazım
      # - post-loss kullanacaksak da lazım
      need_post_block = do_alignment_log or do_alignment_post_loss

      # Eğer post alignment hesaplanacaksa burada mutlaka grad açık olmalı.
      # Aksi halde query gradient'ini çıkaramayız.
      grad_ctx = torch.enable_grad() if need_post_block else torch.set_grad_enabled(meta_train)
      # inner-loop validation / query evaluation
      with grad_ctx:
          self.eval()
          logits_ep = self._inner_forward(x_query[ep], updated_params, ep)
          if use_jacobian_proxy and meta_train:
              loss_qry_proxy = F.cross_entropy(logits_ep, y_query[ep])

              g_qry_proxy = self._get_grads_from_loss(
                  loss_qry_proxy,
                  list(updated_params.values()),
                  retain_graph=(need_post_block or enable_transport_proxy_grads),
                  create_graph=enable_transport_proxy_grads,
                  detach_grads=(not enable_transport_proxy_grads)
              )

              proxy_flat = self._apply_prepared_jacobian_proxy(
                  g_qry_proxy,
                  jacobian_proxy_state
              )

              proxy_loss_ep = self._proxy_grads_to_surrogate_loss(
                  proxy_flat,
                  params
              )
              jacobian_proxy_loss_list.append(proxy_loss_ep)

          # POST alignment bloğuna ihtiyaç var mı?
          if need_post_block:
              # Query loss'unu support update sonrası parametreler (theta') üzerinde hesaplıyoruz.
              loss_qry_post = F.cross_entropy(logits_ep, y_query[ep])

              # Post-loss aktifse gradient graph'ını korumamız gerekir.
              g_qry_post = self._get_grads_from_loss(
                  loss_qry_post,
                  list(updated_params.values()),
                  retain_graph=meta_train,
                  create_graph=do_alignment_post_loss,
                  detach_grads=(not do_alignment_post_loss)
              )

              # POST alignment cosine değeri:
              # support'taki başlangıç gradient'i ile update sonrası query gradient'ini karşılaştırıyoruz.
              align_post = self._cosine_between_grad_lists(g_sup_pre, g_qry_post)

              # Sadece log açıksa metriği kaydet
              if do_alignment_log:
                  align_post_list.append(align_post.detach().item())

              # Sadece post-loss açıksa ceza terimini oluştur
              # Omega_align_post = eta * (1 - cos)
              if do_alignment_post_loss:
                  align_post_loss = alignment_post_weight * (1.0 - align_post)
                  align_post_loss_list.append(align_post_loss)
      logits.append(logits_ep)

    self.train(meta_train)
    logits = torch.stack(logits)
    proxy_loss = None
    if use_jacobian_proxy and meta_train:
        proxy_loss = torch.stack(jacobian_proxy_loss_list).mean()
    if return_metrics:
        metrics = {
            'align_pre_mean': sum(align_pre_list) / len(align_pre_list) if len(align_pre_list) > 0 else None,
            'align_post_mean': sum(align_post_list) / len(align_post_list) if len(align_post_list) > 0 else None,
            'align_pre_loss_mean': torch.stack(align_pre_loss_list).mean() if len(align_pre_loss_list) > 0 else None,
            'align_post_loss_mean': torch.stack(align_post_loss_list).mean() if len(align_post_loss_list) > 0 else None,
        }
        if use_jacobian_proxy and meta_train:
            return logits, metrics, proxy_loss
        return logits, metrics

    if use_jacobian_proxy and meta_train:
        return logits, proxy_loss

    return logits

  #AGAG ALIGNMENT LOGLARI İÇİN EKLENEN FONKSİYONLAR
  def _get_grads_from_loss(
          self,
          loss,
          params,
          retain_graph=False,
          create_graph=False,
          detach_grads=True
  ):
      """
      Bir loss'tan gradient listesi çıkarır.

      Args:
        loss: türev alınacak loss değeri.
        params: gradient'i alınacak parametre listesi.
        retain_graph: mevcut graph sonradan tekrar kullanılacak mı?
        create_graph: gradient'in de türevini alabilmek için graph oluşturulsun mu?
        detach_grads: sadece log/metric için kullanıyorsak gradient'i graph'tan kopar.
                      Loss olarak kullanacaksak False olmalı.
      """
      grads = autograd.grad(
          loss,
          params,
          retain_graph=retain_graph,
          create_graph=create_graph,
          allow_unused=True
      )

      out = []
      for p, g in zip(params, grads):
          # Bazı parametreler için grad None gelebilir.
          # Bu durumda aynı boyutta sıfır tensor koyuyoruz.
          if g is None:
              z = torch.zeros_like(p)
              out.append(z.detach() if detach_grads else z)
          else:
              out.append(g.detach() if detach_grads else g)
      return out

  def _flatten_grads(self, grads):
      return torch.cat([g.reshape(-1) for g in grads])

  def _cosine_between_grad_lists(self, grads1, grads2, eps=1e-12):
      v1 = self._flatten_grads(grads1)
      v2 = self._flatten_grads(grads2)
      return F.cosine_similarity(v1, v2, dim=0, eps=eps)

  def _apply_inner_update_rule_to_grads(
          self,
          params,
          grads,
          transport_mode='none',
          weight_decay=0.0
  ):
      processed_grads = []
      for (name, param), grad in zip(params.items(), grads):
          if grad is None:
              grad = torch.zeros_like(param)
          if weight_decay > 0:
              grad = grad + weight_decay * param
          grad = self._apply_transport_to_grad(
              name=name,
              grad=grad,
              transport_mode=transport_mode,
          )
          processed_grads.append(grad)
      return processed_grads

  def _apply_rank1_jacobian_proxy(
          self,
          qry_grads,
          sup_grads,
          lambda_i,
          alpha,
          normalize=True,
          eps=1e-12
  ):
      """
      Rank-1 Jacobian vekilini uygular.

      v_proxy = v - alpha * lambda_i * (g^T v) * g
      burada:
        g = support gradient yönü
        v = query tarafındaki gradient
      """
      g = self._flatten_grads(sup_grads).float()
      v = self._flatten_grads(qry_grads).float()

      if normalize:
          g = g / (g.norm() + eps)

      coeff = torch.dot(g, v)
      if torch.is_tensor(alpha):
          alpha_vec = alpha.to(device=v.device, dtype=v.dtype)
      else:
          alpha_vec = torch.tensor(alpha, device=v.device, dtype=v.dtype)

      v_proxy = v - alpha_vec * lambda_i * coeff * g
      return v_proxy

  def _estimate_rank1_lambda_fd(
          self,
          x,
          y,
          params,
          episode,
          sup_grads,
          fd_eps=1e-3,
          normalize=True,
          track_gradients=False,
          transport_mode='none',
          weight_decay=0.0,
          eps=1e-12
  ):
      """
      Support gradient doğrultusundaki eğriliği sonlu farkla yaklaşık hesaplar.

      lambda_i ≈ g^T H g   (normalize=True ise yön normalize edilerek)
      """
      g = self._flatten_grads(sup_grads).float()

      if normalize:
          direction = g / (g.norm() + eps)
      else:
          direction = g

      hv_approx = self._estimate_fd_hvp(
          x,
          y,
          params,
          episode,
          direction,
          fd_eps=fd_eps,
          track_gradients=track_gradients,
          transport_mode=transport_mode,
          weight_decay=weight_decay
      )

      lambda_i = torch.dot(direction, hv_approx)
      return lambda_i

  def _build_alpha_flat(self, params, inner_args):
      alpha_chunks = []
      for name, param in params.items():
          if 'encoder' in name:
              lr = float(inner_args['encoder_lr'])
          elif 'classifier' in name:
              lr = float(inner_args['classifier_lr'])
          else:
              raise ValueError('invalid parameter name')
          alpha_chunks.append(
              torch.full((param.numel(),), lr, device=param.device, dtype=torch.float32)
          )
      return torch.cat(alpha_chunks)

  def _orthonormalize_columns(self, matrix, target_rank=None):
      if matrix.dim() == 1:
          matrix = matrix.unsqueeze(1)
      matrix = matrix.float()
      q, _ = torch.linalg.qr(matrix, mode='reduced')
      if target_rank is None:
          return q
      return q[:, :min(int(target_rank), q.size(1))]

  def _estimate_fd_hvp(
          self,
          x,
          y,
          params,
          episode,
          direction,
          fd_eps=1e-3,
          track_gradients=False,
          transport_mode='none',
          weight_decay=0.0
  ):
      shifted_plus = OrderedDict()
      shifted_minus = OrderedDict()

      offset = 0
      for name, p in params.items():
          n = p.numel()
          d_part = direction[offset:offset + n].view_as(p).to(dtype=p.dtype, device=p.device)
          if track_gradients:
              shifted_plus[name] = p + fd_eps * d_part
              shifted_minus[name] = p - fd_eps * d_part
          else:
              shifted_plus[name] = (p + fd_eps * d_part).detach().requires_grad_(True)
              shifted_minus[name] = (p - fd_eps * d_part).detach().requires_grad_(True)
          offset += n

      logits_plus = self._inner_forward(x, shifted_plus, episode)
      loss_plus = F.cross_entropy(logits_plus, y)
      grads_plus = self._get_grads_from_loss(
          loss_plus,
          list(shifted_plus.values()),
          retain_graph=track_gradients,
          create_graph=track_gradients,
          detach_grads=(not track_gradients)
      )

      logits_minus = self._inner_forward(x, shifted_minus, episode)
      loss_minus = F.cross_entropy(logits_minus, y)
      grads_minus = self._get_grads_from_loss(
          loss_minus,
          list(shifted_minus.values()),
          retain_graph=track_gradients,
          create_graph=track_gradients,
          detach_grads=(not track_gradients)
      )

      grads_plus = self._apply_inner_update_rule_to_grads(
          shifted_plus,
          grads_plus,
          transport_mode=transport_mode,
          weight_decay=weight_decay
      )
      grads_minus = self._apply_inner_update_rule_to_grads(
          shifted_minus,
          grads_minus,
          transport_mode=transport_mode,
          weight_decay=weight_decay
      )

      return (
          self._flatten_grads(grads_plus).float() -
          self._flatten_grads(grads_minus).float()
      ) / (2.0 * fd_eps)

  def _estimate_fd_hvp_columns(
          self,
          x,
          y,
          params,
          episode,
          directions,
          fd_eps=1e-3,
          track_gradients=False,
          transport_mode='none',
          weight_decay=0.0
  ):
      hvp_columns = []
      for col_idx in range(directions.size(1)):
          hvp_columns.append(
              self._estimate_fd_hvp(
                  x,
                  y,
                  params,
                  episode,
                  directions[:, col_idx],
                  fd_eps=fd_eps,
                  track_gradients=track_gradients,
                  transport_mode=transport_mode,
                  weight_decay=weight_decay
              )
          )
      return torch.stack(hvp_columns, dim=1)

  def _estimate_rankr_subspace_fd(
          self,
          x,
          y,
          params,
          episode,
          sup_grads,
          rank=4,
          fd_eps=1e-3,
          normalize=True,
          power_iters=1,
          track_gradients=False,
          transport_mode='none',
          weight_decay=0.0,
          eps=1e-12
  ):
      seed_direction = self._flatten_grads(sup_grads).float()
      dim = seed_direction.numel()
      rank = min(max(1, int(rank)), dim)

      if normalize and seed_direction.norm() > eps:
          seed_direction = seed_direction / (seed_direction.norm() + eps)

      basis_init = seed_direction.unsqueeze(1)
      if rank > 1:
          random_cols = torch.randn(
              dim, rank - 1, device=seed_direction.device, dtype=seed_direction.dtype
          )
          basis_init = torch.cat([basis_init, random_cols], dim=1)

      device_type = seed_direction.device.type
      with torch.autocast(device_type=device_type, enabled=False):
          Q = self._orthonormalize_columns(basis_init, target_rank=rank)

          for _ in range(max(0, int(power_iters))):
              HQ = self._estimate_fd_hvp_columns(
                  x,
                  y,
                  params,
                  episode,
                  Q,
                  fd_eps=fd_eps,
                  track_gradients=track_gradients,
                  transport_mode=transport_mode,
                  weight_decay=weight_decay
              )
              Q = self._orthonormalize_columns(HQ, target_rank=rank)

          HQ = self._estimate_fd_hvp_columns(
              x,
              y,
              params,
              episode,
              Q,
              fd_eps=fd_eps,
              track_gradients=track_gradients,
              transport_mode=transport_mode,
              weight_decay=weight_decay
          )
          reduced_hessian = 0.5 * (Q.t() @ HQ + HQ.t() @ Q)
          reduced_hessian = reduced_hessian.float()

          eigvals, eigvecs = torch.linalg.eigh(reduced_hessian)
      order = torch.argsort(eigvals.abs(), descending=True)
      eigvals = eigvals[order[:rank]]
      eigvecs = eigvecs[:, order[:rank]]
      basis = Q @ eigvecs

      return {
          'basis': basis,
          'curvature': eigvals
      }

  def _prepare_jacobian_proxy_state(
          self,
          x,
          y,
          params,
          episode,
          sup_grads,
          inner_args,
          jacobian_proxy_rank=1,
          fd_eps=1e-3,
          normalize=True,
          power_iters=1,
          track_gradients=False,
          transport_mode='none',
          weight_decay=0.0
  ):
      rank = max(1, int(jacobian_proxy_rank))
      if rank == 1:
          alpha_rank1 = self._build_alpha_flat(params, inner_args)
          lambda_i = self._estimate_rank1_lambda_fd(
              x,
              y,
              params,
              episode,
              sup_grads,
              fd_eps=fd_eps,
              normalize=normalize,
              track_gradients=track_gradients,
              transport_mode=transport_mode,
              weight_decay=weight_decay
          )
          return {
              'kind': 'rank1',
              'sup_grads': sup_grads,
              'lambda_i': lambda_i,
              'alpha': alpha_rank1,
              'normalize': normalize
          }

      rankr_state = self._estimate_rankr_subspace_fd(
          x,
          y,
          params,
          episode,
          sup_grads,
          rank=rank,
          fd_eps=fd_eps,
          normalize=normalize,
          power_iters=power_iters,
          track_gradients=track_gradients,
          transport_mode=transport_mode,
          weight_decay=weight_decay
      )
      return {
          'kind': 'rankr',
          'basis': rankr_state['basis'],
          'curvature': rankr_state['curvature'],
          'alpha_flat': self._build_alpha_flat(params, inner_args)
      }

  def _apply_rankr_jacobian_proxy(
          self,
          qry_grads,
          basis,
          curvature,
          alpha_flat
  ):
      v = self._flatten_grads(qry_grads).float()
      coeff = basis.t() @ v
      transported_basis = alpha_flat.unsqueeze(1) * basis
      return v - transported_basis @ (curvature * coeff)

  def _apply_prepared_jacobian_proxy(self, qry_grads, proxy_state):
      if proxy_state is None:
          raise ValueError('proxy_state must be prepared before applying Jacobian proxy')

      if proxy_state['kind'] == 'rank1':
          return self._apply_rank1_jacobian_proxy(
              qry_grads,
              proxy_state['sup_grads'],
              lambda_i=proxy_state['lambda_i'],
              alpha=proxy_state['alpha'],
              normalize=proxy_state['normalize']
          )

      if proxy_state['kind'] == 'rankr':
          return self._apply_rankr_jacobian_proxy(
              qry_grads,
              proxy_state['basis'],
              proxy_state['curvature'],
              proxy_state['alpha_flat']
          )

      raise ValueError(f"Unknown proxy_state kind: {proxy_state['kind']}")

  def _proxy_grads_to_surrogate_loss(self, proxy_flat, params):
      """
      Elle elde edilmiş proxy gradient'i, backward alınabilir skaler loss'a çevirir.
      """
      loss = 0.0
      offset = 0

      for _, p in params.items():
          n = p.numel()
          g_part = proxy_flat[offset:offset + n].view_as(p).to(dtype=p.dtype, device=p.device)
          surrogate = (
              g_part * p.detach() +
              g_part.detach() * p -
              g_part.detach() * p.detach()
          )
          loss = loss + torch.sum(surrogate)
          offset += n

      return loss
