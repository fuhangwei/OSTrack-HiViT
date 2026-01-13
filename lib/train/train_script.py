import os
# loss function related
from lib.utils.box_ops import giou_loss
from torch.nn.functional import l1_loss
from torch.nn import BCEWithLogitsLoss
# train pipeline related
from lib.train.trainers import LTRTrainer
# distributed training related
from torch.nn.parallel import DistributedDataParallel as DDP
# some more advanced functions
from .base_functions import *
# network related
from lib.models.ostrack import build_ostrack
# forward propagation related
from lib.train.actors import OSTrackActor
# for import modules
import importlib

from ..utils.focal_loss import FocalLoss


def run(settings):
    settings.description = 'Training script for ProTeus-H Phase 3'

    # update the default configs with config file
    if not os.path.exists(settings.cfg_file):
        raise ValueError("%s doesn't exist." % settings.cfg_file)
    config_module = importlib.import_module("lib.config.%s.config" % settings.script_name)
    cfg = config_module.cfg
    config_module.update_config_from_file(settings.cfg_file)
    if settings.local_rank in [-1, 0]:
        print("New configuration is shown below.")
        for key in cfg.keys():
            print("%s configuration:" % key, cfg[key])
            print('\n')

    # update settings based on cfg
    update_settings(settings, cfg)

    # 【新增】将 YAML 中的解冻参数传递给 settings
    # 这样 LTRTrainer 就能读到 20 了，而不是默认的 40
    settings.unfreeze_epoch = getattr(cfg.TRAIN, "UNFREEZE_EPOCH", 40)

    # 强制开启每轮保存
    settings.save_every_epoch = True
    settings.save_interval = 1

    # Record the training log
    log_dir = os.path.join(settings.save_dir, 'logs')
    if settings.local_rank in [-1, 0]:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
    settings.log_file = os.path.join(log_dir, "%s-%s.log" % (settings.script_name, settings.config_name))

    # Build dataloaders
    loader_train, loader_val = build_dataloaders(cfg, settings)

    if "RepVGG" in cfg.MODEL.BACKBONE.TYPE or "swin" in cfg.MODEL.BACKBONE.TYPE or "LightTrack" in cfg.MODEL.BACKBONE.TYPE:
        cfg.ckpt_dir = settings.save_dir

    # Create network
    if settings.script_name == "ostrack":
        net = build_ostrack(cfg)
    else:
        raise ValueError("illegal script name")

    # wrap networks to distributed one
    net.cuda()
    if settings.local_rank != -1:
        net = DDP(net, device_ids=[settings.local_rank], find_unused_parameters=True)
        settings.device = torch.device("cuda:%d" % settings.local_rank)
    else:
        settings.device = torch.device("cuda:0")
    settings.deep_sup = getattr(cfg.TRAIN, "DEEP_SUPERVISION", False)
    settings.distill = getattr(cfg.TRAIN, "DISTILL", False)
    settings.distill_loss_type = getattr(cfg.TRAIN, "DISTILL_LOSS_TYPE", "KL")

    # Loss functions and Actors
    if settings.script_name == "ostrack":
        focal_loss = FocalLoss()
        objective = {'giou': giou_loss, 'l1': l1_loss, 'focal': focal_loss, 'cls': BCEWithLogitsLoss()}
        loss_weight = {'giou': cfg.TRAIN.GIOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT, 'focal': 1., 'cls': 1.0}
        actor = OSTrackActor(net=net, objective=objective, loss_weight=loss_weight, settings=settings, cfg=cfg)
    else:
        raise ValueError("illegal script name")

    # ==========================================================
    # --- ProTeus-H Phase 3: 优化器预注册逻辑 (核心) ---
    # ==========================================================
    print(">>> [Phase 3] Configuring Optimizer with Pre-registration Strategy...")

    # 步骤 A: 临时开启所有参数的梯度
    # 目的：欺骗 PyTorch 优化器，让它把 Backbone 和 Mamba 的参数也注册进去
    # 否则如果一开始是 False，优化器永远不会更新它们，即使后面解冻了也没用
    for p in net.parameters():
        p.requires_grad = True

    # 步骤 B: 构建优化器
    # 此时所有参数都在 parameters() 里，且 requires_grad=True
    optimizer, lr_scheduler = get_optimizer_scheduler(net, cfg)

    # 步骤 C: 立即恢复 Phase 3 初始的冻结状态
    # 只有这样，前 UNFREEZE_EPOCH 轮 Backbone 才不会动
    if hasattr(net, "module"):  # DDP 包装的情况
        backbone_net = net.module.backbone
        predictor_net = net.module.predictor
    else:  # 单卡模式
        backbone_net = net.backbone
        predictor_net = net.predictor

    # 重新冻结
    for p in backbone_net.parameters():
        p.requires_grad = False
    for p in predictor_net.parameters():
        p.requires_grad = False

    print(">>> [Phase 3] Optimizer initialized. Backbone & Mamba are currently FROZEN.")
    # ==========================================================

    use_amp = getattr(cfg.TRAIN, "AMP", False)
    trainer = LTRTrainer(actor, [loader_train, loader_val], optimizer, settings, lr_scheduler, use_amp=use_amp)

    # train process
    trainer.train(cfg.TRAIN.EPOCH, load_latest=True, fail_safe=True)