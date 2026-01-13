from . import BaseActor
from lib.utils.misc import NestedTensor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
import torch
import torch.nn.functional as F  # 新增：用于计算正则化 MSE Loss
from lib.utils.merge import merge_template_search
from ...utils.heapmap_utils import generate_heatmap
from ...utils.ce_utils import generate_mask_cond, adjust_keep_rate


class OSTrackActor(BaseActor):
    """ Actor for ProTeus-H Phase 3 training """

    def __init__(self, net, objective, loss_weight, settings, cfg=None):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize
        self.cfg = cfg

    def __call__(self, data):
        """
        args:
            data - 包含 'template', 'search', 'epoch' 等字段
        """
        # forward pass
        out_dict = self.forward_pass(data)

        # 获取当前 epoch 用于动态调整 Loss 权重
        curr_epoch = data.get('epoch', 0)

        # compute losses
        loss, status = self.compute_losses(out_dict, data, curr_epoch=curr_epoch)
        # 1. 计算完各个 Loss 之后，获取当前的 fusion_alpha 值
        # 注意：如果是多卡训练 (DDP)，需要通过 .module 访问
        curr_net = self.net.module if hasattr(self.net, 'module') else self.net

        # 2. 将 alpha 值存入 status 字典
        if hasattr(curr_net, 'fusion_alpha'):
            # 获取原始参数值
            raw_alpha = curr_net.fusion_alpha.item()
            status['alpha'] = raw_alpha

            # 如果你想看经过 tanh 激活后真正生效的系数（推荐）：
            import torch
            status['alpha_eff'] = torch.tanh(torch.tensor(raw_alpha)).item()

        return loss, status

    def forward_pass(self, data):
        assert len(data['template_images']) == 1
        assert len(data['search_images']) == 1

        template_list = []
        for i in range(self.settings.num_template):
            template_img_i = data['template_images'][i].view(-1, *data['template_images'].shape[2:])
            template_list.append(template_img_i)

        search_img = data['search_images'][0].view(-1, *data['search_images'].shape[2:])

        box_mask_z = None
        ce_keep_rate = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            box_mask_z = generate_mask_cond(self.cfg, template_list[0].shape[0], template_list[0].device,
                                            data['template_anno'][0])
            ce_start_epoch = self.cfg.TRAIN.CE_START_EPOCH
            ce_warm_epoch = self.cfg.TRAIN.CE_WARM_EPOCH
            ce_keep_rate = adjust_keep_rate(data['epoch'], warmup_epochs=ce_start_epoch,
                                            total_epochs=ce_start_epoch + ce_warm_epoch,
                                            ITERS_PER_EPOCH=1,
                                            base_keep_rate=self.cfg.MODEL.BACKBONE.CE_KEEP_RATIO[0])

        if len(template_list) == 1:
            template_list = template_list[0]

        # 调用 ProTeus-H 的 forward，它会返回 p_next 和 p_anchor
        out_dict = self.net(template=template_list,
                            search=search_img,
                            ce_template_mask=box_mask_z,
                            ce_keep_rate=ce_keep_rate,
                            return_last_attn=False)
        return out_dict

    def compute_losses(self, pred_dict, gt_dict, return_status=True, curr_epoch=0):
        # 1. 基础检测真值准备
        gt_bbox = gt_dict['search_anno'][-1]
        gt_gaussian_maps = generate_heatmap(gt_dict['search_anno'], self.cfg.DATA.SEARCH.SIZE,
                                            self.cfg.MODEL.BACKBONE.STRIDE)
        gt_gaussian_maps = gt_gaussian_maps[-1].unsqueeze(1)

        # 2. 边界框损失计算
        pred_boxes = pred_dict['pred_boxes']
        if torch.isnan(pred_boxes).any():
            raise ValueError("Network outputs is NAN! Stop Training")
        num_queries = pred_boxes.size(1)
        pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)
        gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0,
                                                                                                           max=1.0)

        try:
            giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)
        except:
            giou_loss, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()

        l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)

        if 'score_map' in pred_dict:
            location_loss = self.objective['focal'](pred_dict['score_map'], gt_gaussian_maps)
        else:
            location_loss = torch.tensor(0.0, device=l1_loss.device)

        # --- ProTeus-H Phase 3：动态正则化权重策略 ---

        # 动态读取 YAML 配置，不再写死！
        unfreeze_epoch = self.cfg.TRAIN.UNFREEZE_EPOCH

        if 'p_next' in pred_dict and 'p_anchor' in pred_dict:
            # 1. 确保 Target 不传梯度给 Backbone (只训练 Mamba)
            target = pred_dict['p_anchor'].detach()
            prediction = pred_dict['p_next']

            # 2. Cosine Embedding Loss
            # 目标是最大化相似度，即最小化 (1 - cos)
            cosine_sim = F.cosine_similarity(prediction, target, dim=-1)
            loss_reg = (1 - cosine_sim).mean()
            # 动态权重：解冻后提高约束强度
            reg_weight = 0.1 if curr_epoch < unfreeze_epoch else self.cfg.TRAIN.REG_WEIGHT
        else:
            loss_reg = torch.tensor(0.0, device=l1_loss.device)
            reg_weight = 0.0
        # ----------------------------------------------

        # 3. 加权总损失
        loss = self.loss_weight['giou'] * giou_loss + \
               self.loss_weight['l1'] * l1_loss + \
               self.loss_weight['focal'] * location_loss + \
               reg_weight * loss_reg

        if return_status:
            mean_iou = iou.detach().mean()
            status = {"Loss/total": loss.item(),
                      "Loss/giou": giou_loss.item(),
                      "Loss/l1": l1_loss.item(),
                      "Loss/location": location_loss.item(),
                      "Loss/reg": loss_reg.item(),
                      "reg_w": reg_weight,  # 记录当前权重，方便在 WandB 查看
                      "IoU": mean_iou.item()}
            return loss, status
        else:
            return loss