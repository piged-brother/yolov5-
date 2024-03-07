# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Loss functions
"""

import torch
import torch.nn as nn

from utils.metrics import bbox_iou
from utils.torch_utils import de_parallel

#平滑处理，0.95正样本值  * 0.05负样本值
def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super().__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # 二分交叉熵损失
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  #  用sigmoid得到预测样本值
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)#然后我们得到的预测值综合正负样本结果 意思是我们必须要正确的得到正样本 和真确的得到负样本
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)#计算正负样本因子
        modulating_factor = (1.0 - p_t) ** self.gamma#用1减去我们的预测正负样本值然后进行gamma的指数倍数乘  意味着我们预测对了那我们的(1.0 - p_t)会很小。反过来 那我们假如预测错了 那么就会很大显然看出差异 损失就会很大
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class ComputeLoss:
    sort_obj_iou = False

    # Compute losses
    def __init__(self, model, autobalance=False):#传入的model是模型又三层 1/8 1/16 1/32
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))#类别损失    二分类交叉熵
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))#目标概率损失

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # 类别的平滑处理  经过实验得知正样本设置0.95 负样本为0.05

        # Focal loss
        g = h['fl_gamma']  # g的值为0 在yaml文件中 所以不会包一层focalloss

        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)#在二分类的基础上在包一层focalloss

        m = de_parallel(model).model[-1]  # 每个模型的最后一层都是detect层
        self.balance = {3: [4.0, 1.0, 0.4]}.get(m.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7  模型有三层每一层的权重属性都不一样
        self.ssi = list(m.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        self.na = m.na  # number of anchors
        self.nc = m.nc  # number of classes
        self.nl = m.nl  # number of layers
        self.anchors = m.anchors
        self.device = device

    def __call__(self, p, targets):  # predictions 原图1/8 ... tensor, targets 原图的目标类别的数据
        lcls = torch.zeros(1, device=self.device)  # class loss
        lbox = torch.zeros(1, device=self.device)  # box loss
        lobj = torch.zeros(1, device=self.device)  # object loss
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # 计算我们的目标值 得到每个目标边框对于类别 ：对应的边框预测值：对应到tensor属于哪个batch，属于哪一列哪一行：对应的yaml里面的anchor

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions遍历每一层预测的tensor
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx 拿到左边的数据
            tobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)  # target obj

            n = b.shape[0]  # 木目标的数量
            if n:
                # pxy, pwh, _, pcls = pi[b, a, gj, gi].tensor_split((2, 4, 5), dim=1)  # faster, requires torch 1.8.0
                pxy, pwh, _, pcls = pi[b, a, gj, gi].split((2, 2, 1, self.nc), 1)  # 根据索引找到pxy pwh pcl的预测值

                # Regression 计算边框的损失 在之前的图里面一样
                pxy = pxy.sigmoid() * 2 - 0.5
                pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]#根据比例得到我们边框预测的宽高
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()  #拿预测边框的真实边框区计算他们的Ciou
                lbox += (1.0 - iou).mean()  # 得到边框损失

                # Objectness做目标概率的损失
                iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    j = iou.argsort()
                    b, a, gj, gi, iou = b[j], a[j], gj[j], gi[j], iou[j]
                if self.gr < 1:
                    iou = (1.0 - self.gr) + self.gr * iou
                tobj[b, a, gj, gi] = iou  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(pcls, self.cn, device=self.device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(pcls, t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # 每一层的目标概率权重是不一样的 我们需要乘
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()#权重会自己调整自己的大小

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = tobj.shape[0]  # batch size

        return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()

    def build_targets(self, p, targets):#传入的p是之前将的predict三层的向量  target是我们处理好经过归一化后得到的图片 就是原图处理好后得到的
        # p[[batch,num_anchor=3,w/8,h/8,nc+5],[....],[...] ]p有三个[]也就是三层, input targets(image,class,x,y,w,h)这里的xywh都是处理到（0，1）区间了
        na, nt = self.na, targets.shape[0]  # 每层有几个anchor, 一共几个正样本 target一共多少行
        tcls, tbox, indices, anch = [], [], [], []#每个目标边框对于类别 ：对应的边框预测值：对应到tensor属于哪个batch，属于哪一列哪一行：对应的yaml里面的anchor
        gain = torch.ones(7, device=self.device)  # 单行七列的数组
        ai = torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None]), 2)  # append anchor indices 加上anchor的索引让每个target对应好他们各自的anchor

        g = 0.5  # bias
        off = torch.tensor(
            [
                [0, 0],
                [1, 0],
                [0, 1],
                [-1, 0],
                [0, -1],  # j,k,l,m
                # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
            ],
            device=self.device).float() * g  # offsets

        for i in range(self.nl):#遍历每一层预测层 一共三层
            anchors, shape = self.anchors[i], p[i].shape#拿到当前层对应的anchor 和当前层 也就是p[i]的shape存入shape
            gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]]  # xyxy gain 把我们的p的宽高放到我们的gain里面

            # Match targets to anchors
            t = targets * gain  # shape(3,n,7) 让我们这个操作的目的是将目标框的坐标转换为相对于当前层的范围，以便与锚点进行匹配。也就是先把对应的结构创建好。
            if nt:#如果被检测的目标的数量不为0
                # Matches
                r = t[..., 4:6] / anchors[:, None]  # 拿出 4 5列 就是我们的最后目标边框的宽和高和我们anchors的宽高比例
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp['anchor_t']  # compare比  拿出的值永远比1大 但是必须小于4  最小不能小于1/4  所以区间在[0.25,4]  因为宽高都必须在这个区间里面 这样可以保证我们的宽高不会太小也不会太大
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # 然后拿到限制后的宽高比例  最后会去用这个比例乘以我们定义的anchor的宽高去进行得到最后在图层显示的anchor

                # Offsets
                gxy = t[:, 2:4]  # grid xy  获得我们当前尺度下面的xy坐标也就是中心点坐标
                gxi = gain[[2, 3]] - gxy  # 这里是将这些点对折过来 比如是8x8的这些点 我们将我们的当前tensor（也就是8x8）减去我们的gxy这些已经是当前8x8图层上的点后得到的点 举个例子就是（0.7，0.7）会变成（8-0.7，8-0.7）这样
                j, k = ((gxy % 1 < g) & (gxy > 1)).T#帅选 就是筛选xy坐标两个都大于1的和xy两个取模1后得到的小鼠要小于g的  这里g是偏移量也就是0.5在上面赋值了  那么得到的jk就是x和y方向的两个bool数组 对应满足条件点的情况
                l, m = ((gxi % 1 < g) & (gxi > 1)).T#反转后的坐标进行筛选
                j = torch.stack((torch.ones_like(j), j, k, l, m))#在原在存在正样本情况下将我们得到的新的样本进行拼接
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define 接下来把我们目标边框为true的坐标减去偏移量
            bc, gxy, gwh, a = t.chunk(4, 1)  # (image, class), grid xy, grid wh, anchors
            a, (b, c) = a.long().view(-1), bc.long().T  # anchors, image, class
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid indices

            # Append 添加正样本就是在我们ground truth中  离我们中心顶啊近的左下或者右上来添加样本
            indices.append((b, a, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)))  # image, anchor, grid
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch
