# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
YOLO-specific modules

Usage:
    $ python models/yolo.py --cfg yolov5s.yaml
"""

import argparse
import contextlib
import os
import platform
import sys
from copy import deepcopy
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if platform.system() != 'Windows':
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import *  # noqa
from models.experimental import *  # noqa
from utils.autoanchor import check_anchor_order
from utils.general import LOGGER, check_version, check_yaml, make_divisible, print_args
from utils.plots import feature_visualization
from utils.torch_utils import (fuse_conv_and_bn, initialize_weights, model_info, profile, scale_img, select_device,
                               time_sync)

try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None


class Detect(nn.Module):
    # YOLOv5 Detect head for detection models
    stride = None  # strides computed during build
    dynamic = False  # force grid reconstruction
    export = False  # export mode

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.empty(0) for _ in range(self.nl)]  # init grid
        self.anchor_grid = [torch.empty(0) for _ in range(self.nl)]  # init anchor grid
        #pytroch里面有两种保存参数的方法
        # 1.nn.Parameter(requires_grad=True)  这个是保存反向传播的梯度，并非进行跟随。这个是可参与训练的参数
        # 第二种就是下面这一行  不参与训练的方式生成'anchors'这个参数  。并且可以直接用self.anchors调用   第一种和第二种都会随着save从而保存

        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv  这里我解释一下参数用models/yolov5l.yaml的detect层进行举例
        #  这个是其detect层 [[17, 20, 23], 1, Detect, [nc, anchors]]   那么上一行代码主要就是生成该层的卷积层因为[17, 20, 23]有三个元素 所以每个元素都生成对应的卷积层
        #  参数x就是[17, 20, 23]，意思是把这一层的输出当作输入因为有三层 那么x可以选的值为17 ，20， 23
        # 那么no*na为输出  因为我们要对每个框进行种类输出 以coco数据集为例子就是85*3  na为3是因为已经定义好了在之前讲过
        #1为kernelsize  就是1x1层   ch就是[17, 20, 23]的长度，也就是层数的多少
        self.inplace = inplace  # use inplace ops (e.g. slice assignment)

    def forward(self, x): #这个x传入的时候应该是这样的[[],[],[]]一个列表 里面有三层  每一层对应原图大小宽高输入为1/8 1/16 1/32
        z = []  # inference output
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)  这里就是把(batchsize,.no * na.宽,高) =>(batchsize,na,宽,高,no)把需要用的参数初始化好
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()#将刚刚弄好的参数进行重新设置，意思是把位置调整一下
#假设训练就走到上一行就完事了，但是假如是在预测，那么下面的都要走完。
            if not self.training:  #
                if self.dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:#这个条件语句检查是否需要重新生成网格（grid）。也就是grid里面的宽高和我们所创建的没一层x[i]的宽高是否一致 不一致则重新生成网格后在进行推理
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)#利用_make_grid函数来进行网格的重新生成在89行

                if isinstance(self, Segment):  # (boxes + masks)
                    xy, wh, conf, mask = x[i].split((2, 2, self.nc + 1, self.no - self.nc - 5), 4)
                    xy = (xy.sigmoid() * 2 + self.grid[i]) * self.stride[i]  # xy 对xy进行转换，把缩放到(0,1)的坐标缩放为原图为   什么乘以2？问得好，应该是v5正样本匹配机制会把最新相邻的两个区域也作为正样本，所以乘以2来让他能达到隔壁的区域
                    wh = (wh.sigmoid() * 2) ** 2 * self.anchor_grid[i]  # wh对wh进行转换，把缩放到(0,1)的坐标缩放为原图
                    y = torch.cat((xy, wh, conf.sigmoid(), mask), 4)#将处理后的边界框坐标、尺寸、置信度和掩模合并为一个张量 y
                else:  # Detect (boxes only)
                    xy, wh, conf = x[i].sigmoid().split((2, 2, self.nc + 1), 4)
                    xy = (xy * 2 + self.grid[i]) * self.stride[i]  # xy
                    wh = (wh * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, conf), 4)
                z.append(y.view(bs, self.na * nx * ny, self.no))#最后将预测好的xywhc拼接在一起返回

        return x if self.training else (torch.cat(z, 1), ) if self.export else (torch.cat(z, 1), x)

    def _make_grid(self, nx=20, ny=20, i=0, torch_1_10=check_version(torch.__version__, '1.10.0')):
        d = self.anchors[i].device#获取设备
        t = self.anchors[i].dtype#获取数据类型
        shape = 1, self.na, ny, nx, 2  # grid shape
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)#使用 torch.arange 函数生成高度和宽度方向上的坐标
        yv, xv = torch.meshgrid(y, x, indexing='ij') if torch_1_10 else torch.meshgrid(y, x)  # 在 torch 1.10 及以上版本中，torch.meshgrid 函数使用 'ij' 索引顺序。# 在 torch 1.10 以下版本中，torch.meshgrid 函数使用默认的索引顺序。
        grid = torch.stack((xv, yv), 2).expand(shape) - 0.5  # add grid offset, i.e. y = 2.0 * x - 0.5 将网格坐标 xv 和 yv 沿第三个维度堆叠，形成形状为 (ny, nx, 2) 的张量，
        # 并扩展为指定形状 shape。由于原始网格的坐标范围是从 0 开始的，这里通过减去 0.5 来得到中心化的坐标，使得坐标范围从 -0.5 到 nx - 0.5。
        anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(shape)#计算锚框在网格中的位置。将锚框尺寸乘以步长 self.stride[i]，然后扩展为指定形状 shape。(1, self.na, 1, 1, 2)=（1 3 1 1 2）因为这个大小是我们的tensor大小 要相匹配做乘法
        return grid, anchor_grid


class Segment(Detect):
    # YOLOv5 Segment head for segmentation models
    def __init__(self, nc=80, anchors=(), nm=32, npr=256, ch=(), inplace=True):
        super().__init__(nc, anchors, ch, inplace)
        self.nm = nm  # number of masks
        self.npr = npr  # number of protos
        self.no = 5 + nc + self.nm  # number of outputs per anchor
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.proto = Proto(ch[0], self.npr, self.nm)  # protos
        self.detect = Detect.forward

    def forward(self, x):
        p = self.proto(x[0])
        x = self.detect(self, x)
        return (x, p) if self.training else (x[0], p) if self.export else (x[0], p, x[1])


class BaseModel(nn.Module):
    # YOLOv5 base model
    def forward(self, x, profile=False, visualize=False):
        return self._forward_once(x, profile, visualize)  # single-scale inference, train 单次前向传播

    def _forward_once(self, x, profile=False, visualize=False):
        y, dt = [], []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
        return x

    def _profile_one_layer(self, m, x, dt):
        c = m == self.model[-1]  # is final layer, copy input as inplace fix
        o = thop.profile(m, inputs=(x.copy() if c else x, ), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPs
        t = time_sync()
        for _ in range(10):
            m(x.copy() if c else x)
        dt.append((time_sync() - t) * 100)
        if m == self.model[0]:
            LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  module")
        LOGGER.info(f'{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}')
        if c:
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        LOGGER.info('Fusing layers... ')
        for m in self.model.modules():
            if isinstance(m, (Conv, DWConv)) and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.forward_fuse  # update forward
        self.info()
        return self

    def info(self, verbose=False, img_size=640):  # print model information
        model_info(self, verbose, img_size)

    def _apply(self, fn):
        # Apply to(), cpu(), cuda(), half() to model tensors that are not parameters or registered buffers
        self = super()._apply(fn)
        m = self.model[-1]  # Detect()
        if isinstance(m, (Detect, Segment)):
            m.stride = fn(m.stride)
            m.grid = list(map(fn, m.grid))
            if isinstance(m.anchor_grid, list):
                m.anchor_grid = list(map(fn, m.anchor_grid))
        return self


class DetectionModel(BaseModel):
    # YOLOv5 detection model
    def __init__(self, cfg='yolov5s.yaml', ch=3, nc=None, anchors=None):  # 参数含义是选择的模型/在model文件夹里面, ch：输入通道, nc：类的数量
        super().__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg, encoding='ascii', errors='ignore') as f:
                self.yaml = yaml.safe_load(f)  # 先把yaml文件读进来，然后加载那些key值

        # Define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # 保存我们传进来的channal参数
        if nc and nc != self.yaml['nc']:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # 如果我们有自己定义类数量 那么我们会重写这个nc
        if anchors:
            LOGGER.info(f'Overriding model.yaml anchors with anchors={anchors}')
            self.yaml['anchors'] = round(anchors)  # 同样自己定义anchors的话也会进行重写
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist  开始解析模型
        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names
        self.inplace = self.yaml.get('inplace', True)

        # Build strides, anchors
        m = self.model[-1]  # Detect()  解析完后拿到最后一层 ，一般都是Detect层
        if isinstance(m, (Detect, Segment)):
            s = 256  # 2x min stride
            m.inplace = self.inplace#是否进行原图操作的标志。
            forward = lambda x: self.forward(x)[0] if isinstance(m, Segment) else self.forward(x)#这个foward是 进行单次训练
            m.stride = torch.tensor([s / x.shape[-2] for x in forward(torch.zeros(1, ch, s, s))])  # forward  就是算三个特征层的下采样倍数吧，就是8、16和32  这里得到的就是[8,16,32]
            check_anchor_order(m)
            m.anchors /= m.stride.view(-1, 1, 1)#把每个anchors除以这个步长也就是缩放倍数[8,16,32]
            self.stride = m.stride
            self._initialize_biases()  # only run once

        # Init weights, biases
        initialize_weights(self)
        self.info()
        LOGGER.info('')

    def forward(self, x, augment=False, profile=False, visualize=False):
        if augment:
            return self._forward_augment(x)  # augmented inference, None
        return self._forward_once(x, profile, visualize)  # single-scale inference, train

    def _forward_augment(self, x):
        img_size = x.shape[-2:]  # height, width
        s = [1, 0.83, 0.67]  # scales
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = self._forward_once(xi)[0]  # forward
            # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)  # clip augmented tails
        return torch.cat(y, 1), None  # augmented inference, train

    def _descale_pred(self, p, flips, scale, img_size):
        # de-scale predictions following augmented inference (inverse operation)
        if self.inplace:
            p[..., :4] /= scale  # de-scale
            if flips == 2:
                p[..., 1] = img_size[0] - p[..., 1]  # de-flip ud
            elif flips == 3:
                p[..., 0] = img_size[1] - p[..., 0]  # de-flip lr
        else:
            x, y, wh = p[..., 0:1] / scale, p[..., 1:2] / scale, p[..., 2:4] / scale  # de-scale
            if flips == 2:
                y = img_size[0] - y  # de-flip ud
            elif flips == 3:
                x = img_size[1] - x  # de-flip lr
            p = torch.cat((x, y, wh, p[..., 4:]), -1)
        return p

    def _clip_augmented(self, y):
        # Clip YOLOv5 augmented inference tails
        nl = self.model[-1].nl  # number of detection layers (P3-P5)
        g = sum(4 ** x for x in range(nl))  # grid points
        e = 1  # exclude layer count
        i = (y[0].shape[1] // g) * sum(4 ** x for x in range(e))  # indices
        y[0] = y[0][:, :-i]  # large
        i = (y[-1].shape[1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
        y[-1] = y[-1][:, i:]  # small
        return y

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:5 + m.nc] += math.log(0.6 / (m.nc - 0.99999)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)


Model = DetectionModel  # retain YOLOv5 'Model' class for backwards compatibility


class SegmentationModel(DetectionModel):
    # YOLOv5 segmentation model
    def __init__(self, cfg='yolov5s-seg.yaml', ch=3, nc=None, anchors=None):
        super().__init__(cfg, ch, nc, anchors)


class ClassificationModel(BaseModel):
    # YOLOv5 classification model
    def __init__(self, cfg=None, model=None, nc=1000, cutoff=10):  # yaml, model, number of classes, cutoff index
        super().__init__()
        self._from_detection_model(model, nc, cutoff) if model is not None else self._from_yaml(cfg)

    def _from_detection_model(self, model, nc=1000, cutoff=10):
        # Create a YOLOv5 classification model from a YOLOv5 detection model
        if isinstance(model, DetectMultiBackend):
            model = model.model  # unwrap DetectMultiBackend
        model.model = model.model[:cutoff]  # backbone
        m = model.model[-1]  # last layer
        ch = m.conv.in_channels if hasattr(m, 'conv') else m.cv1.conv.in_channels  # ch into module
        c = Classify(ch, nc)  # Classify()
        c.i, c.f, c.type = m.i, m.f, 'models.common.Classify'  # index, from, type
        model.model[-1] = c  # replace
        self.model = model.model
        self.stride = model.stride
        self.save = []
        self.nc = nc

    def _from_yaml(self, cfg):
        # Create a YOLOv5 classification model from a *.yaml file
        self.model = None


def parse_model(d, ch):  # model_dict, input_channels(3)
    # Parse a YOLOv5 model.yaml dictionary
    LOGGER.info(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}")
    anchors, nc, gd, gw, act = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple'], d.get('activation')#一开始就会解析anchors，class，宽度因子和深度因子
    if act:
        Conv.default_act = eval(act)  # redefine default activation, i.e. Conv.default_act = nn.SiLU()
        LOGGER.info(f"{colorstr('activation:')} {act}")  # print
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors 获得anchor的数量  就比如在models/yolov5l.yaml文件中  anchors的key值读取出来是二维数组anchors[0]代表第一行 有6个数字，每两个是代表anchor的宽高
    # 除以2就代表着anchor的数量
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5) 我们总共会输出多少，意思是总共又na个框，每个框都有nc（也就是numberclass）和   （ 5是 宽高+中心点坐标+边框置信度）

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out  这里的layers是我们models/yolov5l.yaml中的backbone key里面的。比如[-1, 1, Conv, [64, 6, 2, 2]]这个layer就代表Conv;  save表示这一层那些参数需要保存;c2代表当前层的输出
    #接下来这个for循环就是读取backbone:和head里面的参数
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args  这个是列遍历，比如遍历[-1, 1, Conv, [64, 6, 2, 2]   这个就是便利backbone key对应的数组
        m = eval(m) if isinstance(m, str) else m  # eval strings  然后这里是把一个字符串处理成我们程序所对应的方法或者类  比如把Conv处理成Conv类
        for j, a in enumerate(args):
            with contextlib.suppress(NameError):
                args[j] = eval(a) if isinstance(a, str) else a  #这里还要对每个元素进行遍历，你可能没听懂。我直接上例子，还是上面 遍历[64, 6, 2, 2]这里面的每个元素为什么要遍历 因为在head最后参数里面有
                #[[17, 20, 23], 1, Detect, [nc, anchors]]里面的[nc, anchors]  这个我们需要两个都进行转换成我们所需要的变量  也就是把nc和anchors这两个字符串转换成变量

        n = n_ = max(round(n * gd), 1) if n > 1 else n  # depth gain深度因子乘以1x1层的数量
        if m in {#如果我们用的卷积层属于一下列表的卷积层
                Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, MixConv2d, Focus, CrossConv,
                BottleneckCSP, C3, C3TR, C3SPP, C3Ghost, nn.ConvTranspose2d, DWConvTranspose2d, C3x}:
            c1, c2 = ch[f], args[0]#那么我们就开始调整输入输出 c1是输入通道，c2书输出通道  ch[f]就是上面说的yolov5l.yaml中backbone键里面的第一列 代表 这上一层的输出和这一层的输入
            if c2 != no:  # 当前输出不是属于到最后一层  就是目前不是最后的结果，
                c2 = make_divisible(c2 * gw, 8)#那么我们就会用我们的宽度因子gw，保证这一层输出通道被拉多 所以每一层都会影响到

            args = [c1, c2, *args[1:]]#这里的意思是重新调整[64, 6, 2, 2]这样的数据  让我们新得到的输出和输入通道放进去[c1,c2,kernelsize,stride,padding]
            if m in {BottleneckCSP, C3, C3TR, C3Ghost, C3x}:#假如m层属于左边这三个之一
                args.insert(2, n)  # 把n这个参数加到c2后面从而形成[c1,c2,n.kernelsize,stride,padding]  为什么要加n  因为在common.py文件里面只有这五个类里面有n这个参数 其他的都没有
                n = 1 #此处n=1是表示n最小是1不能比1还小
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:#假如是concat层 比如这个： [[-1, 14], 1, Concat, [1]]  中[-1, 14]表示上一层的输出和第14层的输出加起来当作下一层的输出
            c2 = sum(ch[x] for x in f)
        # TODO: channel, gw, gd
        elif m in {Detect, Segment}:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
            if m is Segment:
                args[3] = make_divisible(args[3] * gw, 8)
        elif m is Contract:#从此到347行 我们最最新的yolov5 6.0版本没有用到
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module  开始去把刚刚处理好的参数列表用 m(*args)进行实例化   这里注意！只会执行m(*args)因为n只有在
        # {BottleneckCSP, C3, C3TR, C3Ghost, C3x}这四个层才会用到  但是仔细看 n插入到列表后又会赋值为1 所以这一行只会执行m(*args)所以只会实例化一次这个模块
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum(x.numel() for x in m_.parameters())  # number params 从此往下就是保存参数的环节了
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        LOGGER.info(f'{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}')  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)#这里是把我们实例化的模块加入到layers中
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)#最后是一次性保存所有我们实例化的模块  处理成模型


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='yolov5s.yaml', help='model.yaml')
    parser.add_argument('--batch-size', type=int, default=1, help='total batch size for all GPUs')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--profile', action='store_true', help='profile model speed')
    parser.add_argument('--line-profile', action='store_true', help='profile model speed layer by layer')
    parser.add_argument('--test', action='store_true', help='test all yolo*.yaml')
    opt = parser.parse_args()
    opt.cfg = check_yaml(opt.cfg)  # check YAML
    print_args(vars(opt))
    device = select_device(opt.device)

    # Create model
    im = torch.rand(opt.batch_size, 3, 640, 640).to(device)
    model = Model(opt.cfg).to(device)

    # Options
    if opt.line_profile:  # profile layer by layer
        model(im, profile=True)

    elif opt.profile:  # profile forward-backward
        results = profile(input=im, ops=[model], n=3)

    elif opt.test:  # test all models
        for cfg in Path(ROOT / 'models').rglob('yolo*.yaml'):
            try:
                _ = Model(cfg)
            except Exception as e:
                print(f'Error in {cfg}: {e}')

    else:  # report fused model summary
        model.fuse()
