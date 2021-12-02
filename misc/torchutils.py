import torch
from torch.optim import lr_scheduler
from torch.utils.data import Subset
import torch.nn.functional as F
import numpy as np
import math
import random
import os
from torch.nn import MaxPool1d,AvgPool1d
from torch import Tensor
from typing import Iterable, Set, Tuple


__all__ = ['cls_accuracy']



def visualize_imgs(*imgs):
    """
    可视化图像，ndarray格式的图像
    :param imgs: ndarray：H*W*C, C=1/3
    :return:
    """
    import matplotlib.pyplot as plt
    nums = len(imgs)
    if nums > 1:
        fig, axs = plt.subplots(1, nums)
        for i, image in enumerate(imgs):
            axs[i].imshow(image, cmap='jet')
    elif nums == 1:
        fig, ax = plt.subplots(1, nums)
        for i, image in enumerate(imgs):
            ax.imshow(image, cmap='jet')
        plt.show()
    plt.show()

def minmax(tensor):
    assert tensor.ndim >= 2
    shape = tensor.shape
    tensor = tensor.view([*shape[:-2], shape[-1]*shape[-2]])
    min_, _ = tensor.min(-1, keepdim=True)
    max_, _ = tensor.max(-1, keepdim=True)
    return min_, max_

def norm_tensor(tensor,min_=None,max_=None, mode='minmax'):
    """
    输入：N*C*H*W / C*H*W / H*W
    输出：在H*W维度的归一化的与原始等大的图
    """
    assert tensor.ndim >= 2
    shape = tensor.shape
    tensor = tensor.view([*shape[:-2], shape[-1]*shape[-2]])
    if mode == 'minmax':
        if min_ is None:
            min_, _ = tensor.min(-1, keepdim=True)
        if max_ is None:
            max_, _ = tensor.max(-1, keepdim=True)
        tensor = (tensor - min_) / (max_ - min_ + 0.00000000001)
    elif mode == 'thres':
        N = tensor.shape[-1]
        thres_a = 0.001
        top_k = round(thres_a*N)
        max_ = tensor.topk(top_k, dim=-1, largest=True)[0][..., -1]
        max_ = max_.unsqueeze(-1)
        min_ = tensor.topk(top_k, dim=-1, largest=False)[0][..., -1]
        min_ = min_.unsqueeze(-1)
        tensor = (tensor - min_) / (max_ - min_ + 0.00000000001)

    elif mode == 'std':
        mean, std = torch.std_mean(tensor, [-1], keepdim=True)
        tensor = (tensor - mean)/std
        min_, _ = tensor.min(-1, keepdim=True)
        max_, _ = tensor.max(-1, keepdim=True)
        tensor = (tensor - min_) / (max_ - min_ + 0.00000000001)
    elif mode == 'exp':
        tai = 1
        tensor = torch.nn.functional.softmax(tensor/tai, dim=-1, )
        min_, _ = tensor.min(-1, keepdim=True)
        max_, _ = tensor.max(-1, keepdim=True)
        tensor = (tensor - min_) / (max_ - min_ + 0.00000000001)
    else:
        raise NotImplementedError
    tensor = torch.clamp(tensor, 0, 1)
    return tensor.view(shape)

    # if tensor.ndim == 4:
    #     B, C, H, W = tensor.shape
    #     tensor = tensor.view([B, C, -1])
    #     min_, _ = tensor.min(-1, keepdim=True)
    #     max_, _ = tensor.max(-1, keepdim=True)
    #     tensor = (tensor - min_) / (max_ - min_ + 0.00000000001)
    #     return tensor.view(B, C, H, W)
    # elif tensor.ndim == 3:
    #     C, H, W = tensor.shape
    #     tensor = tensor.view([C, -1])
    #     min_, _ = tensor.min(-1, keepdim=True)
    #     max_, _ = tensor.max(-1, keepdim=True)
    #     tensor = (tensor - min_) / (max_ - min_ + 0.00000000001)
    #     return tensor.view(C, H, W)
    # elif tensor.ndim == 2:
    #     H, W = tensor.shape
    #     tensor = tensor.view([-1])
    #     min_, _ = tensor.min(-1, keepdim=True)
    #     max_, _ = tensor.max(-1, keepdim=True)
    #     tensor = (tensor - min_) / (max_ - min_ + 0.00000000001)
    #     return tensor.view(H, W)
    # else:
    #     raise NotImplementedError

def visulize_features(features, normalize=False):
    """
    可视化特征图，各维度make grid到一起
    """
    from torchvision.utils import make_grid
    assert features.ndim == 4
    b,c,h,w = features.shape
    features = features.view((b*c, 1, h, w))
    if normalize:
        features = norm_tensor(features)
    grid = make_grid(features)
    visualize_tensors(grid)

def visualize_tensors(*tensors):
    """
    可视化tensor，支持单通道特征或3通道图像
    :param tensors: tensor: C*H*W, C=1/3
    :return:
    """
    import matplotlib.pyplot as plt
    # from misc.torchutils import tensor2np
    images = []
    for tensor in tensors:
        assert tensor.ndim == 3 or tensor.ndim==2
        if tensor.ndim ==3:
            assert tensor.shape[0] == 1 or tensor.shape[0] == 3
        images.append(tensor2np(tensor))
    nums = len(images)
    if nums>1:
        fig, axs = plt.subplots(1, nums)
        for i, image in enumerate(images):
            axs[i].imshow(image, cmap='jet')
        plt.show()
    elif nums == 1:
        fig, ax = plt.subplots(1, nums)
        for i, image in enumerate(images):
            ax.imshow(image, cmap='jet')
        plt.show()


def np_to_tensor(image):
    """
    input: nd.array: H*W*C/H*W
    """
    if isinstance(image, torch.Tensor):
        return image
    elif isinstance(image, np.ndarray):
        if image.ndim == 3:
            if image.shape[2]==3:
                image = np.transpose(image,[2,0,1])
        elif image.ndim == 2:
            image = np.newaxis(image, 0)
        image = torch.from_numpy(image)
        return image.unsqueeze(0)


def seed_torch(seed=2019):

    # 加入以下随机种子，数据输入，随机扩充等保持一致
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # 加入所有随机种子后，模型更新后，中间结果还是不一样，
    # 发现这一的现象：前两轮，的结果还是一样；随着模型更新结果会变；
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True

def simplex(t: Tensor, axis=1) -> bool:
    _sum = t.sum(axis).type(torch.float32)
    _ones = torch.ones_like(_sum, dtype=torch.float32)
    return torch.allclose(_sum, _ones)


# Assert utils
def uniq(a: Tensor) -> Set:
    return set(torch.unique(a.cpu()).numpy())

def sset(a: Tensor, sub: Iterable) -> bool:
    return uniq(a).issubset(sub)

def eq(a: Tensor, b) -> bool:
    return torch.eq(a, b).all()

def one_hot(t: Tensor, axis=1) -> bool:
    return simplex(t, axis) and sset(t, [0, 1])


def class2one_hot(seg: Tensor, C: int) -> Tensor:
    if len(seg.shape) == 2:  # Only w, h, used by the dataloader
        seg = seg.unsqueeze(dim=0)
    assert sset(seg, list(range(C)))

    b, w, h = seg.shape  # type: Tuple[int, int, int]

    res = torch.stack([seg == c for c in range(C)], dim=1).type(torch.int32)
    assert res.shape == (b, C, w, h)
    assert one_hot(res)

    return res

class ChannelMaxPool(MaxPool1d):
    def forward(self, input):
        n, c, w, h = input.size()
        input = input.view(n,c,w*h).permute(0,2,1)
        pooled =  F.max_pool1d(input, self.kernel_size, self.stride,
                        self.padding, self.dilation, self.ceil_mode,
                        self.return_indices)
        _, _, c = pooled.size()
        pooled = pooled.permute(0,2,1)
        return pooled.view(n,c,w,h)

class ChannelAvePool(AvgPool1d):
    def forward(self, input):
        n, c, w, h = input.size()
        input = input.view(n,c,w*h).permute(0,2,1)
        pooled = F.avg_pool1d(input, self.kernel_size, self.stride,
                        self.padding)
        _, _, c = pooled.size()
        pooled = pooled.permute(0,2,1)
        return pooled.view(n,c,w,h)

def cross_entropy(input, target, weight=None, reduction='mean',ignore_index=255):
    """
    logSoftmax_with_loss
    :param input: torch.Tensor, N*C*H*W
    :param target: torch.Tensor, N*1*H*W,/ N*H*W
    :param weight: torch.Tensor, C
    :return: torch.Tensor [0]
    """
    target = target.long()
    if target.dim() == 4:
        target = torch.squeeze(target, dim=1)
    if input.shape[-1] != target.shape[-1]:
        input = F.interpolate(input, size=target.shape[1:], mode='bilinear',align_corners=True)

    return F.cross_entropy(input=input, target=target, weight=weight,
                           ignore_index=ignore_index, reduction=reduction)

def balanced_cross_entropy(input, target, weight=None,ignore_index=255):
    """
    类别均衡的交叉熵损失，暂时只支持2类
    TODO: 扩展到多类C>2
    """
    if target.dim() == 4:
        target = torch.squeeze(target, dim=1)
    if input.shape[-1] != target.shape[-1]:
        input = F.interpolate(input, size=target.shape[1:], mode='bilinear',align_corners=True)

    # print('target.sum',target.sum())
    pos = (target==1).float()
    neg = (target==0).float()
    pos_num = torch.sum(pos) + 0.0000001
    neg_num = torch.sum(neg) + 0.0000001
    # print(pos_num)
    # print(neg_num)
    target_pos = target.float()
    target_pos[target_pos!=1] = ignore_index  # 忽略不为正样本的区域
    target_neg = target.float()
    target_neg[target_neg!=0] = ignore_index  # 忽略不为负样本的区域

    # print('target.sum',target.sum())

    loss_pos = cross_entropy(input, target_pos,weight=weight,reduction='sum',ignore_index=ignore_index)
    loss_neg = cross_entropy(input, target_neg,weight=weight,reduction='sum',ignore_index=ignore_index)
    # print(loss_neg, loss_pos)
    loss = 0.5 * loss_pos / pos_num + 0.5 * loss_neg / neg_num
    # loss = (loss_pos + loss_neg)/ (pos_num+neg_num)
    return loss

def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'poly':
        max_step = opt.niter+opt.niter_decay
        power = 0.9
        def lambda_rule(epoch):
            current_step = epoch + opt.epoch_count
            lr_l = (1.0 - current_step / (max_step+1)) ** float(power)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def mul_cls_acc(preds, targets, topk=(1,)):
    """计算multi-label分类的top-k准确率topk-acc，topk-error=1-topk-acc；
    首先计算每张图的的平均准确率，再计算所有图的平均准确率
    :param pred: N * C
    :param target: N * C
    :param topk:
    :return:
    """
    with torch.no_grad():
        maxk = max(topk)
        bs, C = targets.shape
        _, pred = preds.topk(maxk, 1, True, True)
        pred += 1  # pred 为类别\in [1,C]
        # print('pred: ', pred)
        # print('targets: ', targets)
        correct = torch.zeros([bs, maxk]).long()  # 记录预测正确label数量
        if preds.device != torch.device(type='cpu'):
            correct = correct.cuda()
        for i in range(C):
            label = i + 1
            target = targets[:, i] * label
            # print('target.view: ', target.view(-1, 1).expand_as(pred))
            # print('pred: ', pred)
            correct = correct + pred.eq(target.view(-1, 1).expand_as(pred)).long()
            # print('correct: ', pred.eq(target.view(-1, 1).expand_as(pred)).long())
        n = (targets == 1).long().sum(1)  # N*1, 每张图中含有目标的数量
        # print(n)
        res = []
        for k in topk:
            acc_k = correct[:, :k].sum(1).float() / n.float()  # 每张图的平均正确率，预测正确目标数/总目标数
            # print(correct[:, :k].sum(1).float())
            acc_k = acc_k.sum()/bs
            res.append(acc_k)
            # print(acc_k)
    return res


def cls_accuracy(output, target, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for the specified values of k
    https://github.com/pytorch/examples/blob/ee964a2eeb41e1712fe719b83645c79bcbd0ba1a/imagenet/main.py#L407
    """

    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

class PolyOptimizer(torch.optim.SGD):

    def __init__(self, params, lr, weight_decay, max_step, init_step=0, momentum=0.9):
        super().__init__(params, lr, weight_decay)

        self.global_step = init_step
        print(self.global_step)
        self.max_step = max_step
        self.momentum = momentum

        self.__initial_lr = [group['lr'] for group in self.param_groups]


    def step(self, closure=None):

        if self.global_step < self.max_step:
            lr_mult = (1 - self.global_step / self.max_step) ** self.momentum

            for i in range(len(self.param_groups)):
                self.param_groups[i]['lr'] = self.__initial_lr[i] * lr_mult

        super().step(closure)

        self.global_step += 1


class PolyAdamOptimizer(torch.optim.Adam):
    def __init__(self, params, lr, betas, max_step, momentum=0.9):
        super().__init__(params, lr, betas)

        self.global_step = 0
        self.max_step = max_step
        self.momentum = momentum

        self.__initial_lr = [group['lr'] for group in self.param_groups]


    def step(self, closure=None):

        if self.global_step < self.max_step:
            lr_mult = (1 - self.global_step / self.max_step) ** self.momentum

            for i in range(len(self.param_groups)):
                self.param_groups[i]['lr'] = self.__initial_lr[i] * lr_mult

        super().step(closure)
        self.global_step += 1
#
# from ranger import RangerQH,Ranger
# # https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer/blob/master/ranger/rangerqh.py
#
# class PolyRangerOptimizer(RangerQH):
#
#     def __init__(self, params, lr, betas, max_step, momentum=0.9):
#         super().__init__(params, lr, betas)
#
#         self.global_step = 0
#         self.max_step = max_step
#         self.momentum = momentum
#
#         self.__initial_lr = [group['lr'] for group in self.param_groups]
#
#
#     def step(self, closure=None):
#
#         if self.global_step < self.max_step:
#             lr_mult = (1 - self.global_step / self.max_step) ** self.momentum
#
#             for i in range(len(self.param_groups)):
#                 self.param_groups[i]['lr'] = self.__initial_lr[i] * lr_mult
#
#         super().step(closure)
#         self.global_step += 1

class SGDROptimizer(torch.optim.SGD):

    def __init__(self, params, steps_per_epoch, lr=0, weight_decay=0, epoch_start=1, restart_mult=2):
        super().__init__(params, lr, weight_decay)

        self.global_step = 0
        self.local_step = 0
        self.total_restart = 0

        self.max_step = steps_per_epoch * epoch_start
        self.restart_mult = restart_mult

        self.__initial_lr = [group['lr'] for group in self.param_groups]


    def step(self, closure=None):

        if self.local_step >= self.max_step:
            self.local_step = 0
            self.max_step *= self.restart_mult
            self.total_restart += 1

        lr_mult = (1 + math.cos(math.pi * self.local_step / self.max_step))/2 / (self.total_restart + 1)

        for i in range(len(self.param_groups)):
            self.param_groups[i]['lr'] = self.__initial_lr[i] * lr_mult

        super().step(closure)

        self.local_step += 1
        self.global_step += 1


def split_dataset(dataset, n_splits):

    return [Subset(dataset, np.arange(i, len(dataset), n_splits)) for i in range(n_splits)]


def gap2d(x, keepdims=False):
    out = torch.mean(x.view(x.size(0), x.size(1), -1), -1)
    if keepdims:
        out = out.view(out.size(0), out.size(1), 1, 1)

    return out


def decode_seg(label_mask, toTensor=False):
    """
    :param label_mask: mask (np.ndarray): (M, N)/  tensor: N*C*H*W
    :return: color label: (M, N, 3),
    """
    if not isinstance(label_mask, np.ndarray):
        if isinstance(label_mask, torch.Tensor):  # get the data from a variable
            image_tensor = label_mask.data
        else:
            return label_mask
        label_mask = image_tensor[0][0].cpu().numpy()

    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3),dtype=np.float)
    r = label_mask % 6
    g = (label_mask % 36) // 6
    b = label_mask // 36
    # 归一化到[0-1]
    rgb[:, :, 0] = r / 6
    rgb[:, :, 1] = g / 6
    rgb[:, :, 2] = b / 6
    if toTensor:
        rgb = torch.from_numpy(rgb.transpose([2,0,1])).unsqueeze(0)

    return rgb


def tensor2im(input_image, imtype=np.uint8, normalize=True):
    """"Converts a Tensor array into a numpy image array.
    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        # if image_numpy.shape[0] == 1:  # grayscale to RGB
        #     image_numpy = np.tile(image_numpy, (3, 1, 1))
        if image_numpy.shape[0] == 3:  # if RGB
            image_numpy = np.transpose(image_numpy, (1, 2, 0))
            if normalize:
                image_numpy = (image_numpy + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


def tensor2np(input_image, if_normalize=True):
    """
    :param input_image: C*H*W / H*W
    :return: ndarray, H*W*C / H*W
    """
    if isinstance(input_image, torch.Tensor):  # get the data from a variable
        image_tensor = input_image.data
        image_numpy = image_tensor.cpu().float().numpy()  # convert it into a numpy array

    else:
        image_numpy = input_image
    if image_numpy.ndim == 2:
        return image_numpy
    elif image_numpy.ndim == 3:
        C, H, W = image_numpy.shape
        image_numpy = np.transpose(image_numpy, (1, 2, 0))
        #  如果输入为灰度图C==1，则输出array，ndim==2；
        if C == 1:
            image_numpy = image_numpy[:, :, 0]
        if if_normalize and C == 3:
            image_numpy = (image_numpy + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
            #  add to prevent extreme noises in visual images
            image_numpy[image_numpy<0]=0
            image_numpy[image_numpy>255]=255
            image_numpy = image_numpy.astype(np.uint8)
    return image_numpy


import ntpath
from misc.imutils import save_image
def save_visuals(visuals, img_dir, name, save_one=True, iter='0'):
    """
    """
    # save images to the disk
    for label, image in visuals.items():
        N = image.shape[0]
        if save_one:
            N = 1
        # 保存各个bz的数据
        for j in range(N):
            name_ = ntpath.basename(name[j])
            name_ = name_.split(".")[0]
            # print(name_)
            image_numpy = tensor2np(image[j], if_normalize=True).astype(np.uint8)
            # print(image_numpy)
            img_path = os.path.join(img_dir, iter+'_%s_%s.png' % (name_, label))
            save_image(image_numpy, img_path)