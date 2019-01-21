import torch.nn as nn
import torch
import math
import time
import torch.utils.model_zoo as model_zoo
from utils import BasicBlock, Bottleneck, BBoxTransform, ClipBoxes
from features import PyramidFeaturesEx, PyramidFeatures, ResNetBody
from regression import RegressionModel, ClassificationModel
from anchors import Anchors
import losses
import torch.nn.functional as F
import numpy as np
import sys
sys.path.insert(0, '/data/ssy/detect_steel_bar/retinanet/lib')
#from models.roi_layers import nms
from models.roi_layers import nms

'''Non maximum suppression.
    Args:
      bboxes: (tensor) bounding boxes, sized [N,4].
      scores: (tensor) bbox scores, sized [N,].
      threshold: (float) overlap threshold.
      mode: (str) 'union' or 'min'.
    Returns:
      keep: (tensor) selected indices.
    Reference:
      https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/nms/py_cpu_nms.py
'''
'''
def nms(bboxes, scores, threshold=0.5, mode='union'):
    
    x1 = bboxes[:,:,0]
    y1 = bboxes[:,:,1]
    x2 = bboxes[:,:,2]
    y2 = bboxes[:,:,3]    

    scores = scores[:,:,0].numpy()
    areas = (x2-x1+1) * (y2-y1+1)
    #_, order = scores.sort(0, descending=True)
    #_, order = torch.argsort(scores, dim=0, descending=True)
    test = scores.argsort()[::-1][0]
    #for item in test:
    #    print(item)
    order = torch.from_numpy(test)
    #order = order[:,0]
    #print(bboxes.shape,order.shape, x1[:,0].item())
    #print(scores)

    keep = []
    while order.numel() > 0:
        i = order[0]        
        keep.append(i)

        if order.numel() == 1:
            break

        xx1 = x1[:,order[1:]].clamp(min=x1[:,i].item())
        yy1 = y1[:,order[1:]].clamp(min=y1[:,i].item())
        xx2 = x2[:,order[1:]].clamp(max=x2[:,i].item())
        yy2 = y2[:,order[1:]].clamp(max=y2[:,i].item())

        w = (xx2-xx1+1).clamp(min=0)
        h = (yy2-yy1+1).clamp(min=0)
        inter = w*h

        if mode == 'union':
            ovr = inter / (areas[:,i] + areas[:,order[1:]] - inter)
        elif mode == 'min':
            ovr = inter / areas[:,order[1:]].clamp(max=areas[i])
        else:
            raise TypeError('Unknown nms mode: %s.' % mode)

        ids = (ovr<=threshold).nonzero().squeeze()
        print(areas.shape, order.shape, order[1:].shape, x1.shape, ids[0,:])
        if ids.numel() == 0:
            break
        order = order[ids[0,:]+1]
    return torch.LongTensor(keep)
'''

'''
#def nms(dets, scores, thresh):
##def nms(dets, thresh):
#    dets = dets.numpy()
#    #dets = np.array(dets, np.dtype='float16')
#    x1 = dets[:, :, 0]
#    y1 = dets[:, :, 1]
#    x2 = dets[:, :, 2]
#    y2 = dets[:, :, 3]
#    #scores = dets[:, :, 4]   
#    scores = scores.numpy()
#    #scores = np.array(scores, np.dtype='float16')
#    #scores = scores[]
#
#    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
#    order = scores.argsort()[::-1]
#    order = order[0,:,0]
#    #print(x1, y1, x2, y2)
#
#    keep = []
#    while order.size > 0:
#        i = order.item(0)        
#        keep.append(i)
#        #print(i, x1.shape, scores.shape, order.shape, order.shape, x1[order[1:]])
#        xx1 = np.maximum(x1[:, i], x1[order[1:]])
#        yy1 = np.maximum(y1[:, i], y1[order[1:]])
#        xx2 = np.maximum(x2[:, i], x2[order[1:]])
#        yy2 = np.maximum(y2[:, i], y2[order[1:]])        
#
#        w = np.maximum(0.0, xx2 - xx1 + 1)
#        h = np.maximum(0.0, yy2 - yy1 + 1)
#        inter = w * h
#        ovr = inter / (areas[i] + areas[order[1:]] - inter)
#        #print(order.size, xx1, yy1, xx2, yy2, ovr, thresh)
#
#        inds = np.where(ovr <= thresh)[0]
#        order = order[inds + 1]
#
#    return torch.LongTensor(keep)
'''

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

class RetinaNet(nn.Module):
    def __init__(self, num_classes, block, layers):
        super(RetinaNet, self).__init__()
        self.ResNetBody = ResNetBody(num_classes, block, layers)
        self.regressionModel = RegressionModel(256, num_anchors=15)
        self.classificationModel = ClassificationModel(256, num_anchors=15, num_classes=num_classes)
        self.anchors = Anchors()
        self.regressBoxes = BBoxTransform()
        self.clipBoxes = ClipBoxes()        
        self.focalLoss = losses.FocalLoss()                
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        prior = 0.01
        
        self.classificationModel.output.weight.data.fill_(0)
        self.classificationModel.output.bias.data.fill_(-math.log((1.0-prior)/prior))

        self.regressionModel.output.weight.data.fill_(0)
        self.regressionModel.output.bias.data.fill_(0)
        
    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def forward(self, inputs):

        if self.training:
            img_batch, annotations = inputs
        else:
            img_batch = inputs
            
        features = self.ResNetBody(img_batch)

        regression = torch.cat([self.regressionModel(feature) for feature in features], dim=1)

        classification = torch.cat([self.classificationModel(feature) for feature in features], dim=1)

        anchors = self.anchors(img_batch)

        if self.training:
            return self.focalLoss(classification, regression, anchors, annotations)
        else:
            transformed_anchors = self.regressBoxes(anchors, regression)
            transformed_anchors = self.clipBoxes(transformed_anchors, img_batch)

            scores = torch.max(classification, dim=2, keepdim=True)[0]

            scores_over_thresh = (scores>0.05)[0, :, 0]

            if scores_over_thresh.sum() == 0:
                # no boxes to NMS, just return
                return [torch.zeros([1]).cuda(0), torch.zeros([1]).cuda(0), torch.zeros([1, 4]).cuda(0)]
                #return [torch.zeros(0), torch.zeros(0), torch.zeros(0, 4)]

            classification = classification[:, scores_over_thresh, :]
            transformed_anchors = transformed_anchors[:, scores_over_thresh, :]
            scores = scores[:, scores_over_thresh, :]
            anchors_nms_idx = nms(transformed_anchors[0,:,:], scores[0,:,:].squeeze(1), 0.25)
            nms_scores, nms_class = classification[0, np.array(anchors_nms_idx.cpu(), dtype=float), :].max(dim=1)

            return [nms_scores, nms_class, transformed_anchors[0, anchors_nms_idx, :]]

class FPN(nn.Module):
    '''
    author: syshen
    '''
    def __init__(self, classes, pretrain=False, align=False):
        super(FPN, self).__init__()        
        self.classes = classes
        self.rpn_cls_loss = 0
        self.rpn_bbox_loss = 0
        self.Backstone = pvaHyper()
        self.features = self.Backstone
        self.rpn_regression = rpn_regression()
        self.proposallayer = ProposalLayer(cfg.FEAT_STRIDE[0], \
                                           cfg.ANCHOR_SCALES, cfg.ANCHOR_RATIOS)
        self.proposaltargetlayer = ProposalTargetLayer()
        self.roi_extraction = ROIAlignLayer()
        if not align:
            self.roi_extraction = ROIPoolingLayer()
        if pretrain:
            model = torch.load(cfg.TRAIN.pretainmodel)
            self.Backstone = copy.deepcopy(model.state_dict())

        self.regressionDim = 512
        self.roi_size = 6

        self.Regression = nn.Sequential(OrderedDict([
            ('fc1',nn.Linear(self.regressionDim * self.roi_size * self.roi_size, self.regressionDim)),
            ('fc_relu', nn.LeakyReLU(inplace=True)),
            ('fc2', nn.Linear(self.regressionDim, self.regressionDim, bias=True)),
            ('fc2_relu', nn.LeakyReLU(inplace=True))]))

        self.cls_inner = nn.Sequential(OrderedDict([
            ('fc1',nn.Linear(self.regressionDim, self.n_classes))]))

        self.bbox_pred = nn.Sequential(OrderedDict([
            ('fc1',nn.Linear(self.regressionDim, self.n_classes * 4))]))

    def forward(self, base_feat, im_data, im_info, gt_boxes, num_boxes, training=False):
        batch_size = im_data.size(0)
        im_info = im_info.data
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data
        features = self.Backstone(im_data)
        base_feat, rpn_cls_score, rpn_bbox_pred, rpn_loss_cls, rpn_loss_bbox = \
            self.rpn_regression.forward(features, im_info, gt_boxes, num_boxes)
        if training:
            rois = self.proposallayer.forward(rpn_cls_score.data, \
                                              rpn_bbox_pred.data, im_info)
            rois_label, rois_batch, rois_target, bbox_inside_weights = \
                self.proposaltargetlayer(rois, gt_boxes, num_boxes)
            bbox_outside_weights = np.array(bbox_inside_weights > 0).astype(np.float32)
            rois_label = Variable(rois_label.view(-1).long())
            rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
            bbox_inside_weights = Variable(bbox_inside_weights.\
                                           view(-1, bbox_inside_weights.size(2)))
            bbox_outside_weights = Variable(bbox_outside_weights.\
                                            view(-1, bbox_outside_weights.size(2)))
        else:
            rois_label = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None
            rpn_loss_cls = 0
            rpn_loss_bbox = 0

        rois = Variable(rois)
         # do roi pooling based on predicted rois
        roi_feat = self.roi_extraction(base_feat, rois.view(-1, 5), \
                                          self.roi_size, cfg.TRAIN.spatial_scale)
        inner_product = self.Regression(roi_feat)
        bbox_pred = self.bbox_pred(inner_product)
        cls_inner = self.cls_inner(inner_product)
        cls_prob = F.softmax(cls_inner, 1)

        loss_cls  = 0
        loss_bbox = 0

        if training:
            loss_cls = nn.CrossEntropyLoss()
            loss_bbox = smooth_l1_loss(bbox_pred, rois_target, bbox_inside_weights, \
                                       bbox_outside_weights)

        cls_prob = cls_prob.view(batch_size, rois.size(1), -1)
        bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)

        return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, loss_cls, loss_bbox

def resnet18(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = RetinaNet(num_classes, BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18'], model_dir='/home/user/.torch/models/'), strict=False)
    return model

def resnet34(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = RetinaNet(num_classes, BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34'], model_dir='/home/user/.torch/models/'), strict=False)
    return model

def resnet50(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = RetinaNet(num_classes, Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50'], model_dir='/home/user/.torch/models/'), strict=False)
    return model

def resnet101(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = RetinaNet(num_classes, Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101'], model_dir='/home/user/.torch/models/'), strict=False)
    return model

def resnet152(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = RetinaNet(num_classes, Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152'], model_dir='/home/user/.torch/models/'), strict=False)
    return model

