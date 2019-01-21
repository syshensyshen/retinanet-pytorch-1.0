
`pytorch1.0`,`opencv-python`,`skimage`

step1: 安装
git clone https://github.com/syshensyshen/retinanet-pytorch-1.0.git

'cd retinanet-pytorch-1.0/retinanet/lib/'
python setup.py build develop

step2: 训练
 CUDA_VISIBLE_DEVICES=1 python retinanet/main.py \
	--depth 152 --epochs 100 --dataset voc \
	--voc_train /data/ssy/VOCdevkit/VOC2012/ \
	--voc_val /data/ssy/VOCdevkit/test/
 

step3: 预测
 python retinanet/predict.py
 
 
参考
 
[pytorch-retinanet](https://github.com/yhenon/pytorch-retinanet.git)
[pytorch-retinanet](https://github.com/kuangliu/pytorch-retinanet.git)
[pfaster-rcnn.pytorch](https://github.com/jwyang/faster-rcnn.pytorch.git)

