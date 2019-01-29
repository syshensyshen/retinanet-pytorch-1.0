functions:
retinanet using pytorch-1.0
add multi-scales image train
add panet style pyramid features

`pytorch1.0`,`opencv-python`,`skimage`

step1: install
git clone https://github.com/syshensyshen/retinanet-pytorch-1.0.git

'cd retinanet-pytorch-1.0/retinanet/lib/'
python setup.py build develop

step2: train
 CUDA_VISIBLE_DEVICES=1 python retinanet/main.py \
	--depth 152 --epochs 100 --dataset voc \
	--voc_train /data/ssy/VOCdevkit/VOC2012/ \
	--voc_val /data/ssy/VOCdevkit/test/
 

step3: predict
 python retinanet/predict.py
 
reference code:
 
[pytorch-retinanet](https://github.com/yhenon/pytorch-retinanet.git)
[pytorch-retinanet](https://github.com/kuangliu/pytorch-retinanet.git)
[pfaster-rcnn.pytorch](https://github.com/jwyang/faster-rcnn.pytorch.git)
[detect_steel_bar] (https://github.com/spytensor/detect_steel_bar.git)

