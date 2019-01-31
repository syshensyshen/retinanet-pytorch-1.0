CUDA_VISIBLE_DEVICES=0 python retinanet/trainval_net.py --dataset voc \
	--voc_train /data/ssy/VOCdevkit/VOC2012/ \
	--voc_val /data/ssy/VOCdevkit/test/ \
	--depth 152

