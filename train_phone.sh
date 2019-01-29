CUDA_VISIBLE_DEVICES=1 python retinanet/main.py \
	--depth 101  --epochs 100 --dataset voc \
	--voc_train /data/ssy/front_parts/ \
	--voc_val /data/ssy/front_parts/
