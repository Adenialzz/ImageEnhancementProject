dir=/media/song/ImageEnhancingResults/ \
tbx=vit_enhancer_channels_1e-3 \

python train_enhancer.py \
	--log-level DEBUG \
	--log-freq 50 \
	--lr 1e-3 \
	--batchSize 96 \
	--device cuda:2 \
	--model-path ${dir}weights/${tbx} \
	--summary-path ${dir}summaries/${tbx} \
