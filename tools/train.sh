CUDA_LAUNCH_BLOCKING=1 python train_can_cnn.py \
	--log-freq 1000 \
	--batchSize 16 \
	--device cuda:0 \
	--model-path /media/song/ImageEnhancingResults/model/can_vit \
	--summary-path /media/song/ImageEnhancingResults/summaries/can_vit \
	--lr 1e-6 \
