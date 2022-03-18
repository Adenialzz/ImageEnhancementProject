CUDA_LAUNCH_BLOCKING=1 python train_can_cnn.py \
	--batchSize 32 \
	--device cuda:0 \
	--model-path /media/song/ImageEnhancingResults/weights/can_vit \
	--summary-path /media/song/ImageEnhancingResults/summaries/can_vit \
	--lr 1e-6 \
