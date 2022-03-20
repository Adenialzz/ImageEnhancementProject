CUDA_LAUNCH_BLOCKING=1 python train_can_cnn.py \
	--log-freq 100 \
	--lr 1e-3 \
	--batchSize 64 \
	--device cuda:0 \
	--model-path /media/song/ImageEnhancingResults/model/can_cnn \
	--summary-path /media/song/ImageEnhancingResults/summaries/can_cnn \
