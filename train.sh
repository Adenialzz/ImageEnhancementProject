CUDA_LAUNCH_BLOCKING=1 python train_can_cnn.py \
	--log-level INFO \
	--log-freq 1 \
	--lr 1e-2 \
	--batchSize 48 \
	--device cuda:0 \
	--model-path /media/song/ImageEnhancingResults/model/can_cnn \
	# --summary-path /media/song/ImageEnhancingResults/summaries/can_cnn \
