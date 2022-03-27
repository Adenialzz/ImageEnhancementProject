CUDA_LAUNCH_BLOCKING=1 python train_can_cnn.py \
	--log-level INFO \
	--log-freq 50 \
	--lr 1e-1 \
	--batchSize 96 \
	--device cuda:0 \
	--model-path /media/song/ImageEnhancingResults/weights/vit_editor_channels_d6_lr1e-1 \
	--summary-path /media/song/ImageEnhancingResults/summaries/vit_editor_channels_d6_1e-1 \
