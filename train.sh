CUDA_LAUNCH_BLOCKING=1 python train_can_cnn.py \
	--log-level INFO \
	--log-freq 50 \
	--lr 1e-2 \
	--batchSize 48 \
	--device cuda:0 \
	--model-path /media/song/ImageEnhancingResults/model/vit_editor_tokens_lr1e-2 \
	--summary-path /media/song/ImageEnhancingResults/summaries/vit_editor_tokens_1e-2 \
