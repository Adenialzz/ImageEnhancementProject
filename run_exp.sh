# job_name=debug
# job_name=vit_editor_midCh_1e-1
job_name=iaa/vit_iaa_from_scratch_train09
python train_iaa.py \
	--log-level DEBUG \
	--log-freq 1 \
	--lr 5e-3 \
	--batchSize 48 \
	--device cuda:2 \
	--model-path /media/song/ImageEnhancingResults/weights/${job_name} \
	--summary-path /media/song/ImageEnhancingResults/summaries/${job_name} \
