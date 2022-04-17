# job_name=debug
# job_name=vit_editor_midCh_d6_wx3_1e-2
# job_name=iaa/vit_iaa_colorizer_train0
python tools/train_script/preference_vector.py \
	--log-level INFO \
	--log-freq 64 \
	--epochs 20 \
	--lr 1e-4 \
	--batchSize 4 \
	--device cuda:2 \
	--realbatchSize 64 \
	--target 'B' \
	# --model-path /media/song/ImageEnhancingResults/weights/${job_name} \
	# --summary-path /media/song/ImageEnhancingResults/summaries/${job_name} \
