# job_name=debug
# job_name=vit_editor_midCh_d6_wx3_1e-2
job_name=train_pv/triplet_IABCDE_exp0
python tools/train_script/preference_vector.py \
	--log-level INFO \
	--log-freq 64 \
	--epochs 30 \
	--lr 1e-4 \
	--batchSize 4 \
	--device cuda:2 \
	--realbatchSize 64 \
	--experts IABCDE \
	--same_image \
	# --model-path /media/song/ImageEnhancingResults/weights/${job_name} \
	# --summary-path /media/song/ImageEnhancingResults/summaries/${job_name} \
