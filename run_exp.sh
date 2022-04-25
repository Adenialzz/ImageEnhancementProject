# job_name=debug
# job_name=vit_editor_midCh_d6_wx3_1e-2
experts=IAB
job_name=train_pv/triplet_${experts}_exp1
python tools/train_script/preference_vector.py \
	--log-level INFO \
	--log-freq 64 \
	--epochs 100 \
	--lr 1e-4 \
	--batchSize 1 \
	--device cuda:2 \
	--realbatchSize 64 \
	--experts ${experts} \
	--model-path /media/song/ImageEnhancingResults/weights/${job_name} \
	--summary-path /media/song/ImageEnhancingResults/summaries/${job_name} \
