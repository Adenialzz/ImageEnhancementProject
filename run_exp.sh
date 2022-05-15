experts=ICFGH
arch=resnet34
lr=1e-2
job_name=train_pv/triplet_${experts}_lr${lr}_${arch}_depc_rand
python tools/train_script/preference_vector.py \
	--log-level INFO \
	--log-freq 1 \
	--epochs 100 \
	--lr ${lr} \
	--arch ${arch} \
	--batchSize 64 \
	--device cuda:0 \
	--realbatchSize 1 \
	--experts ${experts} \
	--arch ${arch} \
	--margin 0.2 \
	--pretrained \
	--feat_dim 512 \
	--use_depc_trainer \
	--model-path /media/song/ImageEnhancingResults/weights/${job_name} \
	--summary-path /media/song/ImageEnhancingResults/summaries/${job_name} \
	# --same_image \
