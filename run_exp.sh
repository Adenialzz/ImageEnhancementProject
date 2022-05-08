experts=IABCDE
arch=resnet18
lr=1e-1
job_name=train_pv/triplet_${experts}_lr${lr}_${arch}_m0.3
python tools/train_script/preference_vector.py \
	--log-level INFO \
	--log-freq 1 \
	--epochs 100 \
	--lr ${lr} \
	--arch ${arch} \
	--batchSize 64 \
	--device cuda:2 \
	--realbatchSize 1 \
	--experts ${experts} \
	--arch ${arch} \
	--margin 0.3
	--pretrained \
	--model-path /media/song/ImageEnhancingResults/weights/${job_name} \
	--summary-path /media/song/ImageEnhancingResults/summaries/${job_name} \
	# --same_image \
