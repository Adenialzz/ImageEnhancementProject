mode=$1
experts=ICFGH
arch=vit_tiny_patch16_224
margin=0.2
lr=1e-2
job_name=train_pv/triplet_${experts}_m${margin}_lr${lr}_${arch}_rand

if [ $mode = debug ]; then
	summary_path=debug_summary/
	model_path=debug_saved_model/
else
	summary_path=/media/song/ImageEnhancingResults/summaries/${job_name}
	model_path=/media/song/ImageEnhancingResults/weights/${job_name}
fi

python tools/train_script/preference_vector.py \
	--log-level INFO \
	--log-freq 1 \
	--epochs 100 \
	--lr ${lr} \
	--arch ${arch} \
	--batchSize 16 \
	--device cuda:2 \
	--realbatchSize 1 \
	--experts ${experts} \
	--arch ${arch} \
	--margin ${margin} \
	--feat_dim 512 \
	--model-path ${model_path} \
	--summary-path ${summary_path} \
	# --pretrained \
	# --use_depc_trainer \
	# --same_image \
