lr=1e-2
job_name=classification_lr${lr}_tinet
python tools/train_script/classification.py \
	--log-level INFO \
	--log-freq 1 \
	--epochs 100 \
	--lr ${lr} \
	--batchSize 256 \
	--device cuda:1 \
	--n_classes 9 \
	--model-path classification_results/models/${job_name} \
	--summary-path classification_results/summaries/${job_name} \
