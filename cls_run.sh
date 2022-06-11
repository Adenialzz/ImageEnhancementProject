mode=$1
lr=1e-2
dset=t_inet
job_name=classification_lr${lr}_dset${dset}_wSoftmax

if [ $mode = debug ]; then
	summary_path=debug_summary
	model_path=debug_saved_model
else
	summary_path=classification_results/summaries/${job_name}
	model_path=classification_results/models/${job_name}
fi

python tools/train_script/classification.py \
	--log-level INFO \
	--dataset ${dset} \
	--log-freq 1 \
	--epochs 100 \
	--lr ${lr} \
	--batchSize 256 \
	--device cuda:1 \
	--n_classes 9 \
	--model-path ${model_path} \
	--summary-path ${summary_path} \
