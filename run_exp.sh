set -e -u
mode=$1
lr=1e-2
train_script=pv_editor.py
input_expert=C
target_experts=IFGH
pv_epoch=99
arch=can
job_name=train_pv1/${lr}_${input_expert}_${target_experts}_pvepoch${pv_epoch}_${arch}

if [ $mode = debug ]; then
	summary_path=debug_summary/
	model_path=debug_saved_model/
elif [ $mode = exp ]; then
	summary_path=/media/song/ImageEnhancingResults/summaries/${job_name}
	model_path=/media/song/ImageEnhancingResults/weights/${job_name}
else
	echo "Unknown mode: {$1}"
fi

python tools/train_script/${train_script} \
	--log-level INFO \
	--log-freq 1 \
	--epochs 100 \
	--arch ${arch} \
	--lr ${lr} \
	--batchSize 4 \
	--device cuda:0 \
	--input_expert ${input_expert} \
	--target_experts ${target_experts} \
	--pv_dir /media/song/ImageEnhancingResults/weights/train_pv/ \
	--pv_epoch ${pv_epoch} \
	--model-path ${model_path} \
	--summary-path ${summary_path} \
