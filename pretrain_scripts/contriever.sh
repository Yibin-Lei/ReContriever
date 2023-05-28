rmin=0.05
rmax=0.5
T=0.05
QSIZE=131072
MOM=0.9995
POOL=average
AUG=delete
PAUG=0.1
LC=0.
CROP=normal
mo=bert-base-uncased
mp=none
localrank=$SLURM_LOCALID

name=$POOL-rmin$rmin-rmax$rmax-T$T-$QSIZE-$MOM-$mo-$AUG-$PAUG-$CROP

python train.py \
        --model_path $mp \
        --sampling_coefficient $LC \
        --retriever_model_id $mo --pooling $POOL \
        --augmentation $AUG --prob_augmentation $PAUG \
        --loading_mode split \
        --ratio_min $rmin --ratio_max $rmax --chunk_length 256 \
        --momentum $MOM --queue_size $QSIZE --temperature $T \
        --warmup_steps 20000 --total_steps 500000 --lr 0.00005 \
        --name $name \
        --scheduler linear \
        --optim adamw \
	      --local_rank $localrank \
        --per_gpu_batch_size 128 \
        --num_workers 30 \
        --save_freq 50000 \
        --output_dir /your_output_path \
        --crop_method $CROP \
        --contrastive_mode moco