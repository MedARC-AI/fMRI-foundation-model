OUTPUT_DIR='output/test_agnostic_2'

OMP_NUM_THREADS=1 python -m torch.distributed.launch --use-env --nproc_per_node=1 --master_port=29501 \
        run_mae_pretraining.py \
        --mask_type random \
        --mask_ratio 0.9 \
        --model pretrain_fmrimae_small_patch13 \
        --decoder_depth 4 \
        --batch_size 200 \
        --data_buffer_size 10000 \
        --opt adamw \
        --opt_betas 0.9 0.95 \
        --warmup_epochs 40 \
        --save_ckpt_freq 20 \
        --epochs 801 \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR}