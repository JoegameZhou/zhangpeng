CUDA_VISIBLE_DEVICES=0 python train.py  \
        --config_path="./default_config_base_iwsltdeen_gpu_train.yaml" \
        --distribute="false" \
        --epoch_size=128 \
        --device_target=GPU \
        --enable_save_ckpt="true" \
        --enable_lossscale="true" \
        --do_shuffle="true" \
        --checkpoint_path="/code/mte/ckpt_0/transformer_7-57_1663.ckpt" \
        --save_checkpoint_steps=2500 \
        --save_checkpoint_num=60 \
        --save_checkpoint_path=./ \
        --data_path=./data/128_iwslt_deen_ms_share/deen-l128-mindrecord