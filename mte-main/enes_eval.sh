CUDA_VISIBLE_DEVICES=0 python eval.py  \
        --config_path="./default_config_base_iwsltenes_gpu.yaml" \
        --distribute="false" \
        --epoch_size=128 \
        --device_target=GPU \
        --enable_save_ckpt="true" \
        --enable_lossscale="true" \
        --do_shuffle="false" \
        --checkpoint_path="/code/mte/enes_ckpt_0/transformer_2-11_10396.ckpt" \
        --save_checkpoint_steps=2500 \
        --save_checkpoint_num=60 \
        --save_checkpoint_path=./ \
        --data_path=./data/es_data/test-l128-mindrecord
#