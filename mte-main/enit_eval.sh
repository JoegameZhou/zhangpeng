CUDA_VISIBLE_DEVICES=0 python eval.py  \
        --config_path="./default_config_base_iwsltenit_gpu.yaml" \
        --distribute="false" \
        --epoch_size=128 \
        --device_target=GPU \
        --enable_save_ckpt="true" \
        --enable_lossscale="true" \
        --do_shuffle="false" \
        --checkpoint_path="/code/mte/enit_ckpt_0/transformer_2-10_10400.ckpt" \
        --save_checkpoint_steps=2500 \
        --save_checkpoint_num=60 \
        --save_checkpoint_path=./ \
        --data_path=./data/it_data/test-l128-mindrecord
#