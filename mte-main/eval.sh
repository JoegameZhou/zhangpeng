CUDA_VISIBLE_DEVICES=0 python eval.py  \
        --config_path="./default_config_base_iwsltdeen_gpu.yaml" \
        --distribute="false" \
        --epoch_size=128 \
        --device_target=GPU \
        --enable_save_ckpt="true" \
        --enable_lossscale="true" \
        --do_shuffle="false" \
        --checkpoint_path="/code/mte/deen_ckpt_0/transformer_8-63_1663.ckpt" \
        --save_checkpoint_steps=2500 \
        --save_checkpoint_num=60 \
        --save_checkpoint_path=./ \
        --data_path=./data/data/test-l128-mindrecord
#