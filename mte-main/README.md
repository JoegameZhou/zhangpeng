### 数据集

https://data.statmt.org/opus-100-corpus/v1.0/supervised/

### 环境

```shell
pip install -r requirements.txt
```



### 语素分割

```shell
python get_vocab_morphSeg_iwsltdeen.py
```



### 构建ms数据

```shell
python create_data.py --input_file ./data/iwslt14.de-en.bpe10000/train.all --output_file ./data/iwslt_deen_ms_share/deen-l128-mindrecord --max_seq_length 256 --vocab_file ./data/iwslt14.de-en.bpe10000/vocab_deen.txt --config_path /code/mte/default_config_large_train.yaml

python create_data.py --input_file ./data/iwslt14.de-en.bpe10000/test.all --output_file ./data/data/test-l128-mindrecord --num_splits 1 --max_seq_length 512 --clip_to_max_len True --vocab_file ./data/iwslt14.de-en.bpe10000/vocab_deen.txt --config_path /code/mte/default_config_large.yaml

python create_data.py --input_file ./data/opus/enit/test.all --output_file ./data/it_data/test-l128-mindrecord --num_splits 1 --max_seq_length 512 --clip_to_max_len True --vocab_file ./data/opus/enit/vocab_enit.txt --config_path /code/mte/default_enit_config_large.yaml

python create_data.py --input_file ./data/opus/enru/test.all --output_file ./data/ru_data/test-l128-mindrecord --num_splits 1 --max_seq_length 512 --clip_to_max_len True --vocab_file ./data/opus/enru/vocab_enru.txt --config_path /code/mte/default_enru_config_large.yaml

python create_data.py --input_file ./data/opus/enes/test.all --output_file ./data/es_data/test-l128-mindrecord --num_splits 1 --max_seq_length 512 --clip_to_max_len True --vocab_file ./data/opus/enes/vocab_enes.txt --config_path /code/mte/default_enes_config_large.yaml
```



### 训练 on GPU

```shell
CUDA_VISIBLE_DEVICES=0 python train.py  \
        --config_path="./default_config_base_iwsltdeen_gpu.yaml" \
        --distribute="false" \
        --epoch_size=128 \
        --device_target=GPU \
        --enable_save_ckpt="true" \
        --enable_lossscale="true" \
        --do_shuffle="true" \
        --checkpoint_path="" \
        --save_checkpoint_steps=2500 \
        --save_checkpoint_num=60 \
        --save_checkpoint_path=./ \
        --data_path=./data/iwslt_deen_ms_share/deen-l128-mindrecord
```

###测试
python post_process.py

bash scripts/process_output.sh ./processed_test.en ./processed_output_eval.txt ./data/iwslt14.de-en.bpe10000/vocab_deen.txt

bash scripts/process_output.sh ./test.en ./output_eval.txt ./data/iwslt14.de-en.bpe10000/vocab_deen.txt

bash scripts/process_output.sh ./test.it ./output_eval.txt ./data/opus/enit/vocab_enit.txt

bash scripts/process_output.sh ./test.es ./output_eval.txt ./data/opus/enes/vocab_enes.txt



perl bleu.perl ./test.en.forbleu < ./output_eval.txt.forbleu

perl bleu.perl ./test.it.forbleu < ./output_eval.txt.forbleu

perl bleu.perl ./test.es.forbleu < ./output_eval.txt.forbleu

