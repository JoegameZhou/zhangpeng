# RTrans-ms

This is the code repository of paper Recurrent Transformers for Long Document Understanding (Mindspore version).


## 一、数据预处理-将数据处理成mindrecord的格式

原始的数据集和Bert预训练模型ckpts等文件，可以在以下[Google Drive](https://drive.google.com/drive/folders/1e9N3Ypx_-Ov5fTDa5GUHglVTZYuFdQMp?usp=drive_link)中下载。

```shell
cd ./src/generate_mindrecord
bash generate_finetune_mindrecords.sh
```
注意: 其中我们需要自己实现<kbd>generate_tnews_mindrecord.py</kbd>中的不同数据集的DataProcessor, 可以参考hyperpartition和20news等数据集的处理方法, 同时需要设置相关的参数, 例如: max_seq_length是设置每条样本最大的文本长度。


## 二、Train

GPU requirements: 1 V100 GPUs (32GB)即可

然后预处理好数据后, 可以开始进行微调, 不同的数据集需设置不同的评估指标、训练epoch、标签类别数、文本段长度。

```shell
# Train longdoc-encoder 
bash scripts/run_classifier_gpu.sh
```
