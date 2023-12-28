#!/bin/bash

 python generate_tnews_mindrecord.py --data_dir /sdb/nlp21/Project/LongDocClass/models-r2.0/dataset/20news/ \
                                        --task_name 20news \
                                        --vocab_file /sdb/nlp21/Project/LongDocClass/models-r2.0/bert-base-uncased/vocab.txt \
                                        --output_dir /sdb/nlp21/Project/LongDocClass/models-r2.0/dataset/20news/data_ms \
                                        --max_seq_length 1024