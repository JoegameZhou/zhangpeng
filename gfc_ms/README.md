# GFC_MS
直接运行
bash run.sh
# CWQ数据集
处理好的中间数据：processed_hop_fixed_ms_cwq.pt
https://drive.google.com/file/d/122BGFYTIvv0cPthg-nG6P2CgmfGilwKG/view?usp=drive_link
原始数据集地址：
https://drive.google.com/u/0/uc?id=1ua7h88kJ6dECih6uumLeOIV9a3QNdP-g&export=download
将数据集下载存放于某个data/CWQ/文件夹下，然后修改train.py里面的path_abs为data/的全局路径；

替换data.py里面的BertTokenizer加载目录为自己的加载目录；

运行命令：
python3 train.py --input_dir data/CWQ --save_dir checkpoints/CWQ_t —rev

# WebQSP数据集
中间数据数据：由于实验室服务器出现故障，正在修复，热切数据集很大，待上传；
原始数据：https://drive.google.com/drive/folders/1vqHTcDwfiZ47jxVbp0IDwQU_s-x-l9lZ?usp=drive_link
将数据集下载存放于某个data/WebQSP/文件夹下，然后修改train_hop_final.py里面的path_abs为data/的全局路径；

运行命令：
python3 train_hop_final.py --input_dir data/WebQSP --save_dir checkpoints/WebQSP/

# MetaQA数据集
中间数据集地址：processed_hop_ms_metaqa.pt: 
https://drive.google.com/file/d/1UL4rRCR0-bUL09SD7eJigV8-aE48O0QV/view?usp=drive_link
将数据集下载存放于某个data/CWQ/文件夹下，然后修改train_hop_final.py里面的path_abs为data/的全局路径；

运行命令：
python3 train_metaqa.py --input_dir data/MetaQA --save_dir checkpoints/MetaQA/
