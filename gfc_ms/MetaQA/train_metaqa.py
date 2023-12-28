import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import sys
# sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__)))) 
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../'))) 
# path_abs = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
path_abs = os.path.abspath(os.path.join(os.getcwd(), '../'))
print(path_abs)
import mindspore
import mindspore.nn as nn
import argparse
import shutil
from tqdm import tqdm
import numpy as np
import time
from utils.misc import MetricLogger
from utils.lr_scheduler import get_linear_schedule_with_warmup
from MetaQA.data_hop_final import load_data
from MetaQA.model_metaqa import GFC
from MetaQA.predict_f1_hop import validate
import logging
from mindspore.train import Model
from mindspore import dtype as mstype
import warnings
import mindspore as ms

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s')
logFormatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
rootLogger = logging.getLogger()
import mindspore.dataset as ds

import setproctitle

setproctitle.setproctitle("GFC_metaqa")

from mindspore import context
import mindspore.context as ms_ctx
# context.set_context(mode=ms.PYNATIVE_MODE, device_target="GPU")  #context.GRAPH_MODE
ms.set_context(mode=ms.PYNATIVE_MODE, pynative_synchronize=True,device_target="GPU", graph_kernel_flags="–opt_level=2")
def train(args):
    # 设置设备为GPU（CUDA）或CPU
    # path_abs = '/data/xmh/Projects/Pytorch/GTA'
    path_abs = '/sdb/xmh/Projects/Pytorch/GTA'
    input_dir = path_abs + '/' + args.input_dir
    logging.info(input_dir)
    ent2id, rel2id, triples, train_loader, val_loader = load_data(input_dir, args.bert_name, args.batch_size)
    # logging.info(len(train_loader.data))
    logging.info("Create model.........")
    model = GFC(args, ent2id, rel2id, triples)
    if not args.ckpt == None:
        model.load_state_dict(mindspore.load_checkpoint(args.ckpt))
    # t_total = len(train_loader) * args.num_epoch

    import mindspore.nn as nn
    from mindspore import Parameter, Tensor

    no_decay = ["bias", "LayerNorm.weight"]

    bert_param = list(filter(lambda x: 'bert_encoder' in x.name, model.trainable_params()))
    other_param = list(filter(lambda x: 'bert_encoder' not in x.name, model.trainable_params()))
    print('number of bert param: {}'.format(len(bert_param)))

    optimizer_grouped_parameters = [{'params': bert_param, 'weight_decay': 0.01, 'lr':args.bert_lr}, {'params': other_param, 'weight_decay': 0.01, 'lr': args.lr}]

    optimizer = nn.Adam(optimizer_grouped_parameters)

    meters = MetricLogger(delimiter="  ")
    logging.info("Start training........")

    def forward_fn(data0,data1,data2,data3):
        loss = model.construct(data0, data1, data2, data3)
        logits = 0
        return loss, logits

    grad_fn = mindspore.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)

    def train_step(data):
        (loss, _), grads = grad_fn(data[0], data[1], data[2], data[3])
        optimizer(grads)
        return loss

    def train_loop(dataset):
        iteration = 0
        # logging.info(len(dataset))
        for data in tqdm(dataset, total=len(dataset)):  # .create_tuple_iterator()
            iteration += 1
            loss = train_step(data)

    for epoch in range(20):
        logging.info(f"Epoch {epoch + 1}\n-------------------------------")  # 待删除
        train_loop(train_loader)
        # logging.info("One epoch finished!")
        if epoch % 1 == 0:
            p_info = validate(model, train_loader, epoch+1)
            logging.info(p_info)

    mindspore.save_checkpoint(model, "metaqa.ckpt")
    logging.info("Saved Model to model.ckpt")
            

def main():
    parser = argparse.ArgumentParser()
    # input and output
    parser.add_argument('--input_dir', required=True, help='path to the data')
    parser.add_argument('--save_dir', required=True, help='path to save checkpoints and logs')
    parser.add_argument('--ckpt', default=None)
    # training parameters
    parser.add_argument('--bert_lr', default=3e-5, type=float)
    parser.add_argument('--lr', default=1e-5, type=float)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--num_epoch', default=1, type=int)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--seed', type=int, default=444, help='random seed')
    parser.add_argument('--opt', default='radam', type=str)
    parser.add_argument('--warmup_proportion', default=0.05, type=float)
    # model parameters
    parser.add_argument('--bert_name', default='bert-base-uncased', choices=['roberta-base', 'bert-base-uncased'])
    parser.add_argument('--aux_hop', type=int, default=1, choices=[0, 1],
                        help='utilize question hop to constrain the probability of self relation')
    args = parser.parse_args()

    # make logging.info display into both shell and file
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    time_ = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
    args.log_name = time_ + '_{}_{}_{}.log'.format(args.opt, args.lr, args.batch_size)
    fileHandler = logging.FileHandler(os.path.join(args.save_dir, args.log_name))
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)
    # args display
    for k, v in vars(args).items():
        logging.info(k + ':' + str(v))

    # set random seed
    mindspore.set_seed(args.seed)
    np.random.seed(args.seed)

    train(args)


if __name__ == '__main__':
    main()
