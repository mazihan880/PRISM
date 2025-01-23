import os
import random
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import logging
import time
import pickle
import json
from itertools import product
from utils import Data_Classifier, NewsRecommendationDataset
from component.guided_diffusion import Diffusion, ModelWithEmbeddingIB
from train import Trainer
from torch.utils import data

# 设置可见的GPU设备


# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--args_file', type=str, default='args.json', help='Path to the args JSON file')
parser.add_argument('--param_grid_file', type=str, default='param_grid.json', help='Path to the parameter grid JSON file')
parser.add_argument('--grid_search', action='store_true', help='Enable grid search')
parser.add_argument('--dataset', type=str, help='Dataset to use')
parser.add_argument('--log_file', type=str, help='Log file path')
parser.add_argument('--random_seed', type=int, help='Random seed')
parser.add_argument('--max_len', type=int, help='The max length of sequence')
parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], help='Device to use')
parser.add_argument('--num_gpu', type=int, help='Number of GPU')
parser.add_argument('--epochs', type=int, help='Number of max epochs')
parser.add_argument('--batch_size', type=int, help='Batch size')
parser.add_argument('--layers', type=int, help='GRU layers')
parser.add_argument('--hidden_factor', type=int, help='Number of hidden factors, i.e., embedding size')
parser.add_argument('--timesteps', type=int, help='Timesteps for diffusion')
parser.add_argument('--beta_end', type=float, help='Beta end of diffusion')
parser.add_argument('--beta_start', type=float, help='Beta start of diffusion')
parser.add_argument('--lr', type=float, help='Learning rate')
parser.add_argument('--lr_cls', type=float, help='Learning rate')
parser.add_argument('--l2_decay', type=float, help='L2 loss reg coef')
parser.add_argument('--cuda', type=int, help='CUDA device')
parser.add_argument('--dropout_rate', type=float, help='Dropout rate')
parser.add_argument('--w', type=float, help='Diffusion noise scale')
parser.add_argument('--p', type=float, help='Dropout probability')
parser.add_argument('--report_epoch', type=bool, help='Report frequency')
parser.add_argument('--cls_optimizer', type=str, help='Type of optimizer')
parser.add_argument('--diff_optimizer', type=str, help='Type of optimizer')
parser.add_argument('--beta_sche', type=str, help='Beta scheduling scheme')
parser.add_argument('--descri', type=str, help='Description of the work')
parser.add_argument('--input_dim', type=int, help='News content dim')
parser.add_argument('--hidden_dim', type=int, help='Model hidden layer dimension')
parser.add_argument('--bottleneck_dim', type=int, help='Model ottleneck layer dimension')
parser.add_argument('--eval_epoch', type=int, help='Epochs to run evaluation')
parser.add_argument('--save_epoch', type=int, help='Epochs to save checkpoints')
parser.add_argument('--decay_step', type=int, default=100, help='Decay step for StepLR')
parser.add_argument('--gamma', type=float, default=0.1, help='Gamma for StepLR')
parser.add_argument('--class_epoch', type=int, help='epoch for class')
parser.add_argument('--fusion_mode', type=str,help='fusion mode:clsattn, seqattn')
parser.add_argument('--model_dir', type=str, help='path to save model')
parser.add_argument('--phi', type=float, default=0.8, help='phi for ot distance')
parser.add_argument('--tau', type=float, default=0.4, help='tau for constractive loss')
parser.add_argument('--test_mode', type=int, default=0, help='0 for train and 1 for test')
parser.add_argument('--expname', type=str, help='Name for Experiment')
parser.add_argument('--diff_cof', type=float, help='Diffu cof')

args = parser.parse_args()

# 创建日志文件夹
def create_log_dir(log_file, dataset):
    if not os.path.exists(log_file):
        os.makedirs(log_file)
    if not os.path.exists(log_file + dataset):
        os.makedirs(log_file + dataset)

# 日志配置
def setup_logging(log_file, dataset):
    logging.basicConfig(
        level=logging.INFO, 
        filename=log_file + dataset + '/' + time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()) + '.log',
        datefmt='%Y/%m/%d %H:%M:%S', 
        format='%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(module)s - %(message)s', 
        filemode='w'
    )
    logger = logging.getLogger(__name__)
    return logger

# 设置随机种子确保结果可复现
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

def run_experiment(args, logger):

    # 加载数据集
    tr_path = f'../../dataset/data/{args.dataset}/user_train.json'
    val_path = f'../../dataset/data/{args.dataset}/user_val.json'
    test_path = f'../../dataset/data/{args.dataset}/user_test.json'
    
    
    
    
    tr_embedding = f'../../dataset/data/{args.dataset}/embedding_train.pkl'
    val_embedding = f'../../dataset/data/{args.dataset}/embedding_val.pkl'
    test_embedding = f'../../dataset/data/{args.dataset}/embedding_test.pkl'
    
    
    train_label = f'../../dataset/data/{args.dataset}/news_labels_train.json'
    val_label = f'../../dataset/data/{args.dataset}/news_labels_val.json'
    test_label = f'../../dataset/data/{args.dataset}/news_labels_test.json'
    
    
    cls_path = f'../../dataset/data/{args.dataset}/cls_data.pkl'
    
    
    with open(tr_embedding, 'rb') as f:
        tr_embeddings  = pickle.load(f)["embedding"]
    with open(val_embedding, 'rb') as f:
        val_embeddings  = pickle.load(f)["embedding"]
    with open(test_embedding, 'rb') as f:
        test_embeddings  = pickle.load(f)["embedding"]

    with open(val_label, 'r') as f:
        label4news_val = json.load(f)    
        
        
    with open(cls_path, 'rb') as f:
        cls_data = pickle.load(f)
    
    if args.test_mode == 0:
        with open(tr_path, 'r') as f:
            tr_raw = json.load(f)
        with open(val_path, 'r') as f:
            val_raw = json.load(f)
        train_ds = NewsRecommendationDataset(tr_embeddings, tr_raw, max_len=args.max_len)
        val_ds = NewsRecommendationDataset(val_embeddings, val_raw, max_len=args.max_len)
        tra_data_loader = data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, pin_memory=True)
        val_data_loader = data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    else:
        with open(test_path, 'r') as f:
            test_raw = json.load(f)
        test_data = Data_Sequence(test_raw, test_embeddings, args, 768)
        test_data_loader = test_data.get_pytorch_dataloaders()
        
    


    classifier_data = Data_Classifier(cls_data, args.batch_size)
    classifier_data_loader = classifier_data.get_pytorch_dataloaders()


    diffusion_module = Diffusion(args.timesteps, args.beta_start, args.beta_end, args.w, args.fusion_mode, args.max_len).to(args.device)
    classifier = ModelWithEmbeddingIB(args.input_dim, args.hidden_dim, args.bottleneck_dim, 2).to(args.device)


    trainer = Trainer(args, classifier, diffusion_module, classifier_data_loader, tra_data_loader, val_data_loader, logger, label4news_val, tr_embeddings, val_embeddings)


    trainer.train()

def main(args):
    with open(args.args_file, 'r') as f:
        base_args = json.load(f)

    for key, value in base_args.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    
    create_log_dir(args.log_file, args.dataset)
    logger = setup_logging(args.log_file, args.dataset)
    logger.info(args)
    set_seed(args.random_seed)

    if args.grid_search:
        with open(args.param_grid_file, 'r') as f:
            param_grid = json.load(f)
        
        keys, values = zip(*param_grid.items())
        param_combinations = [dict(zip(keys, v)) for v in product(*values)]
        
        for i, params in enumerate(param_combinations):
            for key, value in params.items():
                setattr(args, key, value)
            logger.info(f"Running experiment {i+1}/{len(param_combinations)} with parameters: {params}")
            run_experiment(args, logger)
    else:
        run_experiment(args, logger)

if __name__ == "__main__":
    main(args)