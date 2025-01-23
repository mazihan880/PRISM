import torch
import torch.nn as nn
import torch.optim as optim
import datetime
from tqdm import tqdm, trange
from numpy import mean
import numpy as np
import os
import pdb
import torch.nn.functional as F
import random
import argparse
import random
import torch.backends.cudnn as cudnn
import json
import pickle
from itertools import product
from utils import Data_Classifier, NewsRecommendationDataset
from component.guided_diffusion import Diffusion, ModelWithEmbeddingIB
from torch.utils import data

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', type=str, default="", help='path to save model')
parser.add_argument('--dataset', type=str, help='Dataset to use')
parser.add_argument('--random_seed', type=int, nargs='+', required=True,help='List of Random Seeds')
parser.add_argument('--args_file', type=str, default='args.json', help='Path to the args JSON file')
parser.add_argument('--max_len', type=int, help='The max length of sequence')
parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], help='Device to use')
parser.add_argument('--batch_size', type=int, help='Batch size')
parser.add_argument('--w', type=float, help='Diffusion noise scale')
parser.add_argument('--p', type=float, help='Dropout probability')
parser.add_argument('--dropout_rate', type=float, help='Dropout rate')
parser.add_argument('--input_dim', type=int, help='News content dim')
parser.add_argument('--hidden_dim', type=int, help='Model hidden layer dimension')
parser.add_argument('--fusion_mode', type=str,help='fusion mode:clsattn, seqattn')
parser.add_argument('--bottleneck_dim', type=int, help='Model ottleneck layer dimension')

parser.add_argument('--timesteps', type=int, help='Timesteps for diffusion')
parser.add_argument('--beta_end', type=float, help='Beta end of diffusion')
parser.add_argument('--beta_start', type=float, help='Beta start of diffusion')
class Tester:
    def __init__(self, args, Classifier, Diffusion_Model, test_set,  authenticity_label, test_embed):
        self.args = args
        self.test_set = test_set
        self.classifier = Classifier
        self.diffusion_model = Diffusion_Model
        self.authenticity_label = authenticity_label  
        self.test_news_embeddings = nn.Embedding.from_pretrained(test_embed, freeze=True)

        # Metrics to track
        self.hit_20 = 0.0
        self.ng_20 = 0.0


    @staticmethod
    def calculate_hit(sorted_list, topk, true_items, hit_purchase, ndcg_purchase):
        for i in range(len(topk)):
            rec_list = sorted_list[:, :topk[i]] 
            for j in range(len(true_items)):
                if true_items[j] in rec_list[j]:

                    rank = np.where(rec_list[j] == true_items[j])[0][0] + 1
                    hit_purchase[i] += 1.0

                    ndcg_purchase[i] += 1.0 / np.log2(rank + 1)
        return hit_purchase, ndcg_purchase

    
    @staticmethod
    def calculate_recall_mrr(recommended_list, target_list, topk):
        recall = []
        mrr = []
        for k in topk:
            recall_k = sum([1 if target in recommended_list[i][:k] else 0 for i, target in enumerate(target_list)]) / len(target_list)
            mrr_k = sum([1 / (recommended_list[i].tolist().index(target) + 1) if target in recommended_list[i][:k] else 0 for i, target in enumerate(target_list)]) / len(target_list)
            recall.append(recall_k)
            mrr.append(mrr_k)
        return recall, mrr
    @staticmethod
    def calculate_fake_news_metrics(recommended_list, authenticity_labels, topk):
        real_news_topk_ratio = [0] * len(topk)
        weighted_fake_news_ratio = [0] * len(topk)
        real_news_first = 0
        total_samples = len(recommended_list)

        for i in range(total_samples):
            for j, k in enumerate(topk):
                top_k_items = recommended_list[i][:k]
                real_news_count = sum([1 for item in top_k_items if authenticity_labels.get(str(item), 1) == 0])
                real_news_topk_ratio[j] += real_news_count

                weighted_sum = sum([(k - idx) for idx, item in enumerate(top_k_items) if authenticity_labels.get(str(item), 0) == 1])
                weighted_fake_news_ratio[j] += weighted_sum

            if authenticity_labels.get(str(recommended_list[i][0]), 1) == 0:
                real_news_first += 1

        real_news_topk_ratio = [ratio / (total_samples * k) for ratio, k in zip(real_news_topk_ratio, topk)]
        real_news_first_ratio = real_news_first / total_samples
        weighted_fake_news_ratio = [1 - (ratio / (total_samples * sum(range(1, k + 1)))) for ratio, k in zip(weighted_fake_news_ratio, topk)]
        
        return real_news_topk_ratio, real_news_first_ratio, weighted_fake_news_ratio
    

    def test(self):
        self.diffusion_model.model.news_embeddings = self.test_news_embeddings.to(self.args.device)
        self.diffusion_model.eval()

        with torch.no_grad():
            hit_purchase, ndcg_purchase, recall_purchase, mrr_purchase = [0] * 3, [0] * 3, [0] * 3, [0] * 3
            total_purchase = 0.0
            topk = [5, 10, 20] 

            real_news_topk = [0] * 3
            real_news_first = 0
            weighted_fake_news = [0] * 3
            total_samples = 0  

            test_data = list(self.test_set)

            for seq_batch, target_batch, attn_mask, embed_seq in test_data:
                seq_batch = [x.to(self.args.device) for x in seq_batch]
                target_batch = target_batch.to(self.args.device)
                mask_batch = attn_mask.to(self.args.device)
                embed_batch = embed_seq.to(self.args.device)

                x, scores = self.diffusion_model.sample(seq_batch, mask_batch.sum(dim=1, keepdim=True), target_batch, 
                                                        (embed_batch, mask_batch), 
                                                        self.classifier.label_embedding(torch.tensor([0]).to(self.args.device)),
                                                        self.classifier.label_embedding(torch.tensor([1]).to(self.args.device)))

                
                
                _, topK = scores.topk(100, dim=1, largest=True, sorted=True)

                topK = topK.cpu().detach().numpy()
                hr, ndcg = self.calculate_hit(topK, topk, target_batch.cpu().detach().numpy(), hit_purchase, ndcg_purchase)
                recall, mrr = self.calculate_recall_mrr(topK, target_batch.cpu().detach().numpy(), topk)
                total_purchase += len(target_batch)

                real_news_metrics = self.calculate_fake_news_metrics(topK, self.authenticity_label, topk)

        hr_list = [hit / total_purchase for hit in hit_purchase]
        ndcg_list = [ndcg / total_purchase for ndcg in ndcg_purchase]
        return hr_list, ndcg_list, recall, mrr, real_news_metrics
    
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False 

    
if __name__ == "__main__":
    args = parser.parse_args()
    
    with open(args.args_file, 'r') as f:
        base_args = json.load(f)

    for key, value in base_args.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    
    for seed in args.random_seed:
        set_seed(seed)
        test_path = f'../../dataset/data/{args.dataset}/user_test.json'
        test_embedding = f'../../dataset/data/{args.dataset}/embedding_test.pkl'
        test_label = f'../../dataset/data/{args.dataset}/news_labels_test.json'
        with open(test_embedding, 'rb') as f:
            test_embeddings  = pickle.load(f)["embedding"]
        with open(test_path, 'r') as f:
            test_raw = json.load(f)
            
        with open(test_label, 'r') as f:
            label4news_test = json.load(f)    
        test_ds = NewsRecommendationDataset(test_embeddings,test_raw, max_len=args.max_len)
        test_data_loader = data.DataLoader(test_ds, batch_size=args.batch_size, shuffle=True, pin_memory=True)
        diffusion_module = Diffusion(args.timesteps, args.beta_start, args.beta_end, args.w, args.fusion_mode, args.max_len).to(args.device)
        classifier = ModelWithEmbeddingIB(args.input_dim, args.hidden_dim, args.bottleneck_dim, 2).to(args.device)
        
        checkpoint_path = f""
        pretrained_state_dict = torch.load(checkpoint_path, map_location=args.device)
        model_state_dict =diffusion_module.state_dict()
        updated_state_dict = {
            k: v for k, v in pretrained_state_dict.items() if k in model_state_dict and v.size() == model_state_dict[k].size()
        }
        model_state_dict.update(updated_state_dict)
        diffusion_module.load_state_dict(model_state_dict)
        print(f"Loaded {len(updated_state_dict)} layers from the checkpoint for seed {seed}.")
        
        cls_checkpoint_path = f""
        cls_pretrained_state_dict = torch.load(cls_checkpoint_path, map_location=args.device)
        cls_model_state_dict =classifier.state_dict()
        cls_updated_state_dict = {
            k: v for k, v in cls_pretrained_state_dict.items() if k in cls_model_state_dict and v.size() == cls_model_state_dict[k].size()
        }
        cls_model_state_dict.update(cls_updated_state_dict)
        classifier.load_state_dict(cls_model_state_dict)
        print(f"Loaded {len(cls_updated_state_dict)} layers from the cls_checkpoint for seed {seed}.")
        
        tester = Tester(args, classifier, diffusion_module,  test_data_loader, label4news_test, test_embeddings)
        hr, ndcg, recall, mrr, fake_news_metrics = tester.test()
    
        
        
   