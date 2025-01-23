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


class Trainer:
    def __init__(self, args, Classifier, Diffusion_Model, cls_set, tr_set, val_set, logger, authenticity_label, train_embed, test_embed):
        self.args = args
        self.tr_set, self.val_set = tr_set, val_set
        self.cls_set = cls_set
        self.classifier = Classifier
        self.diffusion_model = Diffusion_Model
        self.authenticity_label = authenticity_label  
        self.train_news_embeddings = nn.Embedding.from_pretrained(train_embed, freeze=True)
        self.val_news_embeddings = nn.Embedding.from_pretrained(test_embed, freeze=True)

        self.class_optimizers = self.create_optimizers(self.classifier, args,args.cls_optimizer, 'cls')
        self.diffusion_optimizers = self.create_optimizers(self.diffusion_model,args,args.diff_optimizer, 'diff')
        self.logger = logger
        self.rec_loss = nn.CrossEntropyLoss()
        self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.diffusion_optimizers, mode="max", factor=0.3, patience=3,verbose=True)
        # Metrics to track
        self.hit_20 = 0.0
        self.ng_20 = 0.0
        self.best_metric = 0.0  


    def create_optimizers(self, model,args, opt, mode = 'diff'):
        if mode == 'diff':
            lr_used = args.lr
        elif mode == 'cls':
            lr_used = args.lr_cls
        else:
            raise ValueError(f"Unsupported lr: {mode}")

        optimizer_name = opt
        if optimizer_name == 'adam':
            return optim.Adam(model.parameters(), lr=lr_used, weight_decay=args.l2_decay)
        elif optimizer_name == 'sgd':
            return optim.SGD(model.parameters(), lr=lr_used, weight_decay=args.l2_decay, momentum=0.99)
        elif optimizer_name == 'adagrad':
            return optim.Adagrad(model.parameters(), lr=lr_used, eps=1e-8, weight_decay=args.l2_decay)
        elif optimizer_name == 'adamw':
            return optim.AdamW(model.parameters(), lr=lr_used, eps=1e-8, weight_decay=args.l2_decay)
        elif optimizer_name == 'rmsprop':
            return optim.RMSprop(model.parameters(), lr=lr_used, eps=1e-8, weight_decay=args.l2_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {args.optimizer}")

    @staticmethod
    def cls_evaluate(outputs, labels):
        correct = torch.sum(torch.eq(outputs, labels)).item()
        return correct

    def save_model(self, epoch, metric_value, expname, is_best=False):
        """
        Save the model based on the best metric or save by interval.
        """
        
        if is_best or epoch % self.args.save_epoch == 0:
            if metric_value == None:
                torch.save(self.diffusion_model.state_dict(), f'{self.args.model_dir}/diffusion_model_with_epoch{epoch}.pth')
            else:
                suffix = "best" if is_best else f"epoch_{epoch}"
                model_dir = self.args.model_dir


                if not os.path.exists(model_dir):
                    os.makedirs(model_dir)

             
                torch.save(self.diffusion_model.state_dict(), f'{model_dir}/{expname}_diffusion_model_{suffix}with_hr5_{metric_value[0]}_fake_{metric_value[1]}.pth')
                save_type = "best metric" if is_best else "epoch interval"
                print(f'Model saved at epoch: {epoch} ({save_type}) with metric: {metric_value[0]:.4f}')
                self.logger.info(f'Model saved at epoch: {epoch} ({save_type}) with metric: {metric_value[0]:.4f}')

    def freeze_classifier(self,expname):
        """
        Freeze classifier weights to stop training and gradient computation.
        """
        for param in self.classifier.parameters():
            param.requires_grad = False
        if not os.path.exists(self.args.model_dir):
                os.makedirs(self.args.model_dir)
        torch.save(self.classifier.state_dict(), f'{self.args.model_dir}/{expname}_classifier.pth')
        self.classifier.eval()
        print("Classifier model frozen.")
        self.logger.info("Classifier model frozen.")




    def train(self):
        self.classifier.train()
        self.diffusion_model.train()
        
        is_parallel = self.args.num_gpu > 1
        if is_parallel:
            self.classifier = nn.DataParallel(self.classifier)
            self.diffusion_model = nn.DataParallel(self.diffusion_model)

        lr_scheduler = optim.lr_scheduler.StepLR(self.diffusion_optimizers, step_size=self.args.decay_step, gamma=self.args.gamma)

        epoch_bar = trange(self.args.epochs, desc="Epoch")
        for epoch in epoch_bar:
            self.logger.info(f'Epoch: {epoch}')

 
            if epoch < self.args.class_epoch:
                cls_acc = []
                cls_loss = []
                recon_losses = []
                classification_losses = []
                ot_losses = []
                contrastive_losses = []
   
                for clsdata, label in self.cls_set:
                    epoch_bar.set_description(desc = f"Epoch{epoch}, Train the classifier")
                    clsdata, label = clsdata.to(self.args.device), label.long().to(self.args.device)

                    self.class_optimizers.zero_grad()

    
                    x_recon, z_label, z_r, z_i, ot_distance, x_ori, contrastive_loss_zr_zlabel, logits = self.classifier(clsdata, label)
                    
                    recon_loss = F.mse_loss(x_recon, x_ori)
                    classification_loss = F.cross_entropy(logits, label)
                    
                    c_loss = recon_loss + classification_loss + self.args.phi * ot_distance + self.args.tau * contrastive_loss_zr_zlabel 
                    
                    _, predicted = torch.max(logits, 1)
                    acc = self.cls_evaluate(predicted, label) / label.size(0)


                    c_loss.backward()
                    self.class_optimizers.step()

                    cls_acc.append(acc) 
                    cls_loss.append(c_loss.item())
                    recon_losses.append(recon_loss.item())
                    classification_losses.append(classification_loss.item())
                    ot_losses.append(ot_distance.item())
                    contrastive_losses.append(contrastive_loss_zr_zlabel.item())


                avg_cls_loss = mean(cls_loss)
                avg_recon_loss = mean(recon_losses)
                avg_classification_loss = mean(classification_losses)
                avg_ot_loss = mean(ot_losses)
                avg_contrastive_loss = mean(contrastive_losses)
                avg_accuracy = mean(cls_acc)


                epoch_bar.set_postfix(loss = avg_cls_loss, cls_acc = avg_accuracy, ot_loss = avg_ot_loss, cls_loss = avg_classification_loss, recon_loss = avg_recon_loss, cons_loss = avg_contrastive_loss)
                self.logger.info(f'[Epoch {epoch}], Training Classifier, Loss: {avg_cls_loss:.4f}, Recon Loss: {avg_recon_loss:.4f}, Classification Loss: {avg_classification_loss:.4f}, OT Loss: {avg_ot_loss:.4f}, Contrastive Loss: {avg_contrastive_loss:.4f}, Accuracy: {avg_accuracy:.4f}')



            else:
                self.diffusion_model.model.news_embeddings = self.train_news_embeddings.to(self.args.device)
                if epoch == self.args.class_epoch:

                    self.freeze_classifier(self.args.expname)
                

                diff_loss = []
                
           
                for seq_batch, target_batch, attn_mask, embed_seq in self.tr_set:
                    epoch_bar.set_description(desc = f"Epoch{epoch}, Train the diffuser")
                    seq_batch = [x.to(self.args.device) for x in seq_batch]

                    target_batch = target_batch.to(self.args.device)
                    mask_batch = attn_mask.to(self.args.device)
                    embed_batch = embed_seq.to(self.args.device)


                    self.diffusion_optimizers.zero_grad()


                    loss_d, logits = self.diffusion_model.p_losses(seq_batch, mask_batch.sum(dim=1, keepdim=True), target_batch, (embed_batch, mask_batch), authenticity = self.classifier.label_embedding(torch.tensor([0]).to(self.args.device)))

                    loss_r = self.rec_loss(logits, target_batch.squeeze(-1))
                    loss = loss_d*self.args.diff_cof+loss_r

                    loss.backward()
                    self.diffusion_optimizers.step()

                    diff_loss.append(loss.item())

                avg_diff_loss = mean(diff_loss)
                epoch_bar.set_postfix(loss = avg_diff_loss)
           
                self.logger.info(f'[Epoch {epoch}], Training Diffusion Model, Loss: {avg_diff_loss:.4f}')
                lr_scheduler.step()

       
            if epoch % self.args.eval_epoch == 0 and epoch > 0:
                if epoch>self.args.class_epoch:

                    metric_value = self.evaluate()
                    if metric_value[0] > self.best_metric:
                        self.best_metric = metric_value[0]
                        self.save_model(epoch, metric_value, self.args.expname, is_best=True)
                        self.logger.info(f'Best metric updated: {self.best_metric:.4f}')
                else:
                    pass


            if epoch % self.args.save_epoch == 0:
                if epoch < self.args.class_epoch:
                    pass

                elif epoch == self.args.class_epoch:
                    pass
                else:
                    self.save_model(epoch,  None , self.args.expname)
                    self.logger.info(f'Model saved at epoch: {epoch}')

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
    @staticmethod
    def print_and_log_results(logger, hr_list, ndcg_list, recall_list, mrr_list, real_news_metrics,topk):
        real_news_topk_ratio, real_news_first_ratio, weighted_fake_news_ratio = real_news_metrics
        topk_str = ' '.join([f'HR@{k:<7} NDCG@{k:<6} Recall@{k:<5} MRR@{k:<7}' for k in topk])
        metric_str = ' '.join([f'{hr:<11.6f} {ndcg:<11.6f} {recall:<11.6f} {mrr:<11.6f}' 
                            for hr, ndcg, recall, mrr in zip(hr_list, ndcg_list, recall_list, mrr_list)])
        logger.info(f'Start Evaluating: {datetime.datetime.now()}')
        print(topk_str)
        print(metric_str)
        logger.info(topk_str)
        logger.info(metric_str)
        
        print(f'Real News TopK Ratio: {real_news_topk_ratio} Real News First Ratio: {real_news_first_ratio:.6f} Weighted Fake News Suppression Ratio: {weighted_fake_news_ratio}')
        logger.info(f'Real News TopK Ratio: {real_news_topk_ratio} Real News First Ratio: {real_news_first_ratio:.6f} Weighted Fake News Suppression Ratio: {weighted_fake_news_ratio}')

    def evaluate(self):
        print(f'Start Evaluating: {datetime.datetime.now()}')
        self.logger.info(f'Start Evaluating: {datetime.datetime.now()}')
        self.diffusion_model.model.news_embeddings = self.val_news_embeddings.to(self.args.device)
        self.diffusion_model.eval()

        with torch.no_grad():
            hit_purchase, ndcg_purchase, recall_purchase, mrr_purchase = [0] * 3, [0] * 3, [0] * 3, [0] * 3
            total_purchase = 0.0
            topk = [5, 10, 20] 

            real_news_topk = [0] * 3
            real_news_first = 0
            weighted_fake_news = [0] * 3
            total_samples = 0  

            val_data = list(self.val_set)
            if len(val_data) > 500:
                val_data = random.sample(val_data, 500)

            for seq_batch, target_batch, attn_mask, embed_seq in val_data:
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
            
            self.print_and_log_results(self.logger, hr_list, ndcg_list, recall, mrr, real_news_metrics,topk)
        self.lr_scheduler.step(ndcg_list[0])
        return hr_list[0], real_news_metrics[1]