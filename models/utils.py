import torch.utils.data as data_utils
import torch
from torch.utils.data import Dataset

class NewsRecommendationDataset(Dataset):
    def __init__(self,  article_dict, user_dict, max_len=10, embedding_dim=768):
        self.user_dict = user_dict
        self.article_dict = article_dict
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.data = self._prepare_data()

    def _prepare_data(self):
        data = []
        for user_id, news_seq in self.user_dict.items():
            if len(news_seq) < 2:
                continue  
            news_seq = news_seq[:-1]
            if len(news_seq) > self.max_len:
                news_seq = news_seq[-self.max_len:] 
            else:
                news_seq = [0] * (self.max_len - len(news_seq)) + news_seq  

            embeddings = []
            for news_id in news_seq:
                embeddings.append(self.article_dict[news_id])
                
            embeddings = torch.stack(embeddings)



            mask = torch.tensor([0 if news_id == 0 else 1 for news_id in news_seq], dtype=torch.bool)


            label = self.user_dict[user_id][-1]
            news_seq =  torch.tensor(news_seq, dtype=torch.long)

            data.append({
                "user_id": user_id,
                "click_history": embeddings,
                "click_newsid":news_seq,
                "mask": mask,
                "label": [label]
            })
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq_data = self.data[idx]
        embeddings = seq_data["click_history"]
        tokens = seq_data["click_newsid"]
        labels = torch.LongTensor(seq_data["label"])
        mask = torch.BoolTensor(seq_data["mask"])

        return tokens, labels, mask, embeddings


class ClassifierDataset(Dataset):
    def __init__(self, news_data):
        self.news_data = news_data

    def __len__(self):
        return len(self.news_data)

    def __getitem__(self, index):
        data_item = self.news_data[index]["embedding"]
        label_item = self.news_data[index]["label"]
        return data_item, label_item

class Data_Classifier:
    def __init__(self, news_data, batch_size):
        self.news_data = news_data
        self.batch_size = batch_size

    def get_pytorch_dataloaders(self):
        dataset = ClassifierDataset(self.news_data)
        return data_utils.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)