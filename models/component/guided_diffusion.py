import torch
from torch import nn
import torch.nn.functional as F
import math
import pdb
from functools import partial


class Diffusion(nn.Module):
    def __init__(self,
        timesteps = 200,
        beta_start = 0.1,
        beta_end = 0.1,
        hyper_w = 0.1,
        fusion_mode = 'clsattn',
        max_len = 50
                    ):
        super().__init__()

        self.timesteps = timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.w = hyper_w
        
        self.fusion_mode = fusion_mode
        
        self.model = ConditionNet(fusion_mode = fusion_mode, max_len = max_len ,state_size = max_len)
        
        self.betas = linear_beta_schedule(timesteps=self.timesteps, beta_start=self.beta_start, beta_end=self.beta_end)

        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

        self.sqrt_recip_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod - 1)


        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1. - self.alphas_cumprod)

        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
    
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
        
        
    
    def p_losses(self, seq,len_seq,target,content=None, noise=None, loss_type="l2", authenticity= None):

        seq = torch.stack(seq,dim=0)
        
        x_start = self.model.cacu_x(target)
        h,state_hidden =  self.model.cacu_h(seq, len_seq, 0.6, mask[:,1:])
        t = torch.randint(0, self.timesteps, (len_seq.shape[0], ), device=len_seq.device).long()
        
        
        if noise is None:
            noise = torch.randn_like(x_start) 
        
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        
        authenticity = self.model.content_reflect(authenticity)

        predicted_x_feature = self.model(x_noisy, h, t, content_features, state_hidden, authenticity, mask)
        predicted_x = self.model.content_decoder(predicted_x_feature) 
        
        x_origin = self.model.news_embeddings(target).squeeze(1)
        train_emb = self.model.news_embeddings.weight
        encoded_x_list = []
        for i in range(0,train_emb.size(0),seq.size(0)):
            encoded_emb = self.model.content_reflect(train_emb[i:i + seq.size(0)])
            encoded_x_list.append(encoded_emb)
            
        encoded_x = torch.cat(encoded_x_list, dim=0)
        scores = torch.matmul(predicted_x_feature, encoded_x.transpose(0, 1))
        
 
        if loss_type == 'l1':
            loss = F.l1_loss(x_origin, predicted_x)
        elif loss_type == 'l2':
            loss = F.mse_loss(x_origin, predicted_x)
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(x_origin, predicted_x)
        else:
            raise NotImplementedError()
        
        
        
        

        return loss, scores

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )
    
    @torch.no_grad()
    def p_sample(self, model_forward, model_forward_uncon, x, h, t, t_index,content_features,hidden_state,authenticity_0, authenticity_1,attn_mask):

        x_start = (1 + self.w) * model_forward(x, h, t, content_features,hidden_state, authenticity_0,attn_mask) - self.w * model_forward_uncon(x, t ,authenticity_1)
        
        x_t = x 
        model_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)

            return model_mean + torch.sqrt(posterior_variance_t) * noise 
        
    @torch.no_grad()
    def sample(self, seq, len_seq, target,content=None,authenticity_0=None, authenticity_1= None):
        content_features,attn_mask = self.model.content_encoder(content)
        seq = torch.stack(seq,dim=0)
        
        h,hidden_state =  self.model.predict(seq, len_seq,attn_mask[:,1:])
        if h.dim()==1:
            h = h.unsqueeze(0)
        
        x = torch.randn_like(h)


        
        authenticity_0 = self.model.content_reflect(authenticity_0)
        for n in reversed(range(0, self.timesteps)):
            x = self.p_sample(self.model.forward, self.model.forward_uncon, x, h, torch.full((h.shape[0], ), n, device=len_seq.device, dtype=torch.long), n,content_features, hidden_state,authenticity_0, authenticity_1,attn_mask)

        test_emb = self.model.news_embeddings.weight
        
        mask = torch.ones(test_emb.size(0), device=test_emb.device, dtype=torch.bool)
        mask[0] = False 
        
        encoded_x_list = []
        for i in range(0,test_emb.size(0),seq.size(0)):
            encoded_emb = self.model.content_reflect(test_emb[i:i + seq.size(0)])
            encoded_x_list.append(encoded_emb)
        
        encoded_x = torch.cat(encoded_x_list, dim=0)
        scores = torch.matmul(x, encoded_x.transpose(0, 1))

        scores = scores.masked_fill(~mask, float('-inf'))
        output_x = self.model.content_decoder(x)
 
        return output_x, scores


def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def linear_beta_schedule(timesteps, beta_start, beta_end):
    beta_start = beta_start
    beta_end = beta_end
    return torch.linspace(beta_start, beta_end, timesteps)


def cosine_beta_schedule(timesteps, s=0.008):

    t = torch.linspace(0, timesteps, timesteps + 1) / timesteps
    alpha_bar = (torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2)
    alpha_bar = alpha_bar / alpha_bar[0] 
    beta = 1 - alpha_bar[1:] / alpha_bar[:-1]
    return torch.clip(beta, 0.0001, 0.9999)  

def pad_history(itemlist,length,pad_item):
    if len(itemlist)>=length:
        return itemlist[-length:]
    if len(itemlist)<length:
        temp = [pad_item] * (length-len(itemlist))
        itemlist.extend(temp)
        return itemlist

def extract_axis_1(data, indices):
    res = []
    for i in range(data.shape[0]):
        res.append(data[i, indices[i], :])
    res = torch.stack(res, dim=0).unsqueeze(1)
    return res

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings
    
    



class ConditionNet(nn.Module):
    def __init__(self, hidden_size=128, state_size=50, dropout=0.1, fusion_mode='clsattn', device='cuda:0', num_heads=4,max_len = 50):
        super(ConditionNet, self).__init__()
        self.state_size = state_size
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout)
        self.fusion_mode = fusion_mode
        self.device = device
        
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.content_reflect = nn.Sequential(
            nn.Linear(768, hidden_size),
            nn.LeakyReLU(),
            norm_layer(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
        )
        
        self.content_pos = class_token_pos_embed(hidden_size, max_len)
        self.content_embed =  MultiHeadAttention(hidden_size, hidden_size, num_heads, dropout)
        

        
        self.none_embedding = nn.Embedding(
            num_embeddings=1,
            embedding_dim=self.hidden_size,
        )
        
        
        nn.init.normal_(self.none_embedding.weight, 0, 1)
        
        self.positional_embeddings = nn.Embedding(
            num_embeddings=state_size,
            embedding_dim=hidden_size
        )
        # emb_dropout is added
        self.emb_dropout = nn.Dropout(dropout)
        self.ln_1 = nn.LayerNorm(hidden_size)
        self.ln_2 = nn.LayerNorm(hidden_size)
        self.ln_3 = nn.LayerNorm(hidden_size)
        self.ln_4 = nn.LayerNorm(hidden_size)
        self.ln_5 = nn.LayerNorm(hidden_size)
        
        self.mh_attn = MultiHeadAttention(hidden_size, hidden_size, num_heads, dropout)
        self.mh_attn_1 = MultiHeadAttention(hidden_size, hidden_size, num_heads, dropout)
        self.cross_attn_1 = MultiHeadAttention(hidden_size, hidden_size, num_heads, dropout)
        self.cross_attn_2 = MultiHeadAttention(hidden_size, hidden_size, num_heads, dropout)
        
        
        self.nn_1 = nn.Linear(hidden_size, hidden_size)
        self.nn_2 = nn.Linear(2*hidden_size,hidden_size)
        self.nn_3 = nn.Linear(hidden_size,hidden_size)
        self.content_decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            norm_layer(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 768)
        )
        
        
        
        
        self.feed_forward = PositionwiseFeedForward(hidden_size, hidden_size, dropout)

        self.step_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(self.hidden_size),
            nn.Linear(self.hidden_size, self.hidden_size*2),
            nn.GELU(),
            nn.Linear(self.hidden_size*2, self.hidden_size),
        )

        self.emb_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.hidden_size, self.hidden_size*2)
        )

        self.diff_mlp = nn.Sequential(
            nn.Linear(self.hidden_size * 3, self.hidden_size*2),
            nn.GELU(),
            nn.Linear(self.hidden_size*2, self.hidden_size),
        )



        self.diffuser = nn.Sequential(
                nn.Linear(self.hidden_size*4, self.hidden_size)
        )
        self.init_weights()
        
    def init_weights(self):

        for layer in self.content_reflect:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)


    def forward(self, x, h, step, content=None, state_hidden=None, authenticity=None, mask=None):

        content = self.nn_1(content)
        authenticity = self.nn_1(authenticity)
        

        cosine_sim = F.cosine_similarity(content, authenticity.unsqueeze(1), dim=-1)

        gate = (1 + torch.tanh(cosine_sim)) / 2


        content = content * gate.unsqueeze(-1) + authenticity.unsqueeze(1) * (1 - gate).unsqueeze(-1)
        
        t = self.step_mlp(step)
        x = x.squeeze(1)
        
        if self.fusion_mode == 'clsattn':
            x_t = self.nn_2(torch.cat((x, t), dim=1))
            x_t = x_t.unsqueeze(1)
            cross_attn_out = self.cross_attn_1(x_t, content[:, 0].unsqueeze(1))
            cross_attn_out_1 = self.ln_3(x_t + cross_attn_out)
            cross_attn_out_2 = self.cross_attn_2(cross_attn_out_1, h.unsqueeze(1))
            cross_attn_out_3 = self.ln_4(cross_attn_out_1 + cross_attn_out_2)
            res = self.nn_3(cross_attn_out_3) + x_t
            res = res.squeeze(1)



        elif self.fusion_mode == 'seqattn':

            x_t = self.nn_2(torch.cat((x, t), dim=1))
            x_t = x_t.unsqueeze(1)
            cross_attn_out = self.cross_attn_1(x_t, content, content, mask)
            cross_attn_out_1 = self.ln_3(x_t + cross_attn_out)
            cross_attn_out_2 = self.cross_attn_2(cross_attn_out_1, state_hidden, state_hidden, mask[:,1:])
            cross_attn_out_3 = self.ln_4(cross_attn_out_1 + cross_attn_out_2)

            self_att_out = self.mh_attn_1(cross_attn_out_3, h.unsqueeze(1), h.unsqueeze(1))
            res = self.nn_3(self_att_out)
            res = res.squeeze(1)

        else:
            res = self.diffuser(torch.cat((x, h, t, content[:, 0, :]), dim=1))

        return res

    def forward_uncon(self, x, step, authenticity):
        h = self.none_embedding(torch.tensor([0]).to(self.device))
        h = torch.cat([h.view(1, self.hidden_size)]*x.shape[0], dim=0)
        t = self.step_mlp(step)
        if authenticity is not None:
            if self.fusion_mode == 'clsattn':
                authenticity = self.content_reflect(authenticity).repeat(x.shape[0], 1) 
                x_t = self.nn_2(torch.cat((x,t), dim=1))
                x_t = x_t.unsqueeze(1)
                cross_attn_out = self.cross_attn_1(x_t, authenticity.unsqueeze(1))
                cross_attn_out_1 = self.ln_3(x_t + cross_attn_out)
                cross_attn_out_2 = self.cross_attn_2(cross_attn_out_1, h.unsqueeze(1))
                cross_attn_out_3 = self.ln_4(cross_attn_out_1 + cross_attn_out_2)
                res = self.nn_3(cross_attn_out_3)+x_t
                res = res.squeeze(1)


            elif self.fusion_mode == 'seqattn':
                authenticity = self.content_reflect(authenticity).repeat(x.shape[0], 1)
                x_t = self.nn_2(torch.cat((x, t), dim=1))
                x_t = x_t.unsqueeze(1)
                cross_attn_out = self.cross_attn_1(x_t, authenticity.unsqueeze(1), authenticity.unsqueeze(1))
                cross_attn_out_1 = self.ln_3(x_t + cross_attn_out)
                cross_attn_out_2 = self.cross_attn_2(cross_attn_out_1, h.unsqueeze(1),h.unsqueeze(1))
                cross_attn_out_3 = self.ln_4(cross_attn_out_1 + cross_attn_out_2)
                self_att_out  = self.mh_attn_1(cross_attn_out_3, h.unsqueeze(1),h.unsqueeze(1))
                res = self.nn_3(self_att_out)
                res = res.squeeze(1)
                
            else:
                res = self.diffuser(torch.cat((x, h, t, authenticity), dim=1))
 
            

        else:
            h = self.none_embedding(torch.tensor([0]).to(self.device))
            h = torch.cat([h.view(1, self.hidden_size)]*x.shape[0], dim=0)

            t = self.step_mlp(step)
            res = self.diffuser(torch.cat((x, h, t, h), dim=1))
            res = self.ln_4(res)
            res = self.nn_3(res)
            res = res.squeeze(1)
            
        return res

    def cacu_x(self, x):
        x = self.content_reflect(self.news_embeddings(x.long()))
        x = x.squeeze(1)
        
        return x
    
    def content_encoder(self, content):
        text, mask = content
        text = self.content_reflect(text)
        clsheadmask = torch.ones(mask.shape[0], 1, dtype=torch.bool).to(mask.device)
        try:
            mask = torch.cat([clsheadmask, mask],dim=-1)
        except:
            pdb.set_trace()
        

        text = self.content_pos(text)
        content_feature = self.content_embed(text, text, text, mask)
        

        return content_feature, mask

    def cacu_h(self, states, len_states, p ,attn_mask):
        #hidden
        inputs_emb = self.content_reflect(self.news_embeddings(states))
        inputs_emb += self.positional_embeddings(torch.arange(self.state_size).to(self.device))
        seq = self.emb_dropout(inputs_emb)
        news_num = self.news_embeddings.weight.shape[0] - 1
        mask = torch.ne(states, news_num).float().unsqueeze(-1).to(self.device)
        seq *= mask
        seq_normalized = self.ln_1(seq)

        mh_attn_out = self.mh_attn(seq_normalized, seq,seq, attn_mask)
        ff_out = self.feed_forward(self.ln_2(mh_attn_out))
        ff_out *= mask
        ff_out = self.ln_3(ff_out)
        state_hidden = extract_axis_1(ff_out, len_states - 1)
        h = state_hidden.squeeze()

        B, D = h.shape[0], h.shape[1]
        mask1d = (torch.sign(torch.rand(B) - p) + 1) / 2
        maske1d = mask1d.view(B, 1)
        mask = torch.cat([maske1d] * D, dim=1)
        mask = mask.to(self.device)

        h = h * mask + self.none_embedding(torch.tensor([0]).to(self.device)) * (1-mask)


        return h, ff_out
    
    def predict(self, states, len_states,attn_mask):
        #hidden
        inputs_emb = self.content_reflect(self.news_embeddings(states))
        inputs_emb += self.positional_embeddings(torch.arange(self.state_size).to(self.device))
        seq = self.emb_dropout(inputs_emb)
        news_num = self.news_embeddings.weight.shape[0] - 1
        mask = torch.ne(states, news_num).float().unsqueeze(-1).to(self.device)
        seq *= mask
        seq_normalized = self.ln_1(seq)
        mh_attn_out = self.mh_attn(seq_normalized, seq,seq,attn_mask)
        ff_out = self.feed_forward(self.ln_2(mh_attn_out))
        ff_out *= mask
        ff_out = self.ln_3(ff_out)
        state_hidden = extract_axis_1(ff_out, len_states - 1)
        h = state_hidden.squeeze()
        
        return h, ff_out
    
    
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1)
        self.w_2 = nn.Conv1d(d_hid, d_in, 1)
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output

class Attention(nn.Module):
    """
    dot-product attention mechanism
    """
    def __init__(self, attention_dropout=0.5):
        super(Attention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, scale=None, attn_mask=None):

        attention = torch.matmul(q, k.transpose(-2, -1))

    

        if attn_mask is not None:
            if attention.shape[2] == attention.shape[3]:
                attn_mask = attn_mask.unsqueeze(2) * attn_mask.unsqueeze(1)
                attention = attention.masked_fill_(~attn_mask.unsqueeze(1), float(-1e20))
            else:
                attention = attention.masked_fill_(~attn_mask.unsqueeze(1).unsqueeze(1), float(-1e20))
            
                
        
        attention = self.softmax(attention)
        attention = self.dropout(attention)
        attention = torch.matmul(attention, v)


        return attention   
    
class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim=128, out_dim=128, num_heads=8, dropout=0.5):
        super(MultiHeadAttention, self).__init__()
        
        self.model_dim = model_dim
        self.dim_per_head = model_dim // num_heads
        self.num_heads = num_heads
        self.linear_k = nn.Linear(model_dim, self.dim_per_head * num_heads, bias=False)
        self.linear_v = nn.Linear(model_dim, self.dim_per_head * num_heads, bias=False)
        self.linear_q = nn.Linear(model_dim, self.dim_per_head * num_heads, bias=False)

        self.dot_product_attention =Attention(dropout)
        self.linear_final = nn.Linear(model_dim, out_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, query, key, value, attn_mask=None):
        B1, N1, C1 = query.shape
        B2, N2, C2 = key.shape   
        residual = query
        
        dim_per_head = self.dim_per_head
        num_heads = self.num_heads

        key = self.linear_k(key)
        value = self.linear_v(value)
        query = self.linear_q(query)

        key = key.reshape(B2, N2, num_heads, dim_per_head)
        value = value.reshape(B2, N2, num_heads, dim_per_head) 
        query = query.reshape(B1, N1, num_heads, dim_per_head)

        
        key = key.permute(0, 2, 1, 3)
        value = value.permute(0, 2, 1, 3)
        query = query.permute(0, 2, 1, 3)
        scale = (self.dim_per_head)**-0.5
        attention = self.dot_product_attention(query, key, value, 
                                               scale, attn_mask)

        attention = attention.transpose(1, 2)
        attention = attention.reshape(B1, N1, C1)
        

        output = self.linear_final(attention)

        output = self.dropout(output)

        output = self.layer_norm(residual + output)

        
        return output    
    
    
    


class class_token_pos_embed(nn.Module):
    def __init__(self, embed_dim,num_tokens):
        super(class_token_pos_embed, self).__init__()
        
        self.num_tokens = 1  
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens+self.num_tokens, embed_dim))
        
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
    def forward(self, x):  

        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed
        
        return x



class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        return z



class ModelWithEmbeddingIB(nn.Module):
    def __init__(self, input_dim, hidden_dim, bottleneck_dim, num_classes, n_iters=100, logit_scale=1.5, dropout_rate=0.3):
        super(ModelWithEmbeddingIB, self).__init__()
        self.n_iters = n_iters
        self.logit_scale = logit_scale
        self.kl_div_loss = nn.KLDivLoss(reduction='batchmean')

        self.encoder_r = Encoder(input_dim, hidden_dim, bottleneck_dim)


        self.encoder_i = Encoder(input_dim, hidden_dim, bottleneck_dim)

        self.label_embedding = nn.Embedding(num_embeddings=num_classes, embedding_dim=input_dim)

        self.decoder = nn.Sequential(
            nn.Linear(2 * bottleneck_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate), 
            nn.Linear(hidden_dim, input_dim)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),  
            nn.Linear(hidden_dim, num_classes)
        )
        
    def cost_matrix_cosine(self, x, y, eps=1e-5):
        assert x.dim() == y.dim()
        assert x.size(0) == y.size(0)
        assert x.size(1) == y.size(1)
        x_norm = F.normalize(x, p=2, dim=-1, eps=eps)
        y_norm = F.normalize(y, p=2, dim=-1, eps=eps)
        cosine_sim = (x_norm * y_norm).sum(dim=-1, keepdim=True)
        cosine_dist = 1 - cosine_sim
        return cosine_dist.unsqueeze(-1)

    def trace(self, x):
        b, m, n = x.size()
        assert m == n
        mask = torch.eye(n, dtype=torch.uint8, device=x.device
                        ).unsqueeze(0).expand_as(x).bool()
        trace = x.masked_select(mask).contiguous().view(
            b, n).sum(dim=-1, keepdim=False)
        return trace

    @torch.no_grad()
    def ipot(self, C, beta, iteration, k):
        b, m, n = C.size()
        sigma = torch.ones(b, m, dtype=C.dtype, device=C.device)
        T = torch.ones(b, n, m, dtype=C.dtype, device=C.device)
        A = torch.exp(-C.transpose(1, 2) / beta)

        for _ in range(iteration):
            Q = A * T  
            sigma = sigma.view(b, m, 1)
            for _ in range(k):
                delta = 1 / (Q.matmul(sigma).view(b, 1, n))
                sigma = 1 / (delta.matmul(Q))
            T = delta.view(b, n, 1) * Q * sigma
        return T

    def optimal_transport_dist(self, z_label, z_r, cost=None, beta=0.5, iteration=50, k=1):
        """ [B, D], [B, D]"""
        if cost is None:
            cost = self.cost_matrix_cosine(z_label, z_r) 

        T = self.ipot(cost.detach(), beta, iteration, k)
        distance = self.trace(cost.matmul(T.detach()))
        return distance.mean() 

    def contrastive_loss_per_label(self, z_label, z_r, z_i, labels):
        unique_labels = torch.unique(labels)
        total_loss = 0.0

        for label in unique_labels:

            indices = (labels == label).nonzero().squeeze()
            
            if indices.dim() == 0:
                indices = indices.unsqueeze(0) 

            z_label_cur = z_label[indices]
            z_r_cur = z_r[indices]
            z_i_cur = z_i[indices]

            positive_samples = z_r_cur.unsqueeze(1)
            negative_samples = z_i_cur.unsqueeze(1)

            samples = torch.cat([positive_samples, negative_samples], dim=1)
            samples = F.normalize(samples, dim=-1)
            z_label_cur = F.normalize(z_label_cur, dim=-1)

            logits = self.logit_scale * F.cosine_similarity(z_label_cur.unsqueeze(1), samples, dim=2)
            bz, K, _ = positive_samples.shape
            target = torch.cat([torch.ones(bz, K), torch.zeros(bz, K)], dim=1).to(z_label_cur.device)
            logits = logits - torch.logsumexp(logits, dim=1, keepdim=True)  

            loss = self.kl_div_loss(logits, target)
            total_loss += loss

        return total_loss / len(unique_labels)

    def forward(self, x, labels):
        z_r = self.encoder_r(x)
        z_i = self.encoder_i(x)
        label_embed = self.label_embedding(labels)
        z_label = self.encoder_r(label_embed)
        ot_distance = self.optimal_transport_dist(z_label, z_r)
        contrastive_loss_zr_zlabel = self.contrastive_loss_per_label(z_label, z_r, z_i, labels)

        x_recon = []
        x_ori = []
        for label in torch.unique(labels):
            indices = (labels == label).nonzero().squeeze()
            
            if indices.dim() == 0:
                indices = indices.unsqueeze(0)
                
            z_r_shuffled = z_r[indices[torch.randperm(indices.size(0))]]
            z_i_ori = z_i[indices]
            x_recon_label = self.decoder(torch.cat([z_r_shuffled, z_i_ori], dim=1))
            x_recon.append(x_recon_label)
            x_ori.append(x[indices])
        x_recon = torch.cat(x_recon, dim=0)
        x_ori = torch.cat(x_ori, dim=0)
        
        logits = self.classifier(z_r)
        
        return x_recon, z_label, z_r, z_i, ot_distance, x_ori, contrastive_loss_zr_zlabel, logits
