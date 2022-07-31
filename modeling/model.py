import numpy as np
import torch
from module4model.module4dif import *
from module4model.module4tisasrec import *

class SASRec(torch.nn.Module):
    def __init__(self, user_num, item_num, args):
        super(SASRec, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device

        # TODO: loss += args.l2_emb for regularizing embedding vectors during training
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
        self.item_emb = torch.nn.Embedding(self.item_num+1, args.model.args.hidden_units, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(args.model.args.maxlen, args.model.args.hidden_units) # TO IMPROVE
        self.emb_dropout = torch.nn.Dropout(p=args.model.args.dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList() # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(args.model.args.hidden_units, eps=1e-8)

        for _ in range(args.model.args.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(args.model.args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer =  torch.nn.MultiheadAttention(args.model.args.hidden_units,
                                                            args.model.args.num_heads,
                                                            args.model.args.dropout_rate)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.model.args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.model.args.hidden_units, args.model.args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

            # self.pos_sigmoid = torch.nn.Sigmoid()
            # self.neg_sigmoid = torch.nn.Sigmoid()

    def log2feats(self, log_seqs):
        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
        seqs *= self.item_emb.embedding_dim ** 0.5
        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        seqs += self.pos_emb(torch.LongTensor(positions).to(self.dev))
        seqs = self.emb_dropout(seqs)

        timeline_mask = torch.BoolTensor(log_seqs == 0).to(self.dev)
        seqs *= ~timeline_mask.unsqueeze(-1) # broadcast in last dim

        tl = seqs.shape[1] # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs, 
                                            attn_mask=attention_mask)
                                            # key_padding_mask=timeline_mask
                                            # need_weights=False) this arg do not work?
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *=  ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(seqs) # (U, T, C) -> (U, -1, C)

        return log_feats

    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs): # for training        
        log_feats = self.log2feats(log_seqs) # user_ids hasn't been used yet

        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        # pos_pred = self.pos_sigmoid(pos_logits)
        # neg_pred = self.neg_sigmoid(neg_logits)

        return pos_logits, neg_logits # pos_pred, neg_pred

    def predict(self, user_ids, log_seqs, item_indices): # for inference
        log_feats = self.log2feats(log_seqs) # user_ids hasn't been used yet

        final_feat = log_feats[:, -1, :] # only use last QKV classifier, a waste

        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev)) # (U, I, C)

        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        # preds = self.pos_sigmoid(logits) # rank same item list for different users

        return logits # preds # (U, I)

class TiSASRec(torch.nn.Module): # similar to torch.nn.MultiheadAttention
    def __init__(self, user_num, item_num, time_num, args):
        super(TiSASRec, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device

        # TODO: loss += args.l2_emb for regularizing embedding vectors during training
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
        self.item_emb = torch.nn.Embedding(self.item_num+1, args.model.args.hidden_units, padding_idx=0)
        self.item_emb_dropout = torch.nn.Dropout(p=args.model.args.dropout_rate)

        self.abs_pos_K_emb = torch.nn.Embedding(args.model.args.maxlen, args.model.args.hidden_units)
        self.abs_pos_V_emb = torch.nn.Embedding(args.model.args.maxlen, args.model.args.hidden_units)
        self.time_matrix_K_emb = torch.nn.Embedding(args.model.args.time_span+1, args.model.args.hidden_units)
        self.time_matrix_V_emb = torch.nn.Embedding(args.model.args.time_span+1, args.model.args.hidden_units)

        self.item_emb_dropout = torch.nn.Dropout(p=args.model.args.dropout_rate)
        self.abs_pos_K_emb_dropout = torch.nn.Dropout(p=args.model.args.dropout_rate)
        self.abs_pos_V_emb_dropout = torch.nn.Dropout(p=args.model.args.dropout_rate)
        self.time_matrix_K_dropout = torch.nn.Dropout(p=args.model.args.dropout_rate)
        self.time_matrix_V_dropout = torch.nn.Dropout(p=args.model.args.dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList() # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(args.model.args.hidden_units, eps=1e-8)

        for _ in range(args.model.args.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(args.model.args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer = TimeAwareMultiHeadAttention(args.model.args.hidden_units,
                                                            args.model.args.num_heads,
                                                            args.model.args.dropout_rate,
                                                            args.device)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.model.args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.model.args.hidden_units, args.model.args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

            # self.pos_sigmoid = torch.nn.Sigmoid()
            # self.neg_sigmoid = torch.nn.Sigmoid()

    def seq2feats(self, user_ids, log_seqs, time_matrices):
        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
        seqs *= self.item_emb.embedding_dim ** 0.5
        seqs = self.item_emb_dropout(seqs)

        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        positions = torch.LongTensor(positions).to(self.dev)
        abs_pos_K = self.abs_pos_K_emb(positions)
        abs_pos_V = self.abs_pos_V_emb(positions)
        abs_pos_K = self.abs_pos_K_emb_dropout(abs_pos_K)
        abs_pos_V = self.abs_pos_V_emb_dropout(abs_pos_V)

        time_matrices = torch.LongTensor(time_matrices).to(self.dev)
        time_matrix_K = self.time_matrix_K_emb(time_matrices)
        time_matrix_V = self.time_matrix_V_emb(time_matrices)
        time_matrix_K = self.time_matrix_K_dropout(time_matrix_K)
        time_matrix_V = self.time_matrix_V_dropout(time_matrix_V)

        # mask 0th items(placeholder for dry-run) in log_seqs
        # would be easier if 0th item could be an exception for training
        timeline_mask = torch.BoolTensor(log_seqs == 0).to(self.dev)
        seqs *= ~timeline_mask.unsqueeze(-1) # broadcast in last dim

        tl = seqs.shape[1] # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        for i in range(len(self.attention_layers)):
            # Self-attention, Q=layernorm(seqs), K=V=seqs
            # seqs = torch.transpose(seqs, 0, 1) # (N, T, C) -> (T, N, C)
            Q = self.attention_layernorms[i](seqs) # PyTorch mha requires time first fmt
            mha_outputs = self.attention_layers[i](Q, seqs,
                                            timeline_mask, attention_mask,
                                            time_matrix_K, time_matrix_V,
                                            abs_pos_K, abs_pos_V)
            seqs = Q + mha_outputs
            # seqs = torch.transpose(seqs, 0, 1) # (T, N, C) -> (N, T, C)

            # Point-wise Feed-forward, actually 2 Conv1D for channel wise fusion
            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *=  ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(seqs)

        return log_feats

    def forward(self, user_ids, log_seqs, time_matrices, pos_seqs, neg_seqs): # for training
        log_feats = self.seq2feats(user_ids, log_seqs, time_matrices)

        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        # pos_pred = self.pos_sigmoid(pos_logits)
        # neg_pred = self.neg_sigmoid(neg_logits)

        return pos_logits, neg_logits # pos_pred, neg_pred

    def predict(self, user_ids, log_seqs, time_matrices, item_indices): # for inference
        log_feats = self.seq2feats(user_ids, log_seqs, time_matrices)

        final_feat = log_feats[:, -1, :] # only use last QKV classifier, a waste

        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev)) # (U, I, C)

        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        # preds = self.pos_sigmoid(logits) # rank same item list for different users

        return logits # preds # (U, I)

class TiSASRecwithAux(torch.nn.Module): # similar to torch.nn.MultiheadAttention
    def __init__(self, user_num, item_num, time_num, args):
        super(TiSASRecwithAux, self).__init__()
        self.args = args

        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device

        self.maxlen = args.model.args.maxlen
        self.fusion_type_final = args.model.args.fusion_type_final

        # TODO: loss += args.l2_emb for regularizing embedding vectors during training
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
        self.item_emb = torch.nn.Embedding(self.item_num+1, args.model.args.hidden_units, padding_idx=0)
        self.item_emb_dropout = torch.nn.Dropout(p=args.model.args.dropout_rate)

        self.abs_pos_K_emb = torch.nn.Embedding(args.model.args.maxlen, args.model.args.hidden_units)
        self.abs_pos_V_emb = torch.nn.Embedding(args.model.args.maxlen, args.model.args.hidden_units)
        self.time_matrix_K_emb = torch.nn.Embedding(args.model.args.time_span+1, args.model.args.hidden_units)
        self.time_matrix_V_emb = torch.nn.Embedding(args.model.args.time_span+1, args.model.args.hidden_units)

        self.clac_hlv_nm_Q_emb = torch.nn.Embedding(args.model.args.maxlen, args.model.args.seq_attr_hidden_units)
        self.clac_hlv_nm_K_emb = torch.nn.Embedding(args.model.args.maxlen, args.model.args.seq_attr_hidden_units)
        self.clac_mcls_nm_Q_emb = torch.nn.Embedding(args.model.args.maxlen, args.model.args.seq_attr_hidden_units)
        self.clac_mcls_nm_K_emb = torch.nn.Embedding(args.model.args.maxlen, args.model.args.seq_attr_hidden_units)
        buy_am_n_chnl_dv_input_dim = int(args.model.aux_info.buy_am) + int(args.model.aux_info.chnl_dv)
        self.buy_am_n_chnl_dv_Q = torch.nn.Linear(buy_am_n_chnl_dv_input_dim, args.model.args.seq_attr_hidden_units)
        self.buy_am_n_chnl_dv_K = torch.nn.Linear(buy_am_n_chnl_dv_input_dim, args.model.args.seq_attr_hidden_units)

        self.user_emb = torch.nn.Embedding(self.user_num+1, args.model.args.user_attr_emb_size)
        self.ma_fem_dv_emb = torch.nn.Embedding(3, args.model.args.user_attr_emb_size)
        self.ages_emb = torch.nn.Embedding(7, args.model.args.user_attr_emb_size)
        self.zon_hlv_emb = torch.nn.Embedding(18, args.model.args.user_attr_emb_size)
        self.de_dt_month_emb = torch.nn.Embedding(13, args.model.args.user_attr_emb_size)

        self.item_emb_dropout = torch.nn.Dropout(p=args.model.args.dropout_rate)
        self.abs_pos_K_emb_dropout = torch.nn.Dropout(p=args.model.args.dropout_rate)
        self.abs_pos_V_emb_dropout = torch.nn.Dropout(p=args.model.args.dropout_rate)
        self.time_matrix_K_dropout = torch.nn.Dropout(p=args.model.args.dropout_rate)
        self.time_matrix_V_dropout = torch.nn.Dropout(p=args.model.args.dropout_rate)

        self.clac_hlv_nm_Q_dropout = torch.nn.Dropout(p=args.model.args.dropout_rate)
        self.clac_hlv_nm_K_dropout = torch.nn.Dropout(p=args.model.args.dropout_rate)
        self.clac_mcls_nm_Q_dropout = torch.nn.Dropout(p=args.model.args.dropout_rate)
        self.clac_mcls_nm_K_dropout = torch.nn.Dropout(p=args.model.args.dropout_rate)
        self.buy_am_n_chnl_dv_Q_dropout = torch.nn.Dropout(p=args.model.args.dropout_rate)
        self.buy_am_n_chnl_dv_K_dropout = torch.nn.Dropout(p=args.model.args.dropout_rate)

        self.user_dropout = torch.nn.Dropout(p=args.model.args.dropout_rate)
        self.ma_fem_dv_dropout = torch.nn.Dropout(p=args.model.args.dropout_rate)
        self.ages_dropout = torch.nn.Dropout(p=args.model.args.dropout_rate)
        self.zon_hlv_dropout = torch.nn.Dropout(p=args.model.args.dropout_rate)
        self.de_dt_month_dropout = torch.nn.Dropout(p=args.model.args.dropout_rate)

        self.last_layernorm = torch.nn.LayerNorm(args.model.args.hidden_units, eps=1e-8)

        seq_feat_num = int((self.args.model.aux_info.buy_am) or (self.args.model.aux_info.chnl_dv)) + int(self.args.model.aux_info.clac_hlv_nm) +\
                             int(self.args.model.aux_info.clac_mcls_nm)
        self.trm_encoder = DIFTransformerEncoder(
            n_layers = args.model.args.num_blocks,
            n_heads = args.model.args.num_heads,
            hidden_size = args.model.args.hidden_units,
            attribute_hidden_size = args.model.args.seq_attr_hidden_units,
            feat_num = seq_feat_num,
            inner_size = args.model.args.inner_size,
            hidden_dropout_prob = args.model.args.dropout_rate,
            attn_dropout_prob = args.model.args.attn_dropout_rate,
            hidden_act = args.model.args.hidden_act,
            layer_norm_eps = args.model.args.layer_norm_eps,
            fusion_type = args.model.args.fusion_type_item,
            max_len = args.model.args.maxlen
        )

        self.num_user_aux = int(self.args.model.aux_info.user_id) + int(self.args.model.aux_info.de_dt_month) +\
             int(self.args.model.aux_info.ma_fem_dv) + int(self.args.model.aux_info.ages) + int(self.args.model.aux_info.zon_hlv)
        self.userMLP = MLPUserAux(
            user_attr_emb_size = args.model.args.user_attr_emb_size,
            num_user_aux = self.num_user_aux, 
            hidden_size = args.model.args.hidden_units, 
            hidden_act = args.model.args.hidden_act,
            num_layers = args.model.args.num_layers_user_aux,
            dropout_prob = args.model.args.dropout_rate,
            layer_norm_eps = args.model.args.layer_norm_eps
        )

            # self.pos_sigmoid = torch.nn.Sigmoid()
            # self.neg_sigmoid = torch.nn.Sigmoid()

    def get_attention_mask(self, item_seq):
        """Generate left-to-right uni-directional attention mask for multi-head attention."""
        attention_mask = (item_seq > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
        # mask for left-to-right unidirectional
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long().to(item_seq.device)
        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def seq2feats(self, user_ids, log_seqs, time_matrices, buy_am, clac_hlv_nm, clac_mcls_nm, pd_nm, chnl_dv, de_dt_month, ma_fem_dv, ages, zon_hlv):
        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
        seqs *= self.item_emb.embedding_dim ** 0.5
        seqs = self.item_emb_dropout(seqs)

        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        positions = torch.LongTensor(positions).to(self.dev)
        abs_pos_K = self.abs_pos_K_emb(positions)
        abs_pos_V = self.abs_pos_V_emb(positions)
        abs_pos_K = self.abs_pos_K_emb_dropout(abs_pos_K)
        abs_pos_V = self.abs_pos_V_emb_dropout(abs_pos_V)

        time_matrices = torch.LongTensor(time_matrices).to(self.dev)
        time_matrix_K = self.time_matrix_K_emb(time_matrices)
        time_matrix_V = self.time_matrix_V_emb(time_matrices)
        time_matrix_K = self.time_matrix_K_dropout(time_matrix_K)
        time_matrix_V = self.time_matrix_V_dropout(time_matrix_V)

        # mask 0th items(placeholder for dry-run) in log_seqs
        # would be easier if 0th item could be an exception for training
        timeline_mask = torch.BoolTensor(log_seqs == 0).to(self.dev)
        seqs *= ~timeline_mask.unsqueeze(-1) # broadcast in last dim

        feature_table = []
        if (self.args.model.aux_info.clac_hlv_nm == True):
            clac_hlv_nm_Q_ = self.clac_hlv_nm_Q_emb(torch.LongTensor(clac_hlv_nm).to(self.dev))
            clac_hlv_nm_K_ = self.clac_hlv_nm_K_emb(torch.LongTensor(clac_hlv_nm).to(self.dev))
            clac_hlv_nm_Q_ = self.clac_hlv_nm_Q_dropout(clac_hlv_nm_Q_)
            clac_hlv_nm_K_ = self.clac_hlv_nm_K_dropout(clac_hlv_nm_K_)
            clac_hlv_nm_attr = torch.stack([clac_hlv_nm_Q_, clac_hlv_nm_K_])
            feature_table.append(clac_hlv_nm_attr)

        if (self.args.model.aux_info.clac_mcls_nm == True):
            clac_mcls_nm_Q_ = self.clac_mcls_nm_Q_emb(torch.LongTensor(clac_mcls_nm).to(self.dev))
            clac_mcls_nm_K_ = self.clac_mcls_nm_K_emb(torch.LongTensor(clac_mcls_nm).to(self.dev))
            clac_mcls_nm_Q_ = self.clac_mcls_nm_Q_dropout(clac_mcls_nm_Q_)
            clac_mcls_nm_K_ = self.clac_mcls_nm_K_dropout(clac_mcls_nm_K_)
            clac_mcls_nm_attr = torch.stack([clac_mcls_nm_Q_, clac_mcls_nm_K_])
            feature_table.append(clac_mcls_nm_attr)

        if (self.args.model.aux_info.buy_am == True) & (self.args.model.aux_info.chnl_dv == True):
            am_chnl = np.concatenate([buy_am.reshape(-1, self.maxlen, 1), chnl_dv.reshape(-1, self.maxlen, 1)], axis=-1)
        elif (self.args.model.aux_info.buy_am == True) & (self.args.model.aux_info.chnl_dv == False):
            am_chnl = buy_am.reshape(-1, self.maxlen, 1)
        else:
            am_chnl = chnl_dv.reshape(-1, self.maxlen, 1)
        if not (self.args.model.aux_info.buy_am == False) & (self.args.model.aux_info.chnl_dv == False):
            am_chnl_Q_ = self.buy_am_n_chnl_dv_Q(torch.FloatTensor(am_chnl).to(self.dev))
            am_chnl_K_ = self.buy_am_n_chnl_dv_K(torch.FloatTensor(am_chnl).to(self.dev))
            am_chnl_Q_ = self.buy_am_n_chnl_dv_Q_dropout(am_chnl_Q_)
            am_chnl_K_ = self.buy_am_n_chnl_dv_K_dropout(am_chnl_K_)
            am_chnl_attr = torch.stack([am_chnl_Q_, am_chnl_K_]).view((2, -1, self.args.model.args.maxlen, self.args.model.args.seq_attr_hidden_units))
            feature_table.append(am_chnl_attr)

        feature_table = torch.stack(feature_table)

        extended_attention_mask = self.get_attention_mask(torch.LongTensor(log_seqs).to(self.dev))
        trm_output = self.trm_encoder(seqs, feature_table, abs_pos_K, abs_pos_V, time_matrix_K, time_matrix_V,
                                        extended_attention_mask, timeline_mask, output_all_encoded_layers=True)
        output = trm_output[-1]

        return output

    def useraux2feats(self, user, ma_fem_dv, ages, zon_hlv, de_dt_month):
        aux_table = []
        if self.args.model.aux_info.user_id:
            user = self.user_emb(torch.LongTensor(user).to(self.dev))
            user = self.user_dropout(user)
            aux_table.append(user)
        if self.args.model.aux_info.de_dt_month:
            de_dt_month = self.de_dt_month_emb(torch.LongTensor(de_dt_month[:,-1]).to(self.dev))
            de_dt_month = self.de_dt_month_dropout(de_dt_month)
            aux_table.append(de_dt_month)
        if self.args.model.aux_info.ma_fem_dv:
            ma_fem_dv = self.ma_fem_dv_emb(torch.LongTensor(ma_fem_dv[:,-1]).to(self.dev))
            ma_fem_dv = self.ma_fem_dv_dropout(ma_fem_dv)
            aux_table.append(ma_fem_dv)
        if self.args.model.aux_info.ages:
            ages = self.ages_emb(torch.LongTensor(ages[:,-1]).to(self.dev))
            ages = self.ages_dropout(ages)
            aux_table.append(ages)
        if self.args.model.aux_info.zon_hlv:
            zon_hlv = self.zon_hlv_emb(torch.LongTensor(zon_hlv[:,-1]).to(self.dev))
            zon_hlv = self.zon_hlv_dropout(zon_hlv)
            aux_table.append(zon_hlv)

        input = torch.cat(aux_table, axis=-1)
        output = self.userMLP(input)
        return output

    def forward(self, user_ids, log_seqs, time_matrices, buy_am, clac_hlv_nm, clac_mcls_nm, pd_nm, chnl_dv, de_dt_month, ma_fem_dv, ages, zon_hlv, pos_seqs, neg_seqs): # for training
        output_items = self.seq2feats(user_ids, log_seqs, time_matrices, buy_am, clac_hlv_nm, clac_mcls_nm, pd_nm, chnl_dv, de_dt_month, ma_fem_dv, ages, zon_hlv)
        output_useraux = self.useraux2feats(user_ids, ma_fem_dv, ages, zon_hlv, de_dt_month)
        output_useraux = torch.stack([output_useraux for i in range(output_items.shape[1])], dim=1)

        if self.fusion_type_final == "sum":
            output = output_items + output_useraux
        elif self.fusion_type_final == "concat":
            output = torch.cat([output_items, output_useraux], dim=-1)
            output_shape = output_items.shape
            output = nn.Linear(output_shape[0], output_shape[0]//2)(output)

        log_feats = self.last_layernorm(output)

        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        # pos_pred = self.pos_sigmoid(pos_logits)
        # neg_pred = self.neg_sigmoid(neg_logits)

        return pos_logits, neg_logits # pos_pred, neg_pred

    def predict(self, user_ids, log_seqs, time_matrices, buy_am, clac_hlv_nm, clac_mcls_nm, pd_nm, chnl_dv, de_dt_month, ma_fem_dv, ages, zon_hlv, item_indices): # for inference
        output_items = self.seq2feats(user_ids, log_seqs, time_matrices, buy_am, clac_hlv_nm, clac_mcls_nm, pd_nm, chnl_dv, de_dt_month, ma_fem_dv, ages, zon_hlv)
        output_useraux = self.useraux2feats(user_ids, ma_fem_dv, ages, zon_hlv, de_dt_month)
        output_useraux = torch.stack([output_useraux for i in range(output_items.shape[1])], dim=1)

        if self.fusion_type_final == "sum":
            output = output_items + output_useraux
        elif self.fusion_type_final == "concat":
            final_linear = nn.Linear(output_shape[-1], output_shape[-1]//2)
            output = torch.cat([output_items, output_useraux], dim=-1)
            output_shape = output.shape
            output = final_linear(output)

        log_feats = self.last_layernorm(output)

        final_feat = log_feats[:, -1, :] # only use last QKV classifier, a waste

        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev)) # (U, I, C)

        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        # preds = self.pos_sigmoid(logits) # rank same item list for different users

        return logits # preds # (U, I)