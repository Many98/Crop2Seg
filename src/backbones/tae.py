import copy

from torch import Tensor
import numpy as np
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange, Reduce
from einops import reduce, rearrange, repeat

from src.backbones.positional_encoding import PositionalEncoder, AbsolutePositionalEncoder

from src.backbones.utils import experimental


class TAE2d(nn.Module):
    """
    Adjusted original implementation of `L-TAE` (Lightweight Temporal Attention Encoder)
    Now modified and renamed to `TAE2d` to be more generic.
    Particularly it now supports original L-TAE (`attention_type='lightweight'`)
    and temporal attention encoder based on classical Transformer/Attention block (`attention_type='classical'`)
    When used classical attention block there should be also specified `embedding_reduction` parameter and
    `attention_mask_reduction` parameter.
    """
    def __init__(
            self,
            attention_type='lightweight',
            embedding_reduction='cls',
            attention_mask_reduction='cls',
            num_attention_stages=1,
            stack_stages=False,
            num_cls_tokens=1,
            cls_h=16,
            cls_w=16,
            timeunet_flag=False,  # TODO this is specific setting for timeunet_v2
            # --------------------
            in_channels=128,
            n_head=16,
            d_k=4,
            mlp=[256, 128],
            dropout=0.2,
            d_model=256,
            T=1000,
            return_att=False,
            positional_encoding=True,
            *args, **kwargs
    ):
        """
        Temporal Attention Encoder (TAE) for image time series.
        Attention-based sequence encoding that maps a sequence of images to a single feature map.
        A shared TAE is applied to all pixel positions of the image sequence.
        Args:
            attention_type (str): Defines type of attention. Currently are possible `lightweight` (original L-TAE)
                                   and `classical` (uses classical MultiHeadAttention)
                                   Note that if `classical` attention is used then `embedding_reduction` and
                                   `attention_mask_reduction` parameters should be set to obtain e.g. sequence to
                                   embedding mapping
                                   Default is `lightweight`
            num_cls_tokens (int): number of cls tokens used. Default is 1
            cls_h (int): height of cls token ... must correspond with H dim of input tensor
            cls_w (int): width of cls token ... must correspond with W dim of input tensor
            embedding_reduction (None or str): Defines how to perform reduction from time series
                                                i.e. sequence to
                                                embedding mapping (head x B x T x H x W -> head x B x 1 x H x W).
                                                                    Options are `None` ...  none reduction is performed
                                                                    which can be useful if stacking TAE modules to
                                                                    obtain richer representations in time.
                                                                   `cls` which uses additional cls token and
                                                                   `linear` which projects sequence using Linear layer
                                                                   `mean` which simply takes average of attention
                                                                        embeddings
                                                Used only if `attention_type='classical'`
                                                Default is `cls`
            attention_mask_reduction (None or str): Similarly to `embedding_reduction` parameter specifies how
                                                    to perform reduction on attention masks
                                                    (head x B x T x T x H x W -> head x B x 1 x T x H x W).
                                                    Options are `None`,
                                                                `cls` ... takes attention masks corresponding to cls
                                                                            token
                                                                `linear` ... performs linear projection
                                                                `mean`  ... takes average attention mask
                                                    Used only if `attention_type='classical'`
                                                    Default is `cls`
            num_attention_stages (int): Number of attentions stacked one after another. This can improve extraction
                                         of richer representations trough time
                                         (Default 1)
            stack_stages (bool): Whether to stack multiple attention stages
            in_channels (int): Number of channels of the input embeddings.
            n_head (int): Number of attention heads.
            d_k (int): Dimension of the key and query vectors.
            mlp (List[int]): Widths of the layers of the MLP that processes the concatenated outputs of the attention heads.
            dropout (float): dropout
            d_model (int, optional): If specified, the input tensors will first processed by a fully connected layer
                to project them into a feature space of dimension d_model.
            T (int): Period to use for the positional encoding.
            return_att (bool): If true, the module returns the attention masks along with the embeddings (default False)
                                If False then only new embedding is returned
                              Note that here embeddings is what is meant as attention softmax(QK^T)V
                              and attention mask is just softmax(QK^T)
            positional_encoding (bool): If False, no positional encoding is used (default True).
        """
        super().__init__()
        self.in_channels = in_channels
        self.mlp = copy.deepcopy(mlp)
        self.return_att = return_att
        self.n_head = n_head

        self.attention_type = attention_type.lower()
        self.cls_h = cls_h
        self.cls_w = cls_w
        self.num_cls_tokens = num_cls_tokens
        self.embedding_reduction = embedding_reduction.lower()
        self.attention_mask_reduction = attention_mask_reduction.lower()
        self.num_attention_stages = num_attention_stages
        self.stack_stages = stack_stages

        self.timeunet_flag = timeunet_flag  # TODO new specific setting for timeunet_v2

        if d_model is not None:
            self.d_model = d_model
            self.inconv = nn.Conv1d(in_channels, d_model, 1)  # 1x1 convolution
        else:
            self.d_model = in_channels
            self.inconv = None
        assert self.mlp[0] == self.d_model

        if positional_encoding:
            self.positional_encoder = PositionalEncoder(
                self.d_model // n_head, T=T, repeat=n_head
            )  # TODO we could try change this (without repeat and normal d_model)
            #self.positional_encoder = AbsolutePositionalEncoder(
            #    self.d_model // n_head, repeat=n_head
            # )
            #self.positional_encoder = nn.Linear(365, d_model)
        else:
            self.positional_encoder = None

        if self.attention_type == 'lightweight':
            # we can stage attention modules only if it returns embedded sequence
            # but LightWeightMultiHeadAttention returns only one embedding (not sequence)
            self.num_attention_stages = 1

            self.attention_heads = nn.ModuleList(
                LightweightMultiHeadAttention(
                    n_head=n_head, d_k=d_k, d_in=self.d_model
                )

                for _ in range(self.num_attention_stages)
            )

            self.embedding_reduction = None
            self.attention_mask_reduction = None
        elif self.attention_type == 'classical':
            # we also add linear layer after each MultiHeadAttention
            self.attention_heads = nn.ModuleList(

                MultiHeadAttention(
                    n_head=n_head, d_hidden=d_k, d_in=self.d_model
                )
                for _ in range(self.num_attention_stages)
            )

            # TODO linear is removed
            '''
            self.linear = nn.ModuleList(
                PositionwiseFeedForward(d_in=self.d_model, d_hid=self.d_model)
                for _ in range(self.num_attention_stages)
            )
            '''

            if self.embedding_reduction == 'linear':
                self.linear_embedding_reduction = nn.Sequential(nn.AdaptiveAvgPool1d(45), nn.Linear(45, 1))
            if self.attention_mask_reduction == 'linear':
                self.linear_attention_mask_reduction = nn.Sequential(nn.AdaptiveAvgPool1d(45), nn.Linear(45, 1))
            if self.embedding_reduction == 'cls' or self.attention_mask_reduction == 'cls':
                # (H*W) x 1 x d_model
                # self.cls_token = nn.Parameter(torch.randn(cls_h * cls_w, 1, self.d_model), requires_grad=True)
                # 1 x d_model x H x W  | num_cls_tokens x d_model x H x W
                self.cls_token = nn.Parameter(torch.randn(self.num_cls_tokens, self.in_channels, cls_h, cls_w), requires_grad=True)
                self.cls_position = nn.Parameter(-1 * torch.ones((self.num_cls_tokens,)), requires_grad=False)
                self.cls_pad_mask = nn.Parameter(torch.zeros((self.num_cls_tokens,), dtype=torch.bool), requires_grad=False)

                if self.num_cls_tokens > 1:
                    self.cls_emb_conv = nn.Conv1d(self.num_cls_tokens, 1, 1)  # we want to reduce from NUM_CLS -> 1
                    self.cls_attn_conv = nn.Conv1d(self.num_cls_tokens, 1, 1)

        else:
            raise Exception(f'Unknown attention_type {attention_type}. Please specify one of'
                            f'`lightweight` or `classical`')

        self.in_norm = nn.GroupNorm(
            num_groups=n_head,
            num_channels=self.in_channels,
        )
        self.out_norm = nn.GroupNorm(
            num_groups=n_head,
            num_channels=mlp[-1],
        )

        layers = []
        if self.timeunet_flag:  # TODO used timeunet_flag
            for i in range(len(self.mlp) - 1):
                layers.extend(
                    [
                        nn.Linear(self.mlp[i], self.mlp[i + 1]),
                        Rearrange('bhw t d -> bhw d t'),
                        nn.BatchNorm1d(self.mlp[i + 1]),
                        Rearrange('bhw d t -> bhw t d'),
                        nn.ReLU(),
                    ]
                )
        else:
            for i in range(len(self.mlp) - 1):
                layers.extend(
                    [
                        nn.Linear(self.mlp[i], self.mlp[i + 1]),
                        nn.BatchNorm1d(self.mlp[i + 1]),
                        nn.ReLU(),
                    ]
                )
        self.mlp = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, batch_positions=None, pad_mask=None, return_comp=False):
        sz_b, seq_len, d, h, w = x.shape  # `sz_b` denotes B (batch) dim and `seq_len` T (time) dim
        if pad_mask is not None:
            pad_mask = (
                pad_mask.unsqueeze(-1)
                    .repeat((1, 1, h))
                    .unsqueeze(-1)
                    .repeat((1, 1, 1, w))
            )  # BxTxHxW
            pad_mask = (
                pad_mask.permute(0, 2, 3, 1).contiguous().view(sz_b * h * w, seq_len)
            )

        seq_len_cls = seq_len
        if self.embedding_reduction == 'cls' or self.attention_mask_reduction == 'cls':
            # adjust batch_positions info (its shape is B x T)
            # for cls token we position=-1 to indicate that it is auxiliary token

            cls_position = repeat(self.cls_position, 't -> b t', b=sz_b)
            batch_positions = torch.cat([cls_position, batch_positions], dim=1)

            # adjust pad mask
            cls_pad_mask = repeat(self.cls_pad_mask, 't -> new_dim t', new_dim=sz_b * h * w)
            pad_mask = torch.cat([cls_pad_mask, pad_mask], dim=1)

            # adjust input tensor
            cls_token = repeat(self.cls_token, 't in_channels h w -> b t in_channels h w', b=sz_b)  # t=num_cls_tokens
            x = torch.cat([cls_token, x], dim=1)  # B x (T+self.num_cls_tokens) x in_channels x H x W

            # update shape info
            seq_len_cls += self.num_cls_tokens

        # this expects x in shape B x T x d x H x W
        out = x.permute(0, 3, 4, 1, 2).contiguous().view(sz_b * h * w, seq_len_cls, d)
        out = self.in_norm(out.permute(0, 2, 1)).permute(0, 2, 1)

        if self.inconv is not None:
            out = self.inconv(out.permute(0, 2, 1)).permute(0, 2, 1)

        if self.positional_encoder is not None:
            bp = (
                batch_positions.unsqueeze(-1)
                    .repeat((1, 1, h))
                    .unsqueeze(-1)
                    .repeat((1, 1, 1, w))
            )  # BxTxHxW
            bp = bp.permute(0, 2, 3, 1).contiguous().view(sz_b * h * w, seq_len_cls)

            #bp = rearrange(bp, 'b t -> (b t)')
            #bp = nn.functional.one_hot(bp.long(), num_classes=365).float()  # output shape B x T x 365
            #bp = rearrange(bp, '(b t) x-> b t x', b=sz_b*h*w, t=seq_len)
            #pos_emb = self.positional_encoder(bp)

            out = out + self.positional_encoder(bp)  # TODO we could try concat instead of add positional embeddings
            #out = out + pos_emb

        # MULTIPLE ATTENTION STAGES
        # TODO for now we will use only last attn and out
        #  later we can test using every and stack them
        attentions = []
        attention_masks = []

        # attn is attention mask after softmax + dropout
        # out is what is usually called attention (new embedding)
        for i in range(self.num_attention_stages):
            out, attn = self.attention_heads[i](out, pad_mask=pad_mask)
            #out = self.linear[i](out) # TODO linear is removed
            attentions.append(out)
            attention_masks.append(attn)

        # out, attn = self.attention_heads(out, pad_mask=pad_mask)

        # (B*H*W) x d_in is new embedding shape if LightweightAttention is used
        # and (B*H*W) x T x d_in is shape of new embedding if ClassicalAttention is used
        # and we want it to -> (B*H*W) x d_in
        # attn # num_heads x (B*H*W) x T x T and we want -> num_heads x (B*H*W) x 1 x T -> num_heads x (B*H*W) x T

        if self.embedding_reduction == 'linear':
            for i in range(0 if self.stack_stages else len(attentions) - 1, len(attentions)):
                attentions[i] = self.linear_embedding_reduction(
                    rearrange(attentions[i], 'bhw t d -> bhw d t')).contiguous().view(sz_b * h * w, self.d_model)
        elif self.embedding_reduction == 'cls':
            for i in range(0 if self.stack_stages else len(attentions) - 1, len(attentions)):
                if not self.timeunet_flag:  # TODO timeunet_flag used
                    attentions[i] = attentions[i][:, 0:self.num_cls_tokens, :] # .squeeze(1) ; BHW x NUM_CLS x d_model
                    attentions[i] = attentions[i] if self.num_cls_tokens == 1 else self.cls_emb_conv(attentions[i])  # BHW x 1 x d_model
                    attentions[i] = attentions[i].squeeze(1)
                    # we just take 0:num_cls_tokens tokens which corresponds to cls tokens
                    # and then merge it with 1x1 conv

        elif self.embedding_reduction == 'mean':
            for i in range(0 if self.stack_stages else len(attentions) - 1, len(attentions)):
                attentions[i] = attentions[i].mean(dim=1)
                # out = out.mean(dim=1)

        if self.attention_mask_reduction == 'linear':
            for i in range(0 if self.stack_stages else len(attention_masks) - 1, len(attention_masks)):
                attention_masks[i] = self.linear_attention_mask_reduction(
                    rearrange(attention_masks[i], 'n_head bhw t1 t2-> (n_head bhw) t2 t1')).contiguous().view(
                    self.n_head,
                    sz_b * h * w, seq_len)
        elif self.attention_mask_reduction == 'cls':
            for i in range(0 if self.stack_stages else len(attention_masks) - 1, len(attention_masks)):
                attention_masks[i] = attention_masks[i][:, :, 0:self.num_cls_tokens, self.num_cls_tokens:] #  num_head x BHW x NUM_CLS x T
                if self.num_cls_tokens > 1:
                    attention_masks[i] = self.cls_attn_conv(attention_masks[i].contiguous().view(self.n_head*sz_b*h*w,
                                                                                                 self.num_cls_tokens,
                                                                                                 seq_len)).contiguous().view(self.n_head,
                                                                                                                             sz_b*h*w,
                                                                                                                             seq_len)  # num_head x BHW x 1 x T
                else:
                    attention_masks[i] = attention_masks[i].squeeze(2)
                # we just take 0th attn which corresponds to cls token
        elif self.attention_mask_reduction == 'mean':
            for i in range(0 if self.stack_stages else len(attention_masks) - 1, len(attention_masks)):
                attention_masks[i] = attention_masks[i].mean(dim=2)

        if self.stack_stages and not self.timeunet_flag:  # TODO used timeunet_flag
            # TODO fix this because it does not work
            out = torch.cat(attentions, dim=-1)
            attn = torch.cat(attention_masks, dim=0)
        else:
            out = attentions[-1]  # BHW x d_model
            attn = attention_masks[-1]  #  num_head x BHW x T

        out = self.dropout(self.mlp(out))

        if not self.timeunet_flag: # TODO out_norm turned of for timeunet_flag
            out = self.out_norm(out) if self.out_norm is not None else out

        if not self.timeunet_flag: # TODO used timeunet_flag
            out = out.contiguous().view(sz_b, h, w, -1).permute(0, 3, 1, 2)
        else:
            out = out.contiguous().view(sz_b, h, w, seq_len_cls, -1).permute(0, 3, 4, 1, 2)  # b x t x d_in x h x w

        attn = attn.contiguous().view(self.n_head, sz_b, h, w, seq_len).permute(
            0, 1, 4, 2, 3
        )  # head x b x t x h x w

        # attn is attention mask after softmax + dropout
        # out is what is usually called attention (new embedding)
        if self.timeunet_flag:  # TODO if timeunet_flag is set to True
                                #  we return out corresponding to cls embedding which will be passed to skip
                                #  and attention[-1][:, 1:, :]  which corresponds to non cls representation of sequence
                                #  which will be passed to shared conv in timeunet_v2

            return out[:, 1:, :, :, :], None #out[:, 0, :, :, :].squeeze(1)
        if self.return_att:
            return out, attn
        else:
            return out


class MultiHeadAttention(nn.Module):
    """
    Classical implementation of Multi-Head Attention module
        based on github.com/jadore801120/attention-is-all-you-need-pytorch
    """

    def __init__(self, n_head, d_hidden, d_in, dropout=0.05):
        super().__init__()
        self.n_head = n_head
        self.d_hidden = d_hidden
        self.d_in = d_in  # aka d_model  (dimensions of q, k , v)

        # q, k , v inputs are d_in dimensional
        # then they are projected to d_hidden dimensional vectors
        # then is attention calculated which yields d_hidden dimensional representations

        self.fc_k = nn.Linear(d_in, n_head * d_hidden)
        self.fc_v = nn.Linear(d_in, n_head * d_hidden)
        self.fc_q = nn.Linear(d_in, n_head * d_hidden)
        self.fc_out = nn.Linear(n_head * d_hidden, d_in, bias=False)

        self.dropout = nn.Dropout(dropout)

        self.attention = ScaledDotProductAttentionClassical(temperature=np.power(d_hidden, 0.5))
        #self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)

    def forward(self, v, pad_mask=None, return_comp=False):
        d_hidden, d_in, n_head = self.d_hidden, self.d_in, self.n_head
        sz_b, seq_len, _ = v.size()

        # value shape (B*H*W) x T x d_in; d_in=256
        residual = v

        k = self.fc_k(v).view(sz_b, seq_len, n_head, d_hidden)  # note that sz_b is (B*H*W)
        q = self.fc_q(v).view(sz_b, seq_len, n_head, d_hidden)
        v = self.fc_v(v).view(sz_b, seq_len, n_head, d_hidden)

        k = k.transpose(1, 2)  # sz_b x n_head x seq_len x d_hidden
        q = q.transpose(1, 2)  # sz_b x n_head x seq_len x d_hidden
        v = v.transpose(1, 2)  # sz_b x n_head x seq_len x d_hidden

        if pad_mask is not None:
            pad_mask = pad_mask.repeat(
                (n_head, 1)  # replicate pad_mask for each head (B*H*W*n_head) x T
            ).contiguous().view(sz_b, n_head, seq_len).unsqueeze(
                2)  # then reshape and unsqueeze -> (B*H*W) x n_head x 1 x T

        if return_comp:
            output, attn, comp = self.attention(
                q, k, v, pad_mask=pad_mask, return_comp=return_comp
            )
        else:
            output, attn = self.attention(
                q, k, v, pad_mask=pad_mask, return_comp=return_comp
            )

        # input attention shape # (B*H*W) x num_heads x T x T
        attn = attn.transpose(0, 1)  # num_heads x (B*H*W) x T x T

        output = output.transpose(1, 2).contiguous().view(sz_b, seq_len, -1)  # (B*H*W) x T x (n_head*d_hidden)

        output = self.dropout(self.fc_out(output))  # (B*H*W) x T x d_in

        output += residual

        # output = self.layer_norm(output)

        # comp is attention mask before softmax
        # attn is attention mask after softmax + dropout
        # output is what is usually called attention (new embedding)
        if return_comp:
            return output, attn, comp
        else:
            return output, attn


class PositionwiseFeedForward(nn.Module):
    """
    A two-feed-forward-layer module. Used in classical attention/transformer module
    """

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)  # position-wise
        self.w_2 = nn.Linear(d_hid, d_in)  # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x

        x = self.w_2(nn.ReLU()(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x


class LightweightMultiHeadAttention(nn.Module):
    """
    Implementation of original Lightweight Multi-Head Attention module.
        Modified from github.com/jadore801120/attention-is-all-you-need-pytorch
    """

    def __init__(self, n_head, d_k, d_in):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_in = d_in

        # This is change from LTAE master query is learnable parameter
        self.Q = nn.Parameter(torch.zeros((n_head, d_k))).requires_grad_(True)
        nn.init.normal_(self.Q, mean=0, std=np.sqrt(2.0 / (d_k)))

        self.fc1_k = nn.Linear(d_in, n_head * d_k)
        nn.init.normal_(self.fc1_k.weight, mean=0, std=np.sqrt(2.0 / (d_k)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))

    def forward(self, v, pad_mask=None, return_comp=False):
        d_k, d_in, n_head = self.d_k, self.d_in, self.n_head
        sz_b, seq_len, _ = v.size()  # (B*H*W) x T x d where d is dimension of projected vector (256)

        q = torch.stack([self.Q for _ in range(sz_b)], dim=1).view(
            -1, d_k
        )  # (n*b) x d_k

        k = self.fc1_k(v).view(sz_b, seq_len, n_head, d_k)
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, seq_len, d_k)  # (n*b) x lk x dk

        if pad_mask is not None:
            pad_mask = pad_mask.repeat(
                (n_head, 1)
            )  # replicate pad_mask for each head (n_head*B*H*W) x T

        v = torch.stack(v.split(v.shape[-1] // n_head, dim=-1)).view(
            n_head * sz_b, seq_len, -1
        )  # (num_heads*B*H*W) x T x d/num_heads

        if return_comp:
            output, attn, comp = self.attention(
                q, k, v, pad_mask=pad_mask, return_comp=return_comp
            )
        else:
            output, attn = self.attention(
                q, k, v, pad_mask=pad_mask, return_comp=return_comp
            )

        # input attn shape (num_heads * H * W * B) x 1 x T
        attn = attn.view(n_head, sz_b, 1, seq_len)  # num_heads x (B*H*W) x 1 x T
        attn = attn.squeeze(dim=2)  # num_heads x (B*H*W) x T

        # input embedding shape  (B*num_heads*H*W) x 1 x d_in // n_head
        output = output.view(n_head, sz_b, 1, d_in // n_head)  # num_heads x (B*H*W) x 1 x d_in // n_head
        output = output.squeeze(dim=2)  # num_heads x (B*H*W) x d_in // n_head
        output = (
            output.permute(1, 0, 2).contiguous().view(sz_b, -1)
        )  # Concatenate heads  -> # (B*H*W) x d_in

        # comp is attention mask before softmax
        # attn is attention mask after softmax + dropout
        # output is what is usually called attention (new embedding)

        if return_comp:
            return output, attn, comp
        else:
            return output, attn


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention. Used in `LightweightMultiHeadAttention`
        Modified from github.com/jadore801120/attention-is-all-you-need-pytorch
    """

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, pad_mask=None, return_comp=False):
        # shape of query (num_heads * H * W * B) x d_k where d_k is dimension of query vector (d_k=4)
        # shape of value (num_heads * H * W * B) x T x d/num_heads
        # shape of key (num_heads * H * W * B) x T x d_k

        attn = torch.matmul(q.unsqueeze(1), k.transpose(1, 2))  # attn shape  (num_heads * H * W * B) x 1 x T
        attn = attn / self.temperature
        if pad_mask is not None:
            # pad_mask shape before is (n_head*B*H*W) x T
            attn = attn.masked_fill(pad_mask.unsqueeze(1), -1e6)  # pad_mask shape (n_head*B*H*W) x 1 x T
        if return_comp:
            comp = attn
        # compat = attn
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.matmul(attn, v)  # output shape  (num_heads * H * W * B) x 1 x d/num_heads

        # comp is attention mask before softmax
        # attn is attention mask after softmax + dropout
        # output is what is usually called attention (new embedding)
        if return_comp:
            return output, attn, comp
        else:
            return output, attn


class ScaledDotProductAttentionClassical(nn.Module):
    """
    Classical implementation of Scaled Dot-Product Attention
    Based on github.com/jadore801120/attention-is-all-you-need-pytorch
    """

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, pad_mask=None, return_comp=False):
        # shape of query (H * W * B) x num_heads x T x d_hidden where d_hidden=4
        # shape of value (H * W * B) x num_heads x T x d_hidden
        # shape of key (H * W * B) x num_heads x T x d_hidden

        attn = torch.matmul(q, k.transpose(2, 3))  # attn shape  (H * W * B) x num_heads x T x T
        attn = attn / self.temperature

        if pad_mask is not None:
            attn = attn.masked_fill(pad_mask, -1e6)  # TODO maybe here is not correct masking
        if return_comp:
            comp = attn

        # compat = attn
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.matmul(attn, v)  # output shape  (H * W * B) x num_heads x T x d_hidden

        # comp is attention mask before softmax
        # attn is attention mask after softmax + dropout
        # output is what is usually called attention (new embedding)
        if return_comp:
            return output, attn, comp
        else:
            return output, attn


@experimental
class MultiHeadAttentionV1(nn.Module):
    """
    Alternative implementation of MultiHeadAttention (probably additive)

    Notes:
        This is version 1 of alternative implementations of additive `MultiHeadAttention`
    """
    def __init__(self, input_dim, hidden_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.linear_in = nn.Linear(input_dim, hidden_dim * num_heads, bias=False)
        self.linear_out = nn.Linear(hidden_dim * num_heads, input_dim, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, pad_mask=None, return_comp=False):
        batch_size, seq_len, input_dim = x.size()  # BHW x T x d_in

        # Apply the learned linear projection to create multiple "heads"
        heads = self.linear_in(x).contiguous().view(batch_size, seq_len, self.num_heads, -1)  # BHW x T x num_heads x d_hidden

        # Compute the attention scores for each head
        query = torch.tanh(heads)  # BHW x T x num_heads x d_hidden
        key = torch.tanh(heads)  # BHW x T x num_heads x d_hidden
        energy = torch.sum(query * key, dim=-1)  # BHW x T x num_heads
        attn = self.softmax(energy)  # BHW x T x num_head

        attn = rearrange(attn, 'bhw t n_head -> (n_head bhw) t')

        if pad_mask is not None:
            pad_mask = pad_mask.repeat(
                (self.num_heads, 1)
            )  # replicate pad_mask for each head (n_head*B*H*W) x T

            attn = attn.masked_fill(pad_mask, -1e6)

        attn = rearrange(attn, '(n_head bhw) t -> bhw t n_head', n_head=self.num_heads, bhw=batch_size)  # BHW x T x num_heads

        # Apply the attention scores to the input sequence and concatenate the heads
        weighted_heads = attn.unsqueeze(-1) * heads
        weighted_sum = weighted_heads.sum(dim=1).contiguous().view(batch_size, -1)  # BHW x d_hidden * num_heads

        # Linearly transform and output the concatenated heads
        output = self.linear_out(weighted_sum)  # BHW x d_in

        return output, rearrange(attn, 'bhw t n_head -> n_head bhw t')  # rearrange attn to be consistent with LTAE


@experimental
class MultiHeadAttentionV2(nn.Module):
    """
    Alternative implementation of MultiHeadAttention (probably additive)

    Notes:
        This is version 2 of alternative implementations of additive `MultiHeadAttention`
    """

    def __init__(self, d_in, hidden_size, num_heads):
        super().__init__()

        self.num_heads = num_heads
        self.hidden_size = hidden_size

        self.fc1 = nn.Linear(d_in, hidden_size * num_heads)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.fc_out = nn.Linear(hidden_size * num_heads, d_in)

    def forward(self, v: Tensor, pad_mask=None, return_comp=False) -> (Tensor, Tensor):
        # x: BHW x T x d_in
        batch_size, seq_len, d_in = v.size()
        heads = self.fc1(v).view(batch_size, seq_len, self.num_heads, -1)  # BHW x T x num_heads x hidden_size

        scores = self.fc2(torch.tanh(heads))  # BHW x T x num_heads x 1
        attn = torch.softmax(scores, dim=1)  # BHW x T x num_heads x 1

        if pad_mask is not None:
            pad_mask = pad_mask.repeat(
                (self.num_heads, 1)
            )  # replicate pad_mask for each head (n_head*B*H*W) x T

            pad_mask = rearrange(pad_mask, '(n_head bhw) t -> bhw t n_head ()', n_head=self.num_heads, bhw=batch_size)
            # pad_mask shape before is (n_head*B*H*W) x T
            attn = attn.masked_fill(pad_mask, -1e6)

        # this uses directly input x
        # out = torch.sum(attn * x.unsqueeze(2), dim=1)  # [B x num_heads x d_in]

        # or we can use
        out = torch.sum(attn * heads, dim=1)  # [B x num_heads x hidden_size]

        out = out.contiguous().view(batch_size, self.num_heads * self.hidden_size)  # [B x num_heads*hidden_size]

        out = self.fc_out(out)  # [B x d_in]

        return out, attn.squeeze(-1).transpose(0, 1)


@experimental
class MultiHeadAttentionV3(nn.Module):
    """
    Alternative implementation of MultiHeadAttention (probably additive)

    Notes:
        This is version 3 of alternative implementations of additive `MultiHeadAttention`
    """
    def __init__(self, input_size, hidden_size, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        #self.head_size = hidden_size // num_heads

        self.query_projection = nn.Linear(input_size, hidden_size * num_heads, bias=False)
        self.key_projection = nn.Linear(input_size, hidden_size * num_heads, bias=False)
        self.value_projection = nn.Linear(input_size, hidden_size * num_heads, bias=False)
        self.output_projection = nn.Linear(hidden_size * num_heads, input_size, bias=False)

        self.energy_layer = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, x, pad_mask=None, return_comp=False):
        batch_size, seq_len, input_dim = x.size()  # BHW x T x d_in

        query = self.query_projection(x)  # BHW x T x d_hidden * num_heads
        key = self.key_projection(x)  # BHW x T x d_hidden * num_heads
        value = self.value_projection(x)  # BHW x T x d_hidden * num_heads

        # Split the query, key, and value projections into `num_heads` parts
        query = query.view(batch_size, -1, self.num_heads, self.hidden_size)  # BHW x T x num_heads x d_hidden
        key = key.view(batch_size, -1, self.num_heads, self.hidden_size)  # BHW x T x num_heads x d_hidden
        value = value.view(batch_size, -1, self.num_heads, self.hidden_size)  # BHW x T x num_heads x d_hidden

        # Transpose the dimensions for the matrix multiplication
        query = query.transpose(1, 2)  # BHW x num_heads x T x d_hidden
        key = key.transpose(1, 2)  # BHW x num_heads x T x d_hidden
        value = value.transpose(1, 2)  # BHW x num_heads x T x d_hidden

        # Compute the dot product of the query and key vectors
        energy = torch.tanh(query + key)  # BHW x num_heads x T x d_hidden  # TODO not sure if here should be +
        energy = self.energy_layer(energy)  # BHW x num_heads x T x 1

        # Apply softmax to get the attention weights
        attn = torch.softmax(energy, dim=2)  # BHW x num_heads x T x 1

        if pad_mask is not None:
            pad_mask = pad_mask.repeat(
                (self.num_heads, 1)
            )  # replicate pad_mask for each head (n_head*B*H*W) x T

            attn = attn.masked_fill(rearrange(pad_mask, '(n_head bhw) t -> bhw n_head t ()', n_head=self.num_heads,
                                              bhw=batch_size), -1e6)

        # Compute the context vector as the weighted sum of the value vectors
        output = (attn * value).sum(dim=2)  # BHW x num_heads x d_hidden

        # Concatenate the heads and project the result
        output = output.transpose(1, 2)  # BHW x d_hidden x num_heads
        output = output.contiguous().view(batch_size, -1)  # BHW x d_hidden * num_heads
        output = self.output_projection(output)  # BHW x d_in

        return output, rearrange(attn, 'bhw n_head t x -> n_head bhw (t x)')  # x=1

