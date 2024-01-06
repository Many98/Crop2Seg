import copy

from torch import Tensor
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from src.backbones.positional_encoding import PositionalEncoder, AbsolutePositionalEncoder
from src.backbones.utils import experimental


@experimental
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
            embedding_reduction='mean',
            attention_mask_reduction='mean',
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
            use_abs_rel_enc=False,
            num_queries=1,
            add_linear=False,
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
            use_abs_rel_enc (bool): Whether to use both date representations: Relative and absolute (DOY)
            num_queries (int): Number of learnable query vectors which will be averaged
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

        self.use_abs_rel_enc = use_abs_rel_enc
        self.add_linear = add_linear

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
                self.d_model // n_head, T=T, repeat=n_head, add_linear=add_linear
            )
            if self.use_abs_rel_enc:
                self.positional_encoder_abs = AbsolutePositionalEncoder(
                    self.d_model // n_head, repeat=n_head
                )

        else:
            self.positional_encoder = None

        if self.attention_type == 'lightweight':
            # we can stage attention modules only if it returns embedded sequence
            # but LightWeightMultiHeadAttention returns only one embedding (not sequence)
            self.num_attention_stages = 1

            self.attention_heads = nn.ModuleList(
                LightweightMultiHeadAttention(
                    n_head=n_head, d_k=d_k, d_in=self.d_model, n=num_queries
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

            if self.embedding_reduction == 'linear':
                self.linear_embedding_reduction = nn.Sequential(nn.AdaptiveAvgPool1d(45), nn.Linear(45, 1))
            if self.attention_mask_reduction == 'linear':
                self.linear_attention_mask_reduction = nn.Sequential(nn.AdaptiveAvgPool1d(45), nn.Linear(45, 1))
            if self.embedding_reduction == 'cls' or self.attention_mask_reduction == 'cls':
                # (H*W) x 1 x d_model
                # self.cls_token = nn.Parameter(torch.randn(cls_h * cls_w, 1, self.d_model), requires_grad=True)
                # 1 x d_model x H x W  | num_cls_tokens x d_model x H x W
                self.cls_token = nn.Parameter(torch.randn(self.num_cls_tokens, self.in_channels, cls_h, cls_w),
                                              requires_grad=True)
                self.cls_position = nn.Parameter(-1 * torch.ones((self.num_cls_tokens,)), requires_grad=False)
                self.cls_pad_mask = nn.Parameter(torch.zeros((self.num_cls_tokens,), dtype=torch.bool),
                                                 requires_grad=False)

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
            pad_mask = repeat(pad_mask, 'b t -> b t h w', h=h, w=w)  # BxTxHxW
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
            if self.use_abs_rel_enc:
                bp = repeat(batch_positions[..., 0], 'b t -> b t h w', h=h, w=w)
                bp = bp.permute(0, 2, 3, 1).contiguous().view(sz_b * h * w, seq_len_cls)

                bp2 = repeat(batch_positions[..., 1], 'b t -> b t h w', h=h, w=w)  # BxTxHxW
                bp2 = bp2.permute(0, 2, 3, 1).contiguous().view(sz_b * h * w, seq_len_cls)

                out = out + self.positional_encoder(bp) + self.positional_encoder_abs(bp2)
            else:
                bp = repeat(batch_positions, 'b t -> b t h w', h=h, w=w)  # BxTxHxW
                bp = bp.permute(0, 2, 3, 1).contiguous().view(sz_b * h * w, seq_len_cls)

                out = out + self.positional_encoder(bp)

        # MULTIPLE ATTENTION STAGES
        attentions = []
        attention_masks = []

        # attn is attention mask after softmax + dropout
        # out is what is usually called attention (new embedding)
        for i in range(self.num_attention_stages):
            out, attn = self.attention_heads[i](out, pad_mask=pad_mask)
            # out = self.linear[i](out)
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
                attentions[i] = attentions[i][:, 0:self.num_cls_tokens, :]  # .squeeze(1) ; BHW x NUM_CLS x d_model
                attentions[i] = attentions[i] if self.num_cls_tokens == 1 else self.cls_emb_conv(
                    attentions[i])  # BHW x 1 x d_model
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
                attention_masks[i] = attention_masks[i][:, :, 0:self.num_cls_tokens,
                                     self.num_cls_tokens:]  # num_head x BHW x NUM_CLS x T
                if self.num_cls_tokens > 1:
                    attention_masks[i] = self.cls_attn_conv(
                        attention_masks[i].contiguous().view(self.n_head * sz_b * h * w,
                                                             self.num_cls_tokens,
                                                             seq_len)).contiguous().view(self.n_head,
                                                                                         sz_b * h * w,
                                                                                         seq_len)  # num_head x BHW x 1 x T
                else:
                    attention_masks[i] = attention_masks[i].squeeze(2)
                # we just take 0th attn which corresponds to cls token
        elif self.attention_mask_reduction == 'mean':
            for i in range(0 if self.stack_stages else len(attention_masks) - 1, len(attention_masks)):
                attention_masks[i] = attention_masks[i].mean(dim=2)

        if self.stack_stages:
            out = torch.cat(attentions, dim=-1)
            attn = torch.cat(attention_masks, dim=0)

        out = self.dropout(self.mlp(out))

        out = self.out_norm(out) if self.out_norm is not None else out

        out = out.contiguous().view(sz_b, h, w, -1).permute(0, 3, 1, 2)

        attn = attn.contiguous().view(self.n_head, sz_b, h, w, seq_len).permute(
            0, 1, 4, 2, 3
        )  # head x b x t x h x w

        # attn is attention mask after softmax + dropout
        # out is what is usually called attention (new embedding)

        if self.return_att:
            return out, attn
        else:
            return out


class LTAE(nn.Module):
    """
    Original implementation of `L-TAE` (Lightweight Temporal Attention Encoder)
    with small adjustments
    """

    def __init__(
            self,
            in_channels=128,
            n_head=16,
            d_k=4,
            mlp=[256, 128],
            dropout=0.2,
            d_model=256,
            T=1000,
            positional_encoding=True,
            use_abs_rel_enc=False,
            use_doy=False,
            num_queries=1,
            add_linear=False,
            *args, **kwargs
    ):
        """
        Lightweight Temporal Attention Encoder (LTAE) for image time series.
        Attention-based sequence encoding that maps a sequence of images to a single feature map.
        A shared LTAE is applied to all pixel positions of the image sequence.
        Args:
            in_channels (int): Number of channels of the input embeddings.
            n_head (int): Number of attention heads.
            d_k (int): Dimension of the key and query vectors.
            mlp (List[int]): Widths of the layers of the MLP that processes the concatenated outputs of the attention heads.
            dropout (float): dropout
            d_model (int, optional): If specified, the input tensors will first processed by a fully connected layer
                to project them into a feature space of dimension d_model.
            T (int): Period to use for the positional encoding.
            positional_encoding (bool): If False, no positional encoding is used (default True).
            use_abs_rel_enc (bool): Whether to use both date representations: Relative and absolute (DOY)
            num_queries (int): Number of learnable query vectors which will be averaged
        """
        super().__init__()
        self.in_channels = in_channels
        self.mlp = copy.deepcopy(mlp)
        self.n_head = n_head

        self.num_queries = num_queries

        self.use_abs_rel_enc = use_abs_rel_enc
        self.add_linear = add_linear

        if d_model is not None:
            self.d_model = d_model
            self.inconv = nn.Conv1d(in_channels, d_model, 1)  # 1x1 convolution
        else:
            self.d_model = in_channels
            self.inconv = None
        assert self.mlp[0] == self.d_model

        if positional_encoding:
            if use_doy:
                if add_linear:
                    self.positional_encoder = PositionalEncoder(
                        self.d_model // n_head, repeat=n_head, add_linear=add_linear
                    )
                else:
                    self.positional_encoder = AbsolutePositionalEncoder(
                        self.d_model // n_head, repeat=n_head
                    )
            else:
                self.positional_encoder = PositionalEncoder(
                    self.d_model // n_head, repeat=n_head, add_linear=add_linear
                )
            if self.use_abs_rel_enc:
                self.positional_encoder_abs = AbsolutePositionalEncoder(
                    self.d_model // n_head, repeat=n_head
                )

        else:
            self.positional_encoder = None

        self.attention_head = LightweightMultiHeadAttention(
            n_head=n_head, d_k=d_k, d_in=self.d_model, n=num_queries
        )

        self.in_norm = nn.GroupNorm(
            num_groups=n_head,
            num_channels=self.in_channels,
        )

        self.out_norm = nn.GroupNorm(
            num_groups=n_head,
            num_channels=mlp[-1],
        )

        self.mlp = nn.Sequential(
            nn.Linear(self.mlp[0], self.mlp[1]),
            Rearrange('b n c -> b c n'),
            nn.BatchNorm1d(self.mlp[1]),
            Rearrange('b c n -> b n c'),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x, batch_positions=None, pad_mask=None, return_comp=False):
        sz_b, seq_len, d, h, w = x.shape  # `sz_b` denotes B (batch) dim and `seq_len` T (time) dim
        if pad_mask is not None:
            pad_mask = repeat(pad_mask, 'b t -> b t h w', h=h, w=w)
            pad_mask = (
                pad_mask.permute(0, 2, 3, 1).contiguous().view(sz_b * h * w, seq_len)
            )

        # this expects x in shape B x T x d x H x W
        out = x.permute(0, 3, 4, 1, 2).contiguous().view(sz_b * h * w, seq_len, d)
        out = self.in_norm(out.permute(0, 2, 1)).permute(0, 2, 1)

        if self.inconv is not None:
            out = self.inconv(out.permute(0, 2, 1)).permute(0, 2, 1)

        if self.positional_encoder is not None:
            if self.use_abs_rel_enc:
                bp = repeat(batch_positions[..., 0], 'b t -> b t h w', h=h, w=w)
                bp = bp.permute(0, 2, 3, 1).contiguous().view(sz_b * h * w, seq_len)

                bp2 = repeat(batch_positions[..., 1], 'b t -> b t h w', h=h, w=w)  # BxTxHxW
                bp2 = bp2.permute(0, 2, 3, 1).contiguous().view(sz_b * h * w, seq_len)

                out = out + self.positional_encoder(bp) + self.positional_encoder_abs(bp2)
            else:
                bp = repeat(batch_positions, 'b t -> b t h w', h=h, w=w)  # BxTxHxW
                bp = bp.permute(0, 2, 3, 1).contiguous().view(sz_b * h * w, seq_len)

                out = out + self.positional_encoder(bp)

        out, attn = self.attention_head(out, pad_mask=pad_mask)

        # (B*H*W) x n x d_in is new embedding shape
        # attn # num_heads x (B*H*W) x n x T

        out = self.mlp(out)

        out = self.out_norm(out.transpose(1, 2)).transpose(1, 2) if self.out_norm is not None else out

        if self.num_queries == 1:
            attn = attn.contiguous().view(self.n_head, sz_b, h, w, seq_len).permute(
                0, 1, 4, 2, 3
            )  # head x b x t x h x w
            out = out.contiguous().view(sz_b, h, w, -1).permute(0, 3, 1, 2)
        else:
            attn = attn.contiguous().view(self.n_head, sz_b, h, w, self.num_queries, seq_len).permute(
                0, 1, 4, 5, 2, 3
            )  # head x b x n x t x h x w
            out = out.contiguous().view(sz_b, h, w, self.num_queries, -1).permute(0, 3, 4, 1, 2)

        # attn is attention mask after softmax + dropout
        # out is what is usually called attention (new embedding)

        return out, attn


class LTAE4WTAE(nn.Module):
    """
    Original implementation of `L-TAE` (Lightweight Temporal Attention Encoder)
    with small adjustments for WTAE
    """

    def __init__(
            self,
            in_channels=128,
            n_head=16,
            d_k=4,
            d_model=256,
            positional_encoding=True,
            use_abs_rel_enc=False,
            num_queries=1,
            use_doy=False,
            add_linear=False,
            *args, **kwargs
    ):
        """
        Lightweight Temporal Attention Encoder (LTAE) for image time series.
        Attention-based sequence encoding that maps a sequence of images to a single feature map.
        A shared LTAE is applied to all pixel positions of the image sequence.
        Args:
            in_channels (int): Number of channels of the input embeddings.
            n_head (int): Number of attention heads.
            d_k (int): Dimension of the key and query vectors.
            d_model (int, optional): If specified, the input tensors will first processed by a fully connected layer
                to project them into a feature space of dimension d_model.
            T (int): Period to use for the positional encoding.
            positional_encoding (bool): If False, no positional encoding is used (default True).
            use_abs_rel_enc (bool): Whether to use both date representations: Relative and absolute (DOY)
            num_queries (int): Number of learnable query vectors which will be averaged
        """
        super().__init__()
        self.in_channels = in_channels
        self.n_head = n_head

        self.num_queries = num_queries

        self.use_abs_rel_enc = use_abs_rel_enc
        self.add_linear = add_linear

        if d_model is not None:
            self.d_model = d_model
            self.inconv = nn.Conv1d(in_channels, d_model, 1)  # 1x1 convolution
        else:
            self.d_model = in_channels
            self.inconv = None

        if positional_encoding:
            if use_doy:
                if add_linear:
                    self.positional_encoder = PositionalEncoder(
                        self.d_model // n_head, repeat=n_head, add_linear=add_linear
                    )
                else:
                    self.positional_encoder = AbsolutePositionalEncoder(
                        self.d_model // n_head, repeat=n_head
                    )

            else:
                self.positional_encoder = PositionalEncoder(
                    self.d_model // n_head, repeat=n_head, add_linear=add_linear
                )
            if self.use_abs_rel_enc:

                self.positional_encoder_abs = AbsolutePositionalEncoder(
                    self.d_model // n_head, repeat=n_head
                )
        else:
            self.positional_encoder = None

        self.attention_head = LightweightMultiHeadAttention(
            n_head=n_head, d_k=d_k, d_in=self.d_model, n=num_queries
        )

        self.in_norm = nn.GroupNorm(
            num_groups=n_head,
            num_channels=self.in_channels,
        )

    def forward(self, x, batch_positions=None, pad_mask=None, return_comp=False):
        sz_b, seq_len, d, h, w = x.shape  # `sz_b` denotes B (batch) dim and `seq_len` T (time) dim
        if pad_mask is not None:
            pad_mask = repeat(pad_mask, 'b t -> b t h w', h=h, w=w)
            pad_mask = (
                pad_mask.permute(0, 2, 3, 1).contiguous().view(sz_b * h * w, seq_len)
            )

        # this expects x in shape B x T x d x H x W
        out = x.permute(0, 3, 4, 1, 2).contiguous().view(sz_b * h * w, seq_len, d)
        out = self.in_norm(out.permute(0, 2, 1)).permute(0, 2, 1)

        if self.inconv is not None:
            out = self.inconv(out.permute(0, 2, 1)).permute(0, 2, 1)

        if self.positional_encoder is not None:
            if self.use_abs_rel_enc:
                bp = repeat(batch_positions[..., 0], 'b t -> b t h w', h=h, w=w)
                bp = bp.permute(0, 2, 3, 1).contiguous().view(sz_b * h * w, seq_len)

                bp2 = repeat(batch_positions[..., 1], 'b t -> b t h w', h=h, w=w)  # BxTxHxW
                bp2 = bp2.permute(0, 2, 3, 1).contiguous().view(sz_b * h * w, seq_len)

                out = out + self.positional_encoder(bp) + self.positional_encoder_abs(bp2)
            else:
                bp = repeat(batch_positions, 'b t -> b t h w', h=h, w=w)  # BxTxHxW
                bp = bp.permute(0, 2, 3, 1).contiguous().view(sz_b * h * w, seq_len)

                out = out + self.positional_encoder(bp)

        _, attn = self.attention_head(out, pad_mask=pad_mask)

        # attn # num_heads x (B*H*W) x n x T

        if self.num_queries == 1:
            attn = attn.contiguous().view(self.n_head, sz_b, h, w, seq_len).permute(
                0, 1, 4, 2, 3
            )  # head x b x t x h x w
        else:
            attn = attn.contiguous().view(self.n_head, sz_b, h, w, self.num_queries, seq_len).permute(
                0, 1, 4, 5, 2, 3
            )  # head x b x n x t x h x w

        # attn is attention mask after softmax + dropout
        # out is what is usually called attention (new embedding)

        return attn


class MultiHeadAttention(nn.Module):
    """
    Classical implementation of Multi-Head Attention module
        based on github.com/jadore801120/attention-is-all-you-need-pytorch
    """

    def __init__(self, n_head, d_hidden, d_in, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_hidden = d_hidden  # (dimensions of q, k )
        self.d_in = d_in  # aka d_model  (dimensions of  v)

        # q, k , v inputs are d_in dimensional
        # then they are projected to d_hidden and d_in dimensional vectors
        # then is attention calculated which yields d_in dimensional representations

        self.fc_k = nn.Linear(d_in, n_head * d_hidden)
        self.fc_v = nn.Linear(d_in, n_head * d_in)
        self.fc_q = nn.Linear(d_in, n_head * d_hidden)
        self.fc_out = nn.Linear(n_head * d_in, d_in, bias=False)

        self.dropout = nn.Dropout(dropout)

        self.attention = ScaledDotProductAttentionClassical(temperature=np.power(d_hidden, 0.5))
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)

    def forward(self, v, pad_mask=None, return_comp=False):
        d_hidden, d_in, n_head = self.d_hidden, self.d_in, self.n_head
        sz_b, seq_len, _ = v.size()

        # value shape (B*H*W) x T x d_in; d_in=256
        residual = v

        k = self.fc_k(v).view(sz_b, seq_len, n_head, d_hidden)  # note that sz_b is (B*H*W)
        q = self.fc_q(v).view(sz_b, seq_len, n_head, d_hidden)
        v = self.fc_v(v).view(sz_b, seq_len, n_head, d_in)

        k = k.transpose(1, 2)  # sz_b x n_head x seq_len x d_hidden
        q = q.transpose(1, 2)  # sz_b x n_head x seq_len x d_hidden
        v = v.transpose(1, 2)  # sz_b x n_head x seq_len x d_in

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

        output = output.transpose(1, 2).contiguous().view(sz_b, seq_len, -1)  # (B*H*W) x T x (n_head*d_in)

        output = self.dropout(self.fc_out(output))  # (B*H*W) x T x d_in

        output += residual

        output = self.layer_norm(output)

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

    def __init__(self, n_head, d_k, d_in, n=1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_in = d_in
        self.n = n

        # This is change from LTAE master query is learnable parameter
        self.Q = nn.Parameter(torch.zeros((n_head, n, d_k))).requires_grad_(True)
        nn.init.normal_(self.Q, mean=0, std=np.sqrt(2.0 / d_k))

        self.fc1_k = nn.Linear(d_in, n_head * d_k)
        nn.init.normal_(self.fc1_k.weight, mean=0, std=np.sqrt(2.0 / d_k))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))

    def forward(self, v, pad_mask=None, return_comp=False):
        d_k, d_in, n_head, n = self.d_k, self.d_in, self.n_head, self.n
        sz_b, seq_len, _ = v.size()  # (B*H*W) x T x d where d is dimension of projected vector (256)

        q = torch.stack([self.Q for _ in range(sz_b)], dim=1).view(
            -1, n, d_k
        )  # (n_head*b) x d_k -> (n_head*b) x n x d_k

        k = self.fc1_k(v).view(sz_b, seq_len, n_head, d_k)
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, seq_len, d_k)  # (n_head*b) x lk x dk

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

        # input attn shape (num_heads * H * W * B) x 1 x T -> (num_heads * H * W * B) x n x T
        attn = attn.view(n_head, sz_b, n, seq_len)  # num_heads x (B*H*W) x 1 x T -> num_heads x (B*H*W) x n x T

        # input embedding shape  (B*num_heads*H*W) x 1 x d_in // n_head
        output = output.view(n_head, sz_b, n,
                             d_in // n_head)  # num_heads x (B*H*W) x 1 x d_in // n_head -> num_heads x (B*H*W) x n x d_in // n_head

        output = (
            output.permute(1, 2, 0, 3).contiguous().view(sz_b, n, -1)
        )  # Concatenate heads  -> # (B*H*W) x n x d_in

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
        # shape of query (num_heads * H * W * B) x n x d_k where d_k is dimension of query vector (d_k=4)
        # shape of value (num_heads * H * W * B) x T x d/num_heads
        # shape of key (num_heads * H * W * B) x T x d_k

        attn = torch.matmul(q, k.transpose(1, 2))  # attn shape  (num_heads * H * W * B) x n x T
        attn = attn / self.temperature
        if pad_mask is not None:
            # pad_mask shape before is (n_head*B*H*W) x T
            attn = attn.masked_fill(pad_mask.unsqueeze(1), -1e6)  # pad_mask shape (n_head*B*H*W) x 1 x T

        if return_comp:
            comp = attn

        attn = self.softmax(attn)
        attn = self.dropout(attn)

        output = torch.matmul(attn, v)  # output shape  (num_heads * H * W * B) x n x d/num_heads

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
        # shape of value (H * W * B) x num_heads x T x d_in
        # shape of key (H * W * B) x num_heads x T x d_hidden

        attn = torch.matmul(q, k.transpose(2, 3))  # attn shape  (H * W * B) x num_heads x T x T
        attn = attn / self.temperature

        if pad_mask is not None:
            attn = attn.masked_fill(pad_mask, -1e6)
        if return_comp:
            comp = attn

        # compat = attn
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.matmul(attn, v)  # output shape  (H * W * B) x num_heads x T x d_in

        # comp is attention mask before softmax
        # attn is attention mask after softmax + dropout
        # output is what is usually called attention (new embedding)
        if return_comp:
            return output, attn, comp
        else:
            return output, attn
