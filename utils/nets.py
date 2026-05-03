import math

import torch as T
import torch
import torch.nn as nn
import timm

from .resnet_new import resnet14, resnet20, ResNet
from .resnet_bn_new import resnet14 as resnet14_bn, resnet20 as resnet20_bn, ResNet as ResNetBN

from .wrn import WideResNet, WideResNetBlock, WideResNetNoBN, initialize_wrn_fixup
import torch.nn.functional as F

from einops import rearrange, repeat

from typing import Union
from pathlib import Path
from contextlib import contextmanager


SUPPORTED_ACTIVATIONS = {
    'relu': nn.ReLU,
    'silu': nn.SiLU,
    'tanh': nn.Tanh,
}


def activation_factory(name: str) -> nn.Module:
    activation_key = name.lower()
    if activation_key not in SUPPORTED_ACTIVATIONS:
        supported = ', '.join(sorted(SUPPORTED_ACTIVATIONS))
        raise ValueError(f"Unsupported activation '{name}'. Supported activations: {supported}")
    return SUPPORTED_ACTIVATIONS[activation_key]()


def get_model_presets():
    model_presets = {
        'linear': {
            'type': 'linear',
            'params': {
                'hidden_dim': 512,
                'n_layers': 2
            }
        },
        'linear_s': {
            'type': 'linear',
            'params': {
                'hidden_dim': 256,
                'n_layers': 1,
                'bias': True
            }
        },
        'linear_l': {
            'type': 'linear',
            'params': {
                'hidden_dim': 512,
                'n_layers': 4
            }
        },
        'lin_tiny': {
            'type': 'linear',
            'params': {
                'hidden_dim': 2,
                'n_layers': 1
            }
        },
        'mlp': {
            'type': 'mlp',
            'params': {
                'hidden_dim': 512,
                'n_layers': 2,
                'activation': 'relu',
            }
        },
        'mlp2': {
            'type': 'mlp',
            'params': {
                'hidden_dim': 256,
                'n_layers': 2,
                'activation': 'relu',
            }
        },
        'mlp3': {
            'type': 'mlp',
            'params': {
                'hidden_dim': 256,
                'n_layers': 3,
                'activation': 'relu',
            }
        },
        'mlp_s': {
            'type': 'mlp',
            'params': {
                'hidden_dim': 256,
                'n_layers': 1,
                'activation': 'relu',
            }
        },
        'mlp_l': {
            'type': 'mlp',
            'params': {
                'hidden_dim': 512,
                'n_layers': 4,
                'activation': 'relu',
            }
        },

        'mlp_silu': {
            'type': 'mlp',
            'params': {
                'hidden_dim': 512,
                'n_layers': 2,
                'activation': 'silu',
            }
        },
        'mlp_tanh': {
            'type': 'mlp',
            'params': {
                'hidden_dim': 512,
                'n_layers': 2,
                'activation': 'tanh',
            }
        },


        'cnn': {
            'type': 'cnn',
            'params': {
                'hidden_dim': 512,
                'activation': 'relu',
            }
        },

        'cnn_silu': {
            'type': 'cnn',
            'params': {
                'hidden_dim': 512,
                'activation': 'silu',
            }
        },


        'resnet': {
            'type': 'resnet',
            'params': {},
        },
        'resnet_bn': {
            'type': 'resnet_bn',
            'params': {},
        },
        'wrn': {
            'type': 'wrn',
            'params': {
                'depth': 10,
                'width_factor': 2,
            },
        },
        'wrn_no_bn': {
            'type': 'wrn_no_bn',
            'params': {
                'depth': 10,
                'width_factor': 2,
            },
        },
        'vit': {
            'type': 'vit',
            'params': {
                'img_size': 32,
                'patch_size': 4,
                'embed_dim': 64,
                'depth': 2,
                'num_heads': 2,
                'mlp_ratio': 4.0,
                'use_layer_norm': True,
                'residual_scaling': True,
            },
        },
        'sst_transformer': {
            'type': 'sst_transformer',
            'params': {
                'vocab_size': 33278,
                'seq_len': 64,
                'd_model': 64,
                'n_heads': 2,
                'n_layers': 2,
                'n_classes': 1,
                'use_bert_emb': True,
            },
        },
    }
    return model_presets


class SquaredLoss(nn.modules.loss._Loss):
    '''
    Basically MSE, but doesn't average over the dimensions.
    With added support for sampling_vector (aka weighting of the samples, aka mask) the samples!
    Used to do GD with noise to simulate SGD
    '''
    __constants__ = ['reduction']

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean',
                 ) -> None:
        super().__init__(size_average, reduce, reduction)

    def forward(self, input: T.Tensor, target: T.Tensor,
                sampling_vector: T.Tensor = None,
                reduction: str = None
                ) -> T.Tensor:
        if input.shape != target.shape:
            raise ValueError("Input and target must have the same shape for the loss to operate as expected.\nDid you forget to squeeze the output?")

        if sampling_vector is not None:
            total_L2 = F.mse_loss(input, target, reduction='none')

            if len(target.shape) != 1:
                loss_per_sample = total_L2.sum(dim=-1)
            else:
                loss_per_sample = total_L2

            assert len(loss_per_sample.shape) == 1
            sampled_loss = T.dot(loss_per_sample, sampling_vector)
            return sampled_loss

        total_L2 =  F.mse_loss(input, target, reduction='none')
        if len(target.shape) != 1:
            loss_per_sample = total_L2.sum(dim=-1)
        else:
            loss_per_sample = total_L2

        if not reduction is None:
            if reduction == 'none':
                return loss_per_sample

            raise ValueError(f"Are you sure you want to use reduction={reduction}? Double-check what you doing - maybe use self.reduction variable at __init__ instead?\n")

        if self.reduction == 'mean':
            return loss_per_sample.mean()
        if self.reduction == 'sum':
            return loss_per_sample.sum()

        raise ValueError("Unknown reduction type")



class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, output_dim, activation: str = 'relu'):
        super(MLP, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.output_dim = output_dim
        self.activation_name = activation.lower()
        self.activation = activation_factory(self.activation_name)

        self.layers = nn.ModuleList()

        self.layers.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(n_layers-1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        x = x.flatten(1)
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        x = self.layers[-1](x)
        return x

    def __repr__(self):
        return f"MLP({self.input_dim}, {self.hidden_dim}, {self.n_layers}, {self.output_dim}, activation={self.activation_name})"


class CNN(nn.Module):
    def __init__(self, fc_hidden_dim, output_dim, activation: str = 'relu'):
        super(CNN, self).__init__()
        self.fc_hidden_dim = fc_hidden_dim
        self.activation_name = activation.lower()

        def act():
            return activation_factory(self.activation_name)

        self.convs = nn.Sequential(
                nn.Conv2d(3, 64, 3, 1), # 64*30*30
                act(),
                nn.Conv2d(64, 64, 3, 1), # 64*28*28
                act(),
                nn.MaxPool2d(2, 2), # 64, 14

                nn.Conv2d(64, 128, 3, 1), # 128, 12
                act(),
                nn.MaxPool2d(2, 2), # 128, 6
        )
        self.fcs = nn.Sequential(
                nn.Linear(128*6*6, fc_hidden_dim, bias=True),
                act(),
                nn.Linear(fc_hidden_dim, output_dim, bias=True)
        )

    def forward(self, x):
        x = self.convs(x)
        x = rearrange(x, 'b c w h -> b (c w h)')
        x = self.fcs(x)
        return x


class ViT(nn.Module):
    def __init__(self, img_size=32, patch_size=4, embed_dim=192, depth=6,
                 num_heads=3, mlp_ratio=4.0, num_classes=10, use_layer_norm=False,
                 residual_scaling=False):
        super().__init__()
        # Disable flash/mem-efficient SDP so second-order gradients work.
        T.backends.cuda.enable_flash_sdp(False)
        T.backends.cuda.enable_mem_efficient_sdp(False)
        self.residual_scaling = residual_scaling
        norm_layer = nn.LayerNorm if use_layer_norm else nn.Identity
        self.model = timm.create_model(
            'vit_tiny_patch16_224',
            pretrained=False,
            norm_layer=norm_layer,
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            num_classes=num_classes,
        )

    def forward(self, x):
        return self.model(x)


class Linear(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, output_dim, bias=True):
        super(Linear, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.output_dim = output_dim

        self.layers = nn.ModuleList()

        self.layers.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(n_layers-1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim, bias=bias))
        self.layers.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        x = x.flatten(1)
        for layer in self.layers:
            x = layer(x)
        return x

    def __repr__(self):
        return f"Linear({self.input_dim}, {self.hidden_dim}, {self.n_layers}, {self.output_dim})"


class LogisticLoss(nn.Module):
    """Binary logistic loss for {-1, +1} labels.

    Replicates Damian et al. (arXiv:2209.15594) SST-2 criterion:
        L = mean(-log_sigmoid(output * label))
    where output is (B, 1) and label is (B,) float {-1, +1}.
    """
    def forward(self, input, target):
        # input: (B, 1) or (B,), target: (B,) float {-1, +1}
        return -F.logsigmoid(input.squeeze(-1) * target).mean()


_BERT_PROJ_CACHE_PATHS = [
    "/n/holylabs/LABS/kdbrantley_lab/Lab/mwalden/bert_emb_proj64.pt",  # Harvard
    "~/bert_emb_proj64.pt",                                             # MIT / other clusters
]

def load_bert_embeddings_projected(vocab_size=33278, d_model=64):
    """Load bert-base-uncased word embeddings projected to d_model dims via SVD.

    Loads from a precomputed cache file if available (fast). Otherwise computes
    from scratch: SVD of BERT's 30522×768 embedding matrix, top-d_model components,
    normalized to zero mean / unit std. Remaining vocab rows (30522:vocab_size) are zero
    (those token ids never appear in SST-2 data).

    Returns a (vocab_size, d_model) tensor, detached, ready to copy into Embedding.weight.
    """
    import os
    if d_model == 64 and vocab_size == 33278:
        for cache_path in _BERT_PROJ_CACHE_PATHS:
            cache_path = os.path.expanduser(cache_path)
            if os.path.exists(cache_path):
                return torch.load(cache_path, weights_only=True)
    from transformers import AutoModel
    model = AutoModel.from_pretrained("bert-base-uncased")
    W = model.embeddings.word_embeddings.weight.data.float().cpu()  # (30522, 768)
    bert_vocab = W.shape[0]
    del model
    U, S, Vt = torch.linalg.svd(W, full_matrices=False)
    W_proj = U[:, :d_model] * S[:d_model].unsqueeze(0)   # (30522, d_model)
    W_proj = (W_proj - W_proj.mean(0)) / W_proj.std(0).clamp(min=1e-6)
    out = torch.zeros(vocab_size, d_model)
    out[:bert_vocab] = W_proj
    out[0].zero_()  # Treat token id 0 as PAD so masked positions contribute nothing.
    return out.detach()


class SSTTransformerBlock(nn.Module):
    """Bidirectional (encoder-style) transformer block with post-norm.

    Matches Damian et al. (arXiv:2209.15594) models/transformer.py exactly:
      x = LayerNorm(x + MultiHeadAttn(x))          # full attention, no causal mask
      x = LayerNorm(x + Linear(GELU(Linear(x))))   # FF with no hidden expansion
    """
    def __init__(self, d_model, n_heads):
        super().__init__()
        # Disable flash/efficient attention so second-order gradients work
        T.backends.cuda.enable_flash_sdp(False)
        T.backends.cuda.enable_mem_efficient_sdp(False)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True, bias=False)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff1 = nn.Linear(d_model, d_model, bias=False)
        self.ff2 = nn.Linear(d_model, d_model, bias=False)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x, pad_mask=None):
        # Full bidirectional attention — no causal mask
        # pad_mask: (B, S) bool, True = PAD position to ignore (PyTorch key_padding_mask convention)
        attn_out, _ = self.attn(x, x, x, key_padding_mask=pad_mask, need_weights=False)
        x = self.ln1(x + attn_out)
        x = self.ln2(x + self.ff2(F.gelu(self.ff1(x))))
        return x


class SSTTransformer(nn.Module):
    """Small bidirectional transformer classifier for SST-2.

    Replicates Damian et al. (arXiv:2209.15594) models/transformer.py exactly:
      - vocab_size=33278 (bert-base-uncased), d_model=64, n_heads=2, n_layers=2, seq_len=64
      - Token + positional embeddings (both learned)
      - n_layers SSTTransformerBlocks (bidirectional, post-norm)
      - Mean pool over sequence -> linear head (zero-initialized weight)
    Intended for use with LogisticLoss and {-1, +1} labels.
    """
    def __init__(self, vocab_size=33278, seq_len=64, d_model=64, n_heads=2, n_layers=2, n_classes=1, use_bert_emb=False):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        if use_bert_emb:
            self.tok_emb.weight.data.copy_(load_bert_embeddings_projected(vocab_size, d_model))
        self.tok_emb.weight.requires_grad_(False)  # frozen: sparse per-batch gradients would break batch sharpness
        self.pos_emb = nn.Embedding(seq_len, d_model)
        self.blocks = nn.ModuleList([SSTTransformerBlock(d_model, n_heads) for _ in range(n_layers)])
        self.head = nn.Linear(d_model, n_classes, bias=True)
        # Zero-initialize head weight and bias (matches Damian's kernel_init=zeros)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        # x: (B, S) long token ids; id=0 is [PAD]
        B, S = x.shape
        pad_mask = (x == 0)  # (B, S), True for [PAD] positions
        pos = torch.arange(S, device=x.device).unsqueeze(0)  # (1, S)
        h = self.tok_emb(x) + self.pos_emb(pos)
        for block in self.blocks:
            h = block(h, pad_mask=pad_mask)
        # masked mean pool — only over real (non-PAD) positions
        valid = (~pad_mask).float()  # (B, S)
        h = (h * valid.unsqueeze(-1)).sum(1) / valid.sum(1, keepdim=True).clamp(min=1.0)
        return self.head(h)  # (B, n_classes)


def prepare_net(model_type: str,
                params: dict
                ):
    activation = params.get('activation', 'relu')
    if model_type == 'linear':
        net = Linear(params['input_dim'], params['hidden_dim'], params['n_layers'], params['output_dim'], params['bias'])

    if model_type == 'mlp':
        net = MLP(params['input_dim'], params['hidden_dim'], params['n_layers'], params['output_dim'], activation=activation)

    if model_type == 'cnn':
        net = CNN(params['hidden_dim'], params['output_dim'], activation=activation)

    if model_type == 'resnet':
        net = resnet20()

    if model_type == 'resnet_bn':
        net = resnet14_bn()

    if model_type == 'wrn':
        net = WideResNet(
            depth=params['depth'],
            widen_factor=params['width_factor'],
            num_classes=params['output_dim'],
        )

    if model_type == 'wrn_no_bn':
        net = WideResNetNoBN(
            depth=params['depth'],
            widen_factor=params['width_factor'],
            num_classes=params['output_dim'],
        )

    if model_type == 'vit':
        net = ViT(
            img_size=params.get('img_size', 32),
            patch_size=params.get('patch_size', 4),
            embed_dim=params.get('embed_dim', 192),
            depth=params.get('depth', 6),
            num_heads=params.get('num_heads', 3),
            mlp_ratio=params.get('mlp_ratio', 4.0),
            num_classes=params['output_dim'],
            use_layer_norm=params.get('use_layer_norm', False),
            residual_scaling=params.get('residual_scaling', False),
        )

    if model_type == 'sst_transformer':
        net = SSTTransformer(
            vocab_size=params.get('vocab_size', 33278),
            seq_len=params.get('seq_len', 64),
            d_model=params.get('d_model', 64),
            n_heads=params.get('n_heads', 2),
            n_layers=params.get('n_layers', 2),
            n_classes=params.get('n_classes', 1),
            use_bert_emb=params.get('use_bert_emb', True),
        )

    return net

def prepare_net_dataset_specific(model_name: str,
                                 dataset: str,
                                 ):
    '''
    Returns the model specific to the provided dataset
    '''
    from .data import get_dataset_presets

    model_presets = get_model_presets()
    params = model_presets[model_name]['params']
    model_type = model_presets[model_name]['type']

    dataset_presets = get_dataset_presets()
    params['input_dim'] = dataset_presets[dataset]['input_dim']
    params['output_dim'] = dataset_presets[dataset]['output_dim']

    net = prepare_net(model_type, params)

    return net


def _init_linear_with_activation(module: nn.Linear, activation: str, scale: float):
    activation = activation.lower()
    if activation == 'tanh':
        gain = nn.init.calculate_gain('tanh')
        nn.init.xavier_normal_(module.weight, gain=gain)
    else:
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
    module.weight.data.mul_(scale)
    if module.bias is not None:
        nn.init.zeros_(module.bias)


def _init_conv_with_activation(module: nn.Conv2d, activation: str, scale: float):
    activation = activation.lower()
    if activation == 'tanh':
        gain = nn.init.calculate_gain('tanh')
        nn.init.xavier_normal_(module.weight, gain=gain)
    else:
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
    module.weight.data.mul_(scale)
    if module.bias is not None:
        nn.init.zeros_(module.bias)


def initialize_mlp(net, scale=None):
    if scale is None:
        scale=1
    activation = getattr(net, 'activation_name', 'relu')
    for m in net.modules():
        if isinstance(m, nn.Linear):
            _init_linear_with_activation(m, activation, scale)


def initialize_cnn(net, scale=None):
    if scale is None:
        scale = 1.0
    activation = getattr(net, 'activation_name', 'relu')
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            _init_conv_with_activation(m, activation, scale)
        elif isinstance(m, nn.Linear):
            _init_linear_with_activation(m, activation, scale)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


def initialize_resnet(net, scale=None):
    if scale is None:
        scale = 1
    for m in net.modules():
        if isinstance(m, torch.nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            m.weight.data.mul_(scale)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.Linear):
            torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            m.weight.data.mul_(scale)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)


def initialize_resnet_bn(net, scale=None, zero_init_residual=False):
    if scale is None:
        scale = 1

    for module in net.modules():
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            module.weight.data.mul_(scale)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='linear')
            module.weight.data.mul_(scale)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    if not zero_init_residual:
        return

    for module in net.modules():
        final_bn = None
        if hasattr(module, "bn3") and isinstance(module.bn3, nn.BatchNorm2d):
            final_bn = module.bn3
        elif hasattr(module, "bn2") and isinstance(module.bn2, nn.BatchNorm2d):
            final_bn = module.bn2

        if final_bn is not None:
            nn.init.zeros_(final_bn.weight)


def initialize_wrn(net: WideResNet, scale=None):
    if scale is None:
        scale = 1.0

    for module in net.modules():
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            module.weight.data.mul_(scale)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    for block in net.modules():
        if isinstance(block, WideResNetBlock):
            nn.init.zeros_(block.bn2.weight)

    nn.init.kaiming_uniform_(net.fc.weight)
    net.fc.weight.data.mul_(scale)
    if net.fc.bias is not None:
        nn.init.zeros_(net.fc.bias)


def initialize_vit(net, scale=None, residual_scaling=False):
    if scale is None:
        scale = 1.0
    std_base = 0.02 * scale
    std_residual = (std_base / math.sqrt(2 * len(net.model.blocks))
                    if residual_scaling else std_base)
    for m in net.model.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.normal_(m.weight, std=std_base)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    if residual_scaling:
        for block in net.model.blocks:
            nn.init.normal_(block.attn.proj.weight, std=std_residual)
            nn.init.normal_(block.mlp.fc2.weight, std=std_residual)
    if getattr(net.model, 'pos_embed', None) is not None:
        nn.init.normal_(net.model.pos_embed, std=std_base)
    if getattr(net.model, 'cls_token', None) is not None:
        nn.init.normal_(net.model.cls_token, std=std_base)


def initialize_linear(net, scale=None):
    if scale is None:
        scale=1
    for m in net.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            m.weight.data = m.weight.data * scale
            nn.init.zeros_(m.bias)


@contextmanager
def temp_seed(seed):
    '''
    Temporarily sets the seed for the random number generator
    '''
    if seed is None:
        yield
        return

    state = T.get_rng_state()
    T.manual_seed(seed)
    if T.cuda.is_available():
        cuda_state = T.cuda.get_rng_state()
        T.cuda.manual_seed(seed)

    try:
        yield
    finally:
        T.set_rng_state(state)
        if T.cuda.is_available():
            T.cuda.set_rng_state(cuda_state)


def initialize_net(net, scale=None, seed=None):

    with temp_seed(seed):
        if isinstance(net, Linear):
            initialize_linear(net, scale=scale)
        elif isinstance(net, MLP):
            initialize_mlp(net, scale=scale)
        elif isinstance(net, ResNet):
            initialize_resnet(net, scale=scale)
        elif isinstance(net, ResNetBN):
            initialize_resnet_bn(net, scale=scale, zero_init_residual=False)
        elif isinstance(net, CNN):
            initialize_cnn(net, scale=scale)
        elif isinstance(net, WideResNet):
            initialize_wrn(net, scale=scale)
        elif isinstance(net, WideResNetNoBN):
            initialize_wrn_fixup(net)
        elif isinstance(net, ViT):
            initialize_vit(net, scale=scale, residual_scaling=net.residual_scaling)
        elif isinstance(net, SSTTransformer):
            pass  # head zero-init in SSTTransformer.__init__; embeddings/blocks use PyTorch defaults
        else:
            raise ValueError("Unknown net type")



def get_path_of_last_net(path: Union[str, Path], not_final=False):
    path = Path(path)
    if path.is_dir():
        files = list(path.glob('*.pt'))
        if 'net_final.pt' in [file.name for file in files]:
            return path / 'net_final.pt'
        if len(files) == 0:
            return None
        files.sort(key=lambda x: x.stat().st_mtime)

        return files[-1]
    else:
        return path
