from models.qtransformer import TransformerBlockQuantum, TransformerBlockClassical, PositionalEncoder
import math

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence


class Segment(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 num_blocks: int,
                 num_classes: int,
                 vocab_size: int,
                 ffn_dim: int = 32,
                 n_qubits_transformer: int = 0,
                 n_qubits_ffn: int = 0,
                 n_qlayers: int = 1,
                 dropout=0.1,
                 q_device="device.qubit"):
        super(Segment, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.num_classes = num_classes
        self.vocab_size = vocab_size

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = PositionalEncoder(embed_dim)

        print(f"++ There will be {num_blocks} transformer blocks")

        if n_qubits_transformer > 0:
            print(f"++ Transformer will use {n_qubits_transformer} qubits and {n_qlayers} q layers")

            if n_qubits_ffn > 0:
                print(f"The feed-forward head will use {n_qubits_ffn} qubits")
            else:
                print(f"The feed-forward head will be classical")

            print(f"Using quantum device {q_device}")

            transformer_blocks = [
                TransformerBlockQuantum(embed_dim, num_heads, ffn_dim,
                                        n_qubits_transformer=n_qubits_transformer,
                                        n_qubits_ffn=n_qubits_ffn,
                                        n_qlayers=n_qlayers,
                                        q_device=q_device) for _ in range(num_blocks)
            ]
        else:
            transformer_blocks = [
                TransformerBlockClassical(embed_dim, num_heads, ffn_dim) for _ in range(num_blocks)
            ]

        self.transformers = nn.Sequential(*transformer_blocks)

        if self.num_classes > 2:
            self.class_logits = nn.Linear(embed_dim, num_classes)
        else:
            self.class_logits = nn.Linear(embed_dim, 1)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # tokens = self.token_embedding(x)
        # batch_size, seq_len, embed_dim = x.size()
        x = self.pos_embedding(x)
        x = self.transformers(x)
        x = x.mean(dim=1)  # global average pooling, works in 1D
        x = self.dropout(x)
        # x = self.class_logits(x)
        # return F.log_softmax(x, dim=1)
        return self.class_logits(x)


if __name__ == '__main__':
    from data_loaders.slide_loader import SlideLoader

    img_dir = '/home/hades/Desktop/q/data/slide'
    dataloader = SlideLoader(1, True, img_dir).loader()

    cnn_model = torchvision.models.resnet18(pretrained=True)
    cnn_model  = torch.nn.Sequential(*(list(cnn_model.children())[:-2]))

    vocab_size = 1000
    num_heads = 2
    num_blocks = 2
    num_classes = 2
    embed_dim = 3744

    qtrasnsformer = Segment(embed_dim=embed_dim, num_heads=num_heads, num_blocks=num_blocks, num_classes=num_classes,
                            vocab_size=vocab_size, n_qubits_transformer=0, n_qubits_ffn=1)

    for img in dataloader:
        # batch x c x h x w
        f = cnn_model(img)
        batch, c, h, w = f.size()
        # batch x len x feature
        f = f.view(batch, c, -1).long()
        print(f.size())

        # Feed forward
        output = qtrasnsformer(f)
        print(output.size())

