import time
import argparse
import torch
import torchvision
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from data_loaders.slide_loader import SlideLoader
from models.segment_qtransformer import Segment
from models.qresnet import DressedQuantumNet


def train_model():
    parser = argparse.ArgumentParser(description="Trains the quantum Resnet model.")
    parser.add_argument("--pretrain_cnn", type=str, default='', help="Directory to pretrained weights of cnn")
    parser.add_argument("--training_img_dir", type=str, default='data/slide', help="Directory to images")
    parser.add_argument("--runs", type=str, default='runs/exp_qtransformer1/',
                        help="Directory to tensorboard folder")
    args = parser.parse_args()

    # Initialize dataloader
    dataloader = SlideLoader(1, True, args.training_img_dir).loader()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(args.runs)

    # Model CNN
    model_hybrid = torchvision.models.resnet18()
    model_hybrid.fc = DressedQuantumNet()
    model_hybrid.load_state_dict(torch.load('/home/hades/Desktop/q/runs/exp_qresnet3/best_checkpoint_14.pth'))
    model_hybrid.to(device)

    vocab_size = 1000
    num_heads = 2
    num_blocks = 2
    num_classes = 2
    embed_dim = 3744

    qtrasnsformer = Segment(embed_dim=embed_dim, num_heads=num_heads, num_blocks=num_blocks, num_classes=num_classes,
                            vocab_size=vocab_size, n_qubits_transformer=0, n_qubits_ffn=1)
    qtrasnsformer.to(device)

    for img in dataloader:
        # batch x c x h x w
        img = img.to(device)
        print(img.size())
        f = model_hybrid(img)
        print(f.size())
        batch, c, h, w = f.size()
        # batch x len x feature
        f = f.view(batch, c, -1).long()

        # Feed forward
        output = qtrasnsformer(f)
        print(output.size())


if __name__ == "__main__":
    train_model()