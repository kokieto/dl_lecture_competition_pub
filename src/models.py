import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange


class BasicConvClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        seq_len: int,
        num_subjects: int,
        emb_dim: int,
        in_channels: int,
        hid_dim: int
    ) -> None:
        super().__init__()
        print(f"seq_len: {seq_len}, in_channels: {in_channels}")
        
        self.subject_embedding = nn.Embedding(num_subjects, emb_dim)
        
        self.blocks = nn.Sequential(
            ConvBlock(in_channels, hid_dim),
            ConvBlock(hid_dim, hid_dim),
            ConvBlock(hid_dim, hid_dim),
            nn.AdaptiveAvgPool1d(1),
            Rearrange("b d 1 -> b d"),
        )

        self.fc1 = nn.Linear(hid_dim + emb_dim, int(hid_dim*1.25))
        self.fc2 = nn.Linear(int(hid_dim*1.25), num_classes)
        
        self.dropout = nn.Dropout(0.25)

    def forward(self, X: torch.Tensor, subject_idxs: torch.Tensor) -> torch.Tensor:
        """_summary_
        Args:
            X ( b, c, t ): _description_
        Returns:
            X ( b, num_classes ): _description_
        """
        subject_features = self.subject_embedding(subject_idxs)
        X = self.blocks(X)
        X = torch.cat((X, subject_features), dim=1)
        X = F.gelu(self.fc1(X))
        X = self.dropout(X)
        
        return self.fc2(X)


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        kernel_size: int = 4, # もとは３
    ) -> None:
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.conv0 = nn.Conv1d(in_dim, out_dim, kernel_size, padding="same")
        self.conv1 = nn.Conv1d(out_dim, out_dim, kernel_size, padding="same")
        
        self.batchnorm0 = nn.BatchNorm1d(num_features=out_dim)
        self.batchnorm1 = nn.BatchNorm1d(num_features=out_dim)
       
        self.dropout = nn.Dropout(0.25)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.in_dim == self.out_dim:
            X = self.conv0(X) + X  # skip connection
        else:
            X = self.conv0(X)

        X = F.gelu(self.batchnorm0(X))

        X = self.conv1(X) + X  # skip connection
        X = F.gelu(self.batchnorm1(X))
    

        return self.dropout(X)
