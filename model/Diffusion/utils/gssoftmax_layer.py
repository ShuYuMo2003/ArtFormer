# This code is adapted from `https://github.com/bshall/VectorQuantizedVAE/blob/master/model.py#L61`
import math
import torch
from rich import print
from torch import nn
import torch.nn.functional as F
from torch.distributions import RelaxedOneHotCategorical

class VQEmbeddingGSSoft(nn.Module):
    def __init__(self, latent_dim, num_embeddings, embedding_dim):
        super(VQEmbeddingGSSoft, self).__init__()

        self.embedding = nn.Parameter(torch.Tensor(latent_dim, num_embeddings, embedding_dim))
        nn.init.uniform_(self.embedding, -1/num_embeddings, 1/num_embeddings)

    def forward(self, x=None, logits=None, tau=None):
        """
        x: (B, C=(N*D), H=1, W=1)
        logits: (B, N, M)
        """
        assert x is not None or logits is not None, "Should provide x or logits"

        N, M, D = self.embedding.size()

        if x is not None:
            B, C, H, W = x.size()
            assert C == N * D
            # (B, N, D, H, W) --> (N, B, H, W, D)
            x = x.view(B, N, D, H, W).permute(1, 0, 3, 4, 2)
            x_flat = x.reshape(N, -1, D)

            distances = torch.baddbmm(torch.sum(self.embedding ** 2, dim=2).unsqueeze(1) +
                                    torch.sum(x_flat ** 2, dim=2, keepdim=True),
                                    x_flat, self.embedding.transpose(1, 2),
                                    alpha=-2.0, beta=1.0)
            distances = distances.view(N, B, H, W, M)
            logits = -distances
        else:
            B, N, M = logits.size()
            # (B, N, M) --> (N, B, M)
            logits = logits.permute(1, 0, 2)
            # (N, B, M) --> (N, B, H=1, W=1, M)
            logits = logits.unsqueeze(2).unsqueeze(2)

        # print(logits.shape)

        dist = RelaxedOneHotCategorical(tau, logits=logits)

        # if self.training:
        samples = dist.rsample().view(N, -1, M)

        # else:
        #     samples = torch.argmax(dist.probs, dim=-1)
        #     samples = F.one_hot(samples, M).float()
        #     samples = samples.view(N, -1, M)

        quantized = torch.bmm(samples, self.embedding)
        quantized = quantized.view(N, B, 1, 1, D)

        KL = dist.probs * (dist.logits + math.log(M))
        KL[(dist.probs == 0).expand_as(KL)] = 0
        KL = KL.sum(dim=(0, 2, 3, 4)).mean()

        avg_probs = torch.mean(samples, dim=1)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10), dim=-1))

        # (N, B, H=1, W=1, M) --> (B, N, M, H=1, W=1)
        ret_logits = dist.logits.permute(1, 0, 4, 2, 3)

        return quantized.permute(1, 0, 4, 2, 3), KL, perplexity.sum(), ret_logits