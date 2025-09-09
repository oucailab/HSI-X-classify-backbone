from torch import nn
import torch
import math
import torch.nn.functional as F

class MCA(nn.Module):
    def __init__(self, dropout):
        super(MCA, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x1, x2):
        # Assign x1 and x2 to query and key
        B1,C1,H1,W1 = x1.shape
        B2,C2,H2,W2 = x2.shape
        x1 = x1.view(B1,C1,H1*W1)
        x2 = x2.view(B2,C2,H2*W2)
        query = x1
        key = x2
        d = query.shape[-1]
        # Basic attention mechanism formula to get intermediate output A
        scores = torch.bmm(query, key.transpose(1, 2)) / math.sqrt(d)
        output_A = torch.bmm(self.dropout(F.softmax(scores, dim=-1)), x2)
        # Basic attention mechanism formula to get intermediate output B
        scores = torch.bmm(key, query.transpose(1, 2)) / math.sqrt(d)
        output_B = torch.bmm(self.dropout(F.softmax(scores, dim=-1)), x1)
        output_A = output_A.view(B1,C1,H1,W1)
        output_B = output_B.view(B2,C2,H2,W2)
        # Make the summation of the two intermediate outputs
        output = output_A + output_B  # shape (1280, 32, 60)
        return output_A, output_B