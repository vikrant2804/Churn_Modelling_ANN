import torch

def mixup_data(x, y, alpha=0.2):
    if alpha > 0:
        lam = torch.distributions.beta.Beta(alpha, alpha).sample().item()
        batch_size = x.size()[0]
        index = torch.randperm(batch_size).to(x.device)
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam
    return x, y, y, 1.0