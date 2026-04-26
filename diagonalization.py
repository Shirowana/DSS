import torch

def diagonalization(weights:list[torch.Tensor], offset:int=0, lr:float=0.01, max_iters:int=1000, init_A=None, init_B=None):
    """
    :param weights: K matrices, each of shape p*q
    :param offset: how many diagonals to keep (0 means only main diagonal, 1 means main + 1 off-diagonal, etc.)
    :param lr: learning rate for optimization
    :param max_iters: maximum number of optimization iterations
    :param init_A: optional initial A matrix of shape p*p
    :param init_B: optional initial B matrix of shape q*q
    :return: A, B such that A*weights[i]*B is approximate diagonal for all i
    """
    # assert all matrices have the same shape
    p, q = weights[0].shape
    for w in weights:
        assert w.shape == (p, q)

    # Initialize A and B as identity matrices
    A = torch.randn(p, p, requires_grad=True) if init_A is None else init_A.clone().requires_grad_(True)
    B = torch.randn(q, q, requires_grad=True) if init_B is None else init_B.clone().requires_grad_(True)

    # normalize weights
    weights = [w / w.norm() for w in weights]
    mask = (torch.abs(torch.arange(p).view(-1, 1) - torch.arange(q).view(1, -1)) <= offset).float()

    def loss_fn(A, B, weights, mask):
        loss = 0
        A_ = A / A.norm()
        B_ = B / B.norm()
        for w in weights:
            transformed = A_ @ w @ B_
            diag = transformed * mask
            off_diag = transformed - diag
            loss += off_diag.norm() ** 2 / diag.norm() ** 2
        return loss

    optimizer = torch.optim.Adam([A, B], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iters)
    for _ in range(max_iters):
        optimizer.zero_grad()
        loss = loss_fn(A, B, weights, mask)
        loss.backward()
        optimizer.step()
        scheduler.step()
    A, B = A / A.norm(), B / B.norm()
    return A.detach(), B.detach()

if __name__ == '__main__':
    torch.manual_seed(0)
    p, q, offset = 100, 100, 0
    A_or, B_or = torch.randn(p, q), torch.randn(p, q)
    A_inv, B_inv = torch.linalg.inv(A_or), torch.linalg.inv(B_or)
    weights = [A_inv @ torch.diag(torch.rand(p)) @ B_inv for _ in range(10)]

    A, B = diagonalization(weights, offset=offset, lr=0.01, max_iters=10000)
    mask = (torch.abs(torch.arange(p).view(-1, 1) - torch.arange(q).view(1, -1)) <= offset).float()
    diag = [A @ w @ B for w in weights]
    a1, a2 = torch.mean(torch.tensor([(od-od*mask).norm() for od in diag])), torch.mean(torch.tensor([torch.diag(d).norm() for d in diag]))
    b1, b2 = torch.mean(torch.tensor([(od-od*mask).norm() for od in weights])), torch.mean(torch.tensor([torch.diag(d).norm() for d in weights]))
    print("Average off-diagonal norm:", a1/a2, b1/b2, a1, a2, b1, b2)
    # W = A^(-1) @ Lambda @ B^(-1) should use inverse of A and B to reconstruct the original weights