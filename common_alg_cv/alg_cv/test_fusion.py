import torch
import copy
from alg_cv.fusion import fuse_model_inplace

def test(model, dataloader, loss_fn, device, mode="Test"):
    tmp = copy.deepcopy(model).to(device)
    tmp.eval()

    fuse_model_inplace(tmp)
    test_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            pred = tmp(X)
            test_loss += loss_fn(pred, y).item()
            total += y.size(0)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    loss = test_loss / batch_idx

    correct = 100.*correct/total
    print(f"{mode} Error:\n Accuracy: {(correct):>.1f}%, Avg loss: {(loss):>.8f}\n")

