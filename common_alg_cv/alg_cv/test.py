import torch

def test(model, data_loader, criterion, optimizer, device):
    model.eval()
    optimizer.eval()
    size = len(data_loader)
    loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for b, (img, label) in enumerate(data_loader):
            img, label = img.to(device), label.to(device)
            pred = model(img)
            loss += criterion(pred, label).item()
            total += label.size(0)
            correct += (pred.argmax(1) == label).type(torch.float).sum().item()
    loss = loss / (b + 1)
    correct = 100.*correct/total
    if (b + 1) % 100 == 0:
        print(f"Test batch [{b + 1}/{size}], Test AVG Loss: {loss:.4f}, Test Accuracy: {correct:.2f}%")
