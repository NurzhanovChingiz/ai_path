# Train function
def train(model, data_loader, criterion, optimizer, device):
    model.train()
    optimizer.train()
    loss = 0.0
    size = len(data_loader)
    for b, (img, label) in data_loader:
        img, label = img.to(device), label.to(device)
        optimizer.zero_grad()
        pred = model(img)
        loss = criterion(pred, label)
        loss += loss.item()
        loss.backward()
        optimizer.step()
        if (b + 1) % 100 == 0:
            print(f"Batch [{b + 1}/{size}], Loss: {loss.item():.4f}")
