# Train function
def train(model, dataloader, loss_fn, optimizer, device):

    size = len(dataloader.dataset)  

    model.train()  
    for b, (X, y) in enumerate(dataloader):  
        X, y = X.to(device), y.to(
            device)  
        optimizer.zero_grad()  
        pred = model(X)  

        loss = loss_fn(pred, y)  

        loss.backward()  
        optimizer.step()  
        if (b + 1) % 100 == 0:
            loss, current = loss.item(), b * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
