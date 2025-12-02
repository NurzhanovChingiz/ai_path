
def summary(model):
    """
    Print information about the model
    
    Args:
        model: PyTorch model
    """
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    total_params = trainable_params + non_trainable_params
    print("\n" + "="*50)
    print("Model Information")
    print("="*50)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    # Memory estimate
    param_size = total_params * 4 / (1024**2)  # Assuming float32
    print(f"Estimated size: {param_size:.2f} MB")
    print("="*50)
