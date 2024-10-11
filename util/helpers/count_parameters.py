def model_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params

    return {
        "Trainable Parameters": trainable_params,
        "Non-trainable Parameters": non_trainable_params,
        "Total Parameters": total_params,
    }
