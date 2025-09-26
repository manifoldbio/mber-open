import optax
from typing import Optional, Literal

def get_optimizer(
    optimizer_type: Literal["sgd", "adam", "adamw", "schedule_free_sgd", "schedule_free_adam"],
    learning_rate: float = 0.1,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    weight_decay: Optional[float] = None,
    weight_lr_power: float = 2.0,
    warmup_steps: Optional[int] = None,
    **kwargs
) -> optax.GradientTransformation:
    """
    Configure an optimizer treating all types uniformly.
    
    Args:
        optimizer_type: Type of optimizer
        learning_rate: Learning rate
        b1: Beta1 parameter
        b2: Beta2 parameter
        eps: Epsilon parameter for numerical stability
        weight_decay: Weight decay parameter
        weight_lr_power: Weight learning rate power (for schedule_free)
        warmup_steps: Number of warmup steps (for schedule_free)
        **kwargs: Additional optimizer-specific parameters
        
    Returns:
        Configured optimizer
    """
    # Standard optimizers
    if optimizer_type == "sgd":
        optimizer = optax.sgd(
            learning_rate=learning_rate,
            momentum=kwargs.get('momentum', 0.0),
        )
        
    elif optimizer_type == "adam":
        optimizer = optax.adam(
            learning_rate=learning_rate,
            b1=b1,
            b2=b2, 
            eps=eps,
        )
        
    elif optimizer_type == "adamw":
        optimizer = optax.adamw(
            learning_rate=learning_rate,
            b1=b1,
            b2=b2,
            eps=eps,
            weight_decay=weight_decay or 0.0,
        )
        
    # Schedule-free optimizers (treated like standard ones)
    elif optimizer_type == "schedule_free_adam":
        # Ensure b1 is positive
        if b1 <= 0:
            b1 = 0.01
            
        optimizer = optax.contrib.schedule_free_adamw(
            learning_rate=learning_rate,
            b1=b1,
            b2=b2,
            eps=eps,
            weight_decay=weight_decay or 0.0,
            weight_lr_power=weight_lr_power,
            warmup_steps=warmup_steps,
        )
        
    elif optimizer_type == "schedule_free_sgd":
        # Ensure b1 is positive
        if b1 <= 0:
            b1 = 0.01
            
        optimizer = optax.contrib.schedule_free_sgd(
            learning_rate=learning_rate,
            b1=b1,
            weight_decay=weight_decay,
            weight_lr_power=weight_lr_power,
            warmup_steps=warmup_steps,
        )
        
    else:
        raise ValueError(
            f"Unsupported optimizer: {optimizer_type}. "
            "Choose from: sgd, adam, adamw, schedule_free_sgd, schedule_free_adam"
        )
    
    # Apply weight decay for optimizers that don't have it built in
    if weight_decay is not None and weight_decay > 0 and optimizer_type in ["sgd", "adam"]:
        optimizer = optax.chain(
            optax.add_decayed_weights(weight_decay),
            optimizer
        )
    
    return optimizer