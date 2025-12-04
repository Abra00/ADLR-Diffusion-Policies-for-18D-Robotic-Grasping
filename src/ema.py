import torch

class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.original = {}

        # Save a copy of every trainable parameter
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model):
        # Update the shadow copy: Shadow = (1-decay)*Current + decay*Shadow
        for name, param in model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self, model):
        """Put the EMA weights into the main model"""
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.original[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self, model):
        """Restore the original noisy weights"""
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data = self.original[name]