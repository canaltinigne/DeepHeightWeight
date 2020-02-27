import torch

"""
The Sorensen–Dice coefficient (Soft Implementation)
to evaluate intersection between segmentation mask
and target mask.

    Input parameters:
        input_: Output mask with size [Batch Size, 2, Height, Width].
        target: Target mask with size [Batch Size, 2, Height, Width].
        ch: Selected channel (actually redundant) to take mean for both channels.
        
    Output:
        - Pixelwise Mean Dice Coefficient between [0, 1]
"""
def dice_coef(input_, target, ch):
    smooth = 1e-6
    iflat = input_[:,ch,:,:]
    tflat = target[:,ch,:,:]
    intersection = (iflat * tflat).sum(dim=(2,1))
    return torch.mean((2. * intersection + smooth) / (iflat.sum(dim=(2,1)) + tflat.sum(dim=(2,1)) + smooth))

"""
Loss function to minimize for Dice coefficient maximization.

    Input parameters:
        input_: Output mask with size [Batch Size, 2, Height, Width].
        target: Target mask with size [Batch Size, 2, Height, Width].
        ch: Selected channel (actually redundant) to take mean for both channels.
        
    Output:
        - (1 - Dice Coefficient)
"""
def dice_loss(input_, target, ch):
    return 1-dice_coef(input_, target, ch)