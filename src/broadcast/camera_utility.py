import torch


def invert_intrinsic(K: torch.Tensor) -> torch.Tensor:
    """
    Inverts camera intrinsic matrix K.

        [fx, 0, cx]                     [1/fx,     0,   -cx/fx]
    K = [0, fy, cy]      =>     K_inv = [   0,  1/fy,   -cy/fy]
        [0,  0,  1]                     [   0,     0,        1]

    Args:
        K (torch.Tensor): a tensor representing the intrinsic matrix of shape (3, 3).

    Returns:
        torch.Tensor: a tensor of the inverted intrinsic matrix of shape (3, 3).
    """

    assert K.shape == (
        3, 3), f"invert_intrinsic: intrinsic matrix must have shape (3, 3), got {K.shape}."

    device = K.device
    K_inv = torch.eye(3, device=device)
    K_inv[0, 0] = 1. / K[0, 0]
    K_inv[1, 1] = 1. / K[1, 1]
    K_inv[0, 2] = -K[0, 2] / K[0, 0]
    K_inv[1, 2] = -K[1, 2] / K[1, 1]

    return K_inv

def scale_intrinsic(K: torch.Tensor, scale_factor: float) -> torch.Tensor:
    """
    Scales the intrinsic matrix of a specified value.

    Args:
        K (torch.Tensor): the intrinsic matrix, as a tensor of shape (3, 3)
        scale_factor (float): the scale factor to apply.

    Returns:
        torch.Tensor: the scaled intrinsic matrix.
    """

    assert K.shape == (
        3, 3), f"scale_intrinsic: intrinsic matrix must have shape (3, 3), got {K.shape}."

    new_K = torch.eye(3, device=K.device)
    new_K[:2, :] = K[:2, :] * scale_factor
    return new_K


__all__ = ["invert_intrinsic",
           "scale_intrinsic"]
