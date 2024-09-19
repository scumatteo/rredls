import torch

def invert_transformation_matrices(T: torch.Tensor) -> torch.Tensor:
    """
    Inverts transformation matrices.

    Args:
        T (torch.Tensor): the transformation matrices to invert, a tensor of shape (..., D+1, D+1). D+1 because for example if we're in 3D, a transformation matrix has shape (4, 4).

    Returns:
        torch.Tensor: the inverted transformation matrices, as a tensor of shape (..., D+1, D+1).

    A transformation matrix is a squared matrix composed of a orthogonal matrix of shape (D, D) which is the rotation matrix R
    and a translation vector t of shape (D,) as follow:

    T = [R, t]
        [0ⁿ⁻¹, 1]

    To speed up the computation, the inverse of a transformation matrix can be computed as follow:
    inv(R) = transpose(R) since it's orthogonal;
    inv(t) = -transpose(R) ⋅ t;

    T⁻¹ = [R⁻¹, -R⁻¹ ⋅ t]
          [0ⁿ⁻¹, 1]
    """

    assert T.shape[
        -2] == T.shape[-1], f"invert_transformation_matrices: transformation matrices must have shape (..., D, D), got {T.shape}."

    dims = T.dim()
    D = T.shape[-1] - 1

    T_inv = T.clone()
    T_inv[..., :D, :D] = torch.transpose(T[..., :D, :D], dims - 2, dims - 1)
    T_inv[..., :D, D] = torch.einsum("...ij,...jk->...ik", -T_inv[..., :D, :D], T[..., :D, D, None]).squeeze(-1)
    return T_inv


def compose_transformation_matrices(R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Creates transformation matrices given rotation matrices and translation vectors.

    Args:
        R (torch.Tensor): the rotation matrices, as tensor of shape (..., D, D).
        t (torch.Tensor): the translation vectors, as tensor of shape (..., D,) or (..., D, 1).

    Returns:
        torch.Tensor: transformation matrices as tensor of shape (..., D+1, D+1).
    """
    assert R.device == t.device, f"compose_transformation_matrices: R and t must be on the same device, got {R.device}, {t.device}"
    assert R.shape[
        -2] == R.shape[-1], f"compose_transformation_matrices: R must have shape (..., D, D), got {R.shape}."
    assert t.shape[-1] == R.shape[-1] or (t.shape[-1] == 1 and t.shape[-2] == R.shape[-1]
                                          ), f"compose_transformation_matrices: t must have shape (..., D,) or (..., D, 1), got {t.shape}."

    D = R.shape[-1]
    batch_dims = R.shape[:-2]
    T = torch.zeros(*batch_dims, D+1, D+1, device=R.device)
    T[..., :D, :D] = R
    T[..., :D, D] = t
    T[..., D, D] = 1.

    return T


def concatenate_transformations(initial: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    """
    Concatenates relative transformations in cascade to a initial one.

    Args:
        initial (torch.Tensor): the initial transformation, as tensor of shape (D+1, D+1).
        T (torch.Tensor): the transformation matrices to apply, as tensor of shape (N, D+1, D+1).

    Returns:
        torch.Tensor: transformation matrices that are the concatenation of relative transformations, as tensor of shape (N, D+1, D+1).

    In this case is not possible to broadcast the operations because they are inherently sequential.
    """
    assert initial.device == T.device, f"concatenate_transformations: transformation matrices must be on the same device, got {initial.device}, {T.device}"
    assert initial.shape[-2] == initial.shape[-1] == T.shape[-2] == T.shape[
        -1], f"concatenate_transformations: transformation matrices must have last two dimensions of shape (D+1, D+1), got {initial.shape[-2]}, {T.shape[-2]}"
    assert initial.dim(
    ) == 2, f"concatenate_transformations: initial must be of shape (D+1, D+1), got {initial.shape} instead"
    assert T.dim(
    ) == 3, f"concatenate_transformations: initial must be of shape (N, D+1, D+1), got {T.shape} instead"

    device = T.device
    batch_dims = T.shape[0]
    D = T.shape[-1] - 1
    T_concat = torch.zeros(batch_dims + 1, D + 1, D + 1, device=device)
    T_concat[0] = initial
    for i in range(batch_dims):
        T_concat[i + 1] = T_concat[i] @ T[i]

    return T_concat


def compute_relative_transformations(T: torch.Tensor) -> torch.Tensor:
    """
    Computes relative transformations between consecutive transformation matrices.

    Args:
        T (torch.Tensor): the transformation matrices, as tensor of shape (..., N, D+1, D+1).

    Returns:
        torch.Tensor: relative transformation matrices, as tensor of shape (..., N-1, D+1, D+1).

    R_rel = inv(R_1) · R_2
    t_rel = inv(R_1) · (t_2 - t_1)

    where R_1, t_1 and R_2, t_2 are two consecutive rotation matrices and translation vectors respectively, that form two consecutive
    transformation matrices together.

    So, it is equivalent to:
    T_rel = inv(T_1) · T_2

    All the transformations can be computed in parallel, by duplicating the T tensor.

    Input tensor = [...][T_1, T_2, ..., T_N]

    prev_T = [...][T_1, T_2, ..., T_N-1]
    next_T = [...][T_2, T_3, ..., T_N]

    So, the prev_T are inverted and the dot product is computed for the entire batch in parallel with the next_T.

    relative_T = [...][relative_T_1_T_2, relative_T_2_T_3, ..., relative_T_N-1_T_N]
    """
    assert T.shape[-2] == T.shape[-1]

    prev_T = T[..., :-1, :, :]
    next_T = T[..., 1:, :, :]

    prev_T_inv = invert_transformation_matrices(prev_T)
    relative_T = torch.einsum("...ij,...jk->...ik", prev_T_inv, next_T)
    return relative_T


__all__ = ["invert_transformation_matrices",
           "compose_transformation_matrices",
           "concatenate_transformations",
           "compute_relative_transformations"
           ]
