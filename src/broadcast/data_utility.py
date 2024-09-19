import torch


def to_homogeneous(points: torch.Tensor) -> torch.Tensor:
    """
    Converts points to homogeneous coordinates. It adds 1 to the last dimension.
    Example: if the points are 2D with shape (N, 2), it returns 3D points (N, 3) where the z element of the points is set to 1.

    Args:
        points (torch.Tensor): the points to transform to homogeneous coordinates.

    Returns:
        torch.Tensor: the points in homogeneous coordinates.
    """
    ones = torch.ones(*points.shape[:-1], 1,
                      device=points.device, dtype=points.dtype)
    return torch.cat([points, ones], dim=-1)


def from_homogeneous(points: torch.Tensor) -> torch.Tensor:
    """
    Converts points from homogeneous to cartesian coordinates. It removes the last dimension and normalizes by the value in the last dimension.

    Args:
        points (torch.Tensor): the points to transform to cartesian coordinates.

    Returns:
        torch.Tensor: the points in cartesian coordinates.
    """
    return points[..., :-1] / points[..., -1, None]


def generate_grid(width: int, height: int, device: torch.device) -> torch.Tensor:
    """
    Generates a 2D grid of shape (width, height, 2) with coordinates of each point in the grid.

    Args:
        width (int): the number of columns in the grid (x-axis).
        height (int): the number of rows in the grid (y-axis).
        device (torch.device): the device on which the grid should be created (e.g., 'cpu' or 'cuda').

    Returns:
        torch.Tensor: a tensor of shape (width, height, 2) where each element represents the 
                      (x, y) coordinates of the corresponding point in the grid.
    """
    x_coords = torch.arange(width, device=device)
    y_coords = torch.arange(height, device=device)
    grid = torch.stack(torch.meshgrid(
        x_coords, y_coords, indexing='ij'), dim=-1)
    return grid


def linearize_points(points: torch.Tensor) -> torch.Tensor:
    """
    Linearizes points, transforming them from shape (..., W, H, C) to shape (..., W*H, C).

    Args:
        points (torch.Tensor): the points of shape (..., W, H, C).

    Returns:
        torch.Tensor: the linearized points of shape (..., W*H, C).
    """
    device = points.device
    batch_dims = points.shape[:-3]
    W, H, C = points.shape[-3], points.shape[-2], points.shape[-1]
    return points.view(*batch_dims, H * W, C).to(device)


def depth_images_to_point_clouds(depth_images: torch.Tensor, K_inv: torch.Tensor, depth_scale: float = 0.001) -> torch.Tensor:
    """
    Creates point clouds from depth images.

    Args:
        depth_images (torch.Tensor): the depth images, as tensor of shape (..., H, W).
        K_inv (torch.Tensor): the inverse of the intrinsic matrix, as tensor of shape (3, 3).
        depth_scale (float, optional): value to transform any pixel values from depth images to meters.
                                       Example: if depth is the distance in millimeters, then meters = 0.001 * millimeters. Defaults to 0.001.

    Returns:
        torch.Tensor: the point clouds, as a tensor of shape (..., H*W, 3).

    [X]   [1/fx     0       -cx/fx]   [x]
    [Y] = [0        1/fy    -cy/fy] * [y] * Z
    [Z]   [0        0        1    ]   [1]

    where Z is the depth in meter for each point.
    """

    assert K_inv.shape == (
        3, 3), f"depth_images_to_pointclouds: intrinsic matrix must have shape (3, 3), got {K_inv.shape}."

    device = depth_images.device
    height, width = depth_images.shape[-2], depth_images.shape[-1]
    batch_dims = depth_images.shape[:-2]
    depth_permuted = torch.einsum(
        "...ij->...ji", depth_images).to(device)  # (..., W, H)
    image_coordinates = generate_grid(width, height, device)  # (W, H, 2)
    homogeneus_coordinates_points = to_homogeneous(
        image_coordinates).float()  # (W, H, 3)

    target_shape = (*batch_dims, width, height, 3)
    expanded_homogeneus_coordinates_points = homogeneus_coordinates_points.view(
        target_shape)  # (..., W, H, 3)

    Z = (depth_permuted * depth_scale).unsqueeze(-1).repeat(*([1] * depth_images.dim()), 3)  # (..., W, H, 3)
    
    # (..., W, H, 3), 3D points projected with intrinsic matrix
    points_3D = torch.einsum(
        "ij,...jk->...ik", K_inv, expanded_homogeneus_coordinates_points[..., None] * Z[..., None]).squeeze(-1)  # (..., W, H, 3)
    # (..., H*W, 3) linearized point clouds
    point_clouds = linearize_points(points_3D)
    return point_clouds


def apply_transformations_to_point_clouds(point_clouds: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    """
    Applies transformations to points clouds.

    Args:
        point_clouds (torch.Tensor): the point clouds, as tensor of shape (..., N, 3) or (..., N, 6) if colorized (X, Y, Z, R, G, B).
        T (torch.Tensor):  the transformation matrices, as tensor of shape (..., 4, 4).

    Returns:
        torch.Tensor: the transformed point clouds, as a tensor of shape (..., N, 3) or (..., N, 6) if colorized (X, Y, Z, R, G, B).
    """
    assert point_clouds.device == T.device, f"apply_transformations_to_point_clouds: point clouds and transformation matrices should be on same device! Got {point_clouds.device} and {T.device} instead"
    assert point_clouds.shape[-1] == 3, f"apply_transformations_to_point_clouds: point clouds last dimension must have shape = 3, got {point_clouds.shape[-1]}"
    assert T.shape[-1] == T.shape[-2] == 4, f"apply_transformations_to_point_clouds: transformation matrices must have shape (..., 4, 4), got {T.shape} instead"

    colorized = point_clouds.shape[-1] > 3

    # (..., N, 4) also if colorized, we don't want to apply the transformation to the RGB colors.
    homogeneus_coordinates_point_clouds = to_homogeneous(point_clouds[..., :3])
    homogeneous_coordinates_transformed_point_clouds = torch.einsum(
        "...ij,...njk->...nik", T, homogeneus_coordinates_point_clouds[..., None]).squeeze(-1)  # (..., N, 4)
    transformed_point_clouds = from_homogeneous(
        homogeneous_coordinates_transformed_point_clouds)  # (..., N, 3)

    if colorized:
        transformed_point_clouds = torch.cat(
            [transformed_point_clouds, point_clouds[..., 3:]], dim=-1)  # (..., N, 6)

    return transformed_point_clouds


__all__ = ["to_homogeneous",
           "from_homogeneous",
           "generate_grid",
           "linearize_points",
           "depth_images_to_point_clouds",
           "apply_transformations_to_point_clouds"
           ]
