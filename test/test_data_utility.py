import torch

import sys
sys.path.insert(0, "src/")

from utility.benchmarking_utility import evaluate_time
from broadcast.data_utility import (
    to_homogeneous,
    from_homogeneous,
    generate_grid,
    linearize_points,
    depth_images_to_point_clouds,
    apply_transformations_to_point_clouds)
from broadcast.camera_utility import invert_intrinsic
from broadcast.io_utility import load_depth_image_as_tensor, visualize_point_cloud

def test_to_homogeneous(points: torch.Tensor) -> torch.Tensor:
    homogeneous_points = to_homogeneous(points)
    assert homogeneous_points.shape[-1] == points.shape[-1] + 1
    assert torch.allclose(homogeneous_points[..., -1], torch.ones(
        points.shape[:-1], device=homogeneous_points.device))
    return homogeneous_points


def test_from_homogeneous(homogeneous_points: torch.Tensor) -> torch.Tensor:
    points = from_homogeneous(homogeneous_points)
    assert points.shape[-1] == homogeneous_points.shape[-1] - 1
    assert torch.allclose(
        points, homogeneous_points[..., :-1] / homogeneous_points[..., -1, None])
    return points


def test_generate_grid(width: int, height: int, device: str) -> torch.Tensor:
    grid = generate_grid(width, height, device)
    assert grid.shape == (width, height, 2)
    assert torch.allclose(grid[:, 0, 0], torch.arange(0, width, device=device))
    assert torch.allclose(
        grid[0, :, 1], torch.arange(0, height, device=device))
    return grid


def test_linearize_points(points: torch.Tensor) -> torch.Tensor:
    linearized_points = linearize_points(points)
    height, width = points.shape[-3], points.shape[-2]
    assert linearized_points.shape[-2] == width * height


def test_depth_images_to_point_clouds(depth_images: torch.Tensor, K_inv: torch.Tensor, depth_scale: float = 0.001) -> torch.Tensor:
    point_clouds: torch.Tensor = depth_images_to_point_clouds(depth_images, K_inv, depth_scale)
    assert point_clouds.shape[-2] == depth_images.shape[-2] * depth_images.shape[-1]
    return point_clouds

if __name__ == "__main__":
    device = "cpu"

    points = torch.rand(5, 2, 10, 2, device=device)

    homogeneous_points = test_to_homogeneous(points)
    test_from_homogeneous(homogeneous_points)

    width, height = 1920, 1280
    grid = test_generate_grid(width, height, device=device)

    test_linearize_points(homogeneous_points)

    K = torch.tensor([960., 0., 960., 0., 960., 540., 0., 0., 1.], device=device).reshape((3, 3))
    K_inv = invert_intrinsic(K)
    depth_image = load_depth_image_as_tensor("./res/depth.png").unsqueeze(0)
    point_cloud = test_depth_images_to_point_clouds(depth_image, K_inv)
    visualize_point_cloud(point_cloud[0])

    
    # rotation of 90° around x-axis, 30° around y-axis and 45° around z-axis
    T = torch.tensor([0.6660276, -0.1056860,  0.7384021, 0, 0.7222504, -0.1560583, -0.6737953, 0, 0.1864445,  0.9820775, -0.0276074, 1, 0, 0, 0, 1]).view((4, 4)).unsqueeze(0)
    transformed_pcd = apply_transformations_to_point_clouds(point_cloud, T)
    visualize_point_cloud(transformed_pcd[0])
    
    # test with more dimensions
    depth_image = depth_image.unsqueeze(0)
    point_cloud = test_depth_images_to_point_clouds(depth_image, K_inv)
    visualize_point_cloud(point_cloud[0, 0])
    transformed_pcd = apply_transformations_to_point_clouds(point_cloud, T)
    visualize_point_cloud(transformed_pcd[0, 0])
    
    
    