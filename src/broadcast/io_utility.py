import torch
import numpy as np
import imageio
import matplotlib.pyplot as plt
import open3d

def _load_image(path: str) -> np.ndarray:
    """
    Loads a single image from a given path.

    Args:
        path (str): the path of the depth image to load.

    Returns:
        np.ndarray: the with shape (H, W).
    """
    return imageio.imread(path)


def load_depth_image(path: str) -> np.ndarray:
    """
    Loads a single depth image from a given path.

    Args:
        path (str): the path of the depth image to load.

    Returns:
        np.array: the depth image with shape (H, W).
    """
    return _load_image(path)


def load_rgb_image(path: str, normalize: bool = True) -> np.ndarray:
    """
    Loads a single rgb image from a given path.

    Args:
        path (str): the path of the rgb image to load.
        normalize (bool, optional): whether to normalize the image in range [0, 1]. Defaults to True.

    Returns:
        np.array: the rgb image with shape (H, W, 3).
    """
    img = _load_image(path)
    if normalize:
        img = img.astype(np.float32)
        img /= 255.
    return img

def load_depth_image_as_tensor(path: str) -> torch.Tensor:
    """
    Loads a single depth image from a given path.

    Args:
        path (str): the path of the depth image to load.

    Returns:
        torch.Tensor: the depth image with shape (H, W).
    """
    return torch.tensor(load_depth_image(path).astype(np.float32))

def load_rgb_image_as_tensor(path: str, normalize: bool = True) -> torch.Tensor:
    """
    Loads a single rgb image from a given path.

    Args:
        path (str): the path of the rgb image to load.
        normalize (bool, optional): whether to normalize the image in range [0, 1]. Defaults to True.

    Returns:
        torch.Tensor: the rgb image with shape (H, W, 3).
    """
    torch.tensor(load_rgb_image(path, normalize=normalize))
    
def _visualize(image: np.ndarray) -> None:
    """
    Displays the image on the screen.

    Args:
        image (np.ndarray): the image.
    """
    plt.imshow(image)
    plt.show()
    
def visualize_depth(path: str) -> None:
    """
    Displays a depth image.

    Args:
        path (str): the path of the image.
    """
    depth = load_depth_image(path)
    _visualize(depth)

def visualize_rgb(path: str) -> None:
    """
    Displays an rgb image.

    Args:
        path (str): the path of the image.
    """
    rgb = load_rgb_image(path, False)
    _visualize(rgb)
    
def visualize_point_cloud(point_cloud: torch.Tensor) -> None:
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(point_cloud.detach().cpu().numpy())
    open3d.visualization.draw_geometries([pcd])

__all__ = ["load_depth_image",
           "load_rgb_image",
           "load_depth_image_as_tensor",
           "load_rgb_image_as_tensor"]

# use this to visualize images
if __name__ == "__main__":
    visualize_depth("./res/depth.png")
    visualize_rgb("./res/rgb.jpeg")