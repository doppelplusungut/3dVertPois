import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchExtractor(nn.Module):
    """Extracts patch features from a batch of feature maps given the patch
    centroids."""

    def __init__(self, patch_size, feature_extraction_model):
        super().__init__()

        self.patch_size = patch_size

        self.feature_extraction_model = feature_extraction_model

    def forward(self, x, centroids):
        """
        x: (B, C, H, W, D)
        centroids: (B, N, 3)
        """
        # Extract the patches
        patches = self.extract_patches(
            x, centroids
        )  # (B, N, C, patch_size, patch_size, patch_size)

        # Pass the patches through 3 conv layers
        B, N, C, P, _, _ = patches.shape
        patches = patches.view(B * N, C, P, P, P)  # Batch flatten
        out = self.feature_extraction_model(patches)  # (B*N, out_channels)
        out = out.view(B, N, -1)  # (B, N, out_channels)

        return out

    def extract_patches(self, vol, centroids):
        """Extract 3D patches from the input volume using slicing, ensuring all patches
        are of uniform size, applying zero padding where necessary.

        Parameters:
        - vol: Input volume tensor of shape (B, C, H, W, D).
        - centroids: Tensor of centroids of shape (B, N, 3).
        - patch_size: Integer, the size of the patch to extract.

        Returns:
        - Tensor of extracted patches of shape (B, N, C, patch_size, patch_size, patch_size).
        """
        patch_size = self.patch_size
        B, C, H, W, D = vol.shape
        patches = torch.zeros(
            B,
            centroids.shape[1],
            C,
            patch_size,
            patch_size,
            patch_size,
            device=vol.device,
        )

        for b in range(B):
            for n in range(centroids.shape[1]):
                x, y, z = centroids[b, n].long()
                z_min = max(z - patch_size // 2, 0)
                y_min = max(y - patch_size // 2, 0)
                x_min = max(x - patch_size // 2, 0)
                z_max = min(z + patch_size // 2 + 1, D)
                y_max = min(y + patch_size // 2 + 1, W)
                x_max = min(x + patch_size // 2 + 1, H)

                # Extract the patch
                patch = vol[b, :, x_min:x_max, y_min:y_max, z_min:z_max]

                # Calculate padding needed to reach the desired patch size
                pad = [
                    0,
                    patch_size - patch.size(3),  # Padding for X dimension
                    0,
                    patch_size - patch.size(2),  # Padding for Y dimension
                    0,
                    patch_size - patch.size(1),  # Padding for Z dimension
                ]

                # Apply padding
                patch_padded = F.pad(patch, pad, "constant", 0)
                patches[b, n] = patch_padded

        return patches
