def check_dimensions_and_sanity(dataloader, device):
    """
    Utility function to extract dimensions and perform sanity checks after applying transformations.

    Args:
        dataloader (torch.utils.data.DataLoader): DataLoader containing the dataset.
        device (torch.device): Device for computation (e.g., 'cuda' or 'cpu').

    Returns:
        None
    """
    for batch_idx, batch in enumerate(dataloader):
        # Extract image and mask from the batch
        images, masks = batch['image'], batch['mask']

        # Move tensors to the specified device
        images, masks = images.to(device), masks.to(device)

        # Extract image dimensions
        batch_size, channels, height, width = images.shape
        print(f"Images: (batch_size={batch_size}, channels={channels}, height={height}, width={width})")

        # Handle cases where masks may have different shapes
        if masks.dim() == 4:  # Masks with channel dimension
            mask_batch, mask_channels, mask_height, mask_width = masks.shape
            print(f"Masks: (batch_size={mask_batch}, channels={mask_channels}, height={mask_height}, width={mask_width})")
        elif masks.dim() == 3:  # Masks without channel dimension
            mask_batch, mask_height, mask_width = masks.shape
            print(f"Masks: (batch_size={mask_batch}, height={mask_height}, width={mask_width})")
        else:
            raise ValueError(f"Unexpected mask dimensions: {masks.shape}")

        # Sanity checks
        assert images.dim() == 4, "Images should be 4D (batch_size, channels, height, width)"
        assert masks.dim() in [3, 4], "Masks should be 3D or 4D"
        assert images.size(2) == masks.size(-2), "Image and mask heights must match"
        assert images.size(3) == masks.size(-1), "Image and mask widths must match"

        # Stop after checking the first batch
        break
