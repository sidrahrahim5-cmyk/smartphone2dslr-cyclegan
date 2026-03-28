import random
import torch


class ReplayBuffer:
    """
    Image history buffer to update discriminators.

    Paper Section 4: To reduce model oscillation, discriminators
    are updated using a history of 50 previously generated images
    rather than only the latest output.

    This prevents the generator from fooling a discriminator that
    has forgotten previous mistakes.
    """

    def __init__(self, max_size=50):
        assert max_size > 0, "Buffer size must be positive"
        self.max_size = max_size
        self.buffer   = []

    def push_and_pop(self, data):
        """
        Add new images to buffer, return a mix of
        new and previously stored images.

        Args:
            data: tensor of shape [B, C, H, W]

        Returns:
            tensor of same shape — mix of old and new images
        """
        result = []

        for element in data.data:
            element = element.unsqueeze(0)

            if len(self.buffer) < self.max_size:
                # Buffer not full yet — just store and return
                self.buffer.append(element)
                result.append(element)
            else:
                # 50% chance: return a stored image, replace it
                # 50% chance: return the new image directly
                if random.random() > 0.5:
                    idx = random.randint(0, self.max_size - 1)
                    stored = self.buffer[idx].clone()
                    self.buffer[idx] = element
                    result.append(stored)
                else:
                    result.append(element)

        return torch.cat(result, dim=0)