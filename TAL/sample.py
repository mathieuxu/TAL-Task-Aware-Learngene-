import numpy as np
import torch
import random

class TaskSampler:
    def __init__(self, train_loaders, dataset_sizes, temperature=10, batch_size=128):
        """
        Initialize the TaskSampler class.

        Args:
            train_loaders: A list of DataLoaders, one for each task.
            dataset_sizes: A list containing the size of each dataset.
            temperature: Temperature value to adjust task sampling weights.
            batch_size: Batch size, used to calculate `steps_per_epoch`.
        """
        self.train_loaders = train_loaders
        self.dataset_sizes = dataset_sizes
        self.temperature = temperature
        self.batch_size = batch_size
        self.weights = self.generate_tasks_distribution()
        self.steps_per_epoch = self.calculate_steps_per_epoch()

        # Use iterators to manage DataLoader iteration
        self.iterators = {i: iter(train_loader) for i, train_loader in enumerate(train_loaders)}

    def generate_tasks_distribution(self):
        """Calculate sampling weights for each task based on dataset size and temperature."""
        total_size = sum(self.dataset_sizes)
        weights = np.array([(size / total_size) ** (1.0 / self.temperature) for size in self.dataset_sizes])
        weights = weights / np.sum(weights)
        return torch.as_tensor(weights, dtype=torch.double)

    def calculate_steps_per_epoch(self):
        """Calculate the number of steps per epoch based on dataset sizes and batch size."""
        total_size = sum(self.dataset_sizes)
        return (total_size + self.batch_size - 1) // self.batch_size

    def sample_task(self):
        """Sample a task based on temperature-adjusted weights and return a batch and task index."""
        task_index = random.choices(range(len(self.train_loaders)), weights=self.weights.numpy(), k=1)[0]
        # Get the next batch from the persistent iterator
        try:
            images, labels = next(self.iterators[task_index])
        except StopIteration:
            # If the DataLoader is exhausted, reinitialize the iterator
            self.iterators[task_index] = iter(self.train_loaders[task_index])
            images, labels = next(self.iterators[task_index])
        return images, labels, task_index