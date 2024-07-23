import os
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.data = []  # Load your data here
        self.labels = []  # Load your labels here

        # Example of loading data
        for file_name in os.listdir(data_dir):
            file_path = os.path.join(data_dir, file_name)
            # Load data and label from file_path
            data_item = ...  # Implement data loading
            label_item = ...  # Implement label loading
            self.data.append(data_item)
            self.labels.append(label_item)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample, label

def get_dataloader(data_dir, batch_size, shuffle=True, transform=None):
    dataset = CustomDataset(data_dir, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader
