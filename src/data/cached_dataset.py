from torch.utils.data import Dataset


class CachedSortedVVBPDataset(Dataset):
    """Dataset backed by cached raw sorted VVBP and target tensors."""
    def __init__(self, cache):
        self.values_sorted = cache["values_sorted"]
        self.target = cache["target"]
        self.center_base = cache["center_base"]
        self.local_3x3_base = cache["local_3x3_base"]

    def __len__(self):
        return self.target.shape[0]

    def __getitem__(self, idx):
        return (
            self.values_sorted[idx],
            self.target[idx],
            self.center_base[idx],
            self.local_3x3_base[idx],
        )
