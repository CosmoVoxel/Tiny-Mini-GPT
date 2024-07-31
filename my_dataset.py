import torch
import torch.utils.data
import numpy as np
import tqdm


def plot_token_length_distribution(tokens):
    import matplotlib.pyplot as plt
    # Calculate the token lengths
    token_lengths = np.array([len(token) for token in tokens])

    # Create the histogram
    hist, bins = np.histogram(token_lengths, bins=50)

    # Plot the histogram
    plt.bar(bins[:-1], hist, width=np.diff(bins), align='edge')
    plt.xlabel('Token Length')
    plt.ylabel('Frequency')
    plt.title('Token Length Distribution')
    plt.show()

def group_nested_tokens_by_length(nested_array):
    lengths = np.array([len(arr) for arr in nested_array])
    sorted_indices = np.argsort(lengths)
    unique_lengths = np.unique(lengths)
    grouped_by_length = {length: [] for length in unique_lengths}

    for idx in sorted_indices:
        length = lengths[idx]
        grouped_by_length[length].append(nested_array[idx])

    return grouped_by_length

class BucketsDataset(torch.utils.data.Dataset):
    def __init__(self, grouped_by_length,batch_size=2,max_length=512):
        self.grouped_by_length = grouped_by_length
        self.lengths = list(grouped_by_length.keys())
        self.buckets = []
        self.batch_size = batch_size
        self.max_length = max_length

        self._create_buckets()

    def _create_buckets(self):
        for length, group in self.grouped_by_length.items():
            if self.max_length >= length:
                for i in range(0, len(group), self.batch_size):
                    self.buckets.append(group[i:i + self.batch_size])

    def __len__(self):
        return len(self.buckets)

    def __getitem__(self, idx): 
        return self.buckets[idx]

def collate_fn(batch):
    input_text = torch.tensor(batch).squeeze(0)
    output_text = input_text.clone()
    return input_text[:,:-1],output_text[:,1:]

def load_tokenized_text_with_nested_arrays(file):
    tokens = np.load(file,allow_pickle=True)
    tokens = tokens[tokens.astype(bool)]
    return tokens