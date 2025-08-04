from torch.utils.data import random_split,Dataset,DataLoader
import torch as t

class LSTMClassificationDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = t.FloatTensor(x_data)
        self.y_data = t.LongTensor(y_data)
    
    def __len__(self):
        return len(self.x_data)
    
    def __getitem__(self, index):
        return self.x_data(index), self.y_data(index)
    
def train_test_split(dataset,train_test_split,batch_size):
    train_size = int(train_test_split * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader