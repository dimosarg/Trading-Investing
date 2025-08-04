import torch as t
import torch.nn as nn
import time
from torch import optim

def getdevice():
        device = t.device("cuda" if t.cuda.is_available() else "cpu")
        print(f"Using {device} as device\n")
        if device.type == "cuda":
             t.backends.cudnn.benchmark=True
             
        return device

def set_device_cpu():
     device=t.device("cpu")
     print(f"Using {device} as device\n")

     return device


def train_model(model:nn.Module, train_loader, test_loader, epochs, learning_rate, device=None, start_from_saved=False,
                checkpoint_path:str=None,target_val_loss=0.005,class_weights=5.0):
    """
    Train the super-resolution encoder-decoder model
    
    Args:
        model: Your encoder_decoder_cnn model
        train_loader: DataLoader with training data (low-res, high-res pairs)
        val_loader: DataLoader with validation data
        epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        device: Device to train on (cuda/cpu)
    """

    if device is None:
        device = getdevice()

    loss_list=[]
    val_loss_list=[]
    batch_loss_list=[]
    
    print(f"Using {device} as device\n")

    # Move model to device
    model = model.to(device)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    #Weights
    weights = t.tensor([1.0, class_weights, class_weights])

    # Loss function
    criterion = nn.CrossEntropyLoss(weight=weights)
        
    if start_from_saved==True:
         state=t.load(checkpoint_path,weights_only=True)
         model.load_state_dict(state['model_state_dict'])
         optimizer.load_state_dict(state['optimizer_state_dict'])
         loss=state['loss']
         epoch=state['epoch']
    
    val_loss=1
    
    # Training loop
    
    for epoch in range(epochs):
        if val_loss>=target_val_loss:
            model.train() 
            epoch_loss = 0.0
            start_time = time.time()
        
        
            for batch_idx, (x_data, y_data) in enumerate(train_loader):
                x_data = x_data.to(device)
                y_data = y_data.to(device)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = model(x_data)
                
                # Calculate loss
                loss = criterion(outputs, y_data)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()

                batch_loss_list.append(loss.item())
                
                epoch_loss += loss.item()
                
                # Print progress
                if batch_idx % 100 == 0:
                    print(f'Epoch: {epoch+1}/{epochs} | Batch: {batch_idx}/{len(train_loader)} | '
                        f'Batch Loss: {loss.item():.6f}')
            
            # Calculate epoch statistics
            avg_loss = epoch_loss / len(train_loader)
            epoch_time = time.time() - start_time
            loss_list.append(avg_loss)
            
            # Validation
            val_loss = validate_model(model, test_loader, criterion, device)
            val_loss_list.append(val_loss)

            print(f'Epoch {epoch+1} completed in {epoch_time:.2f}s | '
                f'Training Loss: {avg_loss:.6f} | Validation Loss: {val_loss:.6f}\n')
            # Save checkpoint
            if (epoch + 1) % 2 == 0:
                if start_from_saved==True:
                    split_vals=checkpoint_path.split(sep="_")
                    try:
                        splittier_vals=split_vals[2].split(sep=".")
                        previous_epoch_state=int(splittier_vals[0])
                        t.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': avg_loss,
                    }, f'checkpoints\\checkpoint_epoch_{epoch+1+previous_epoch_state}.pth')
                        start_from_saved=False
                    except:
                        print("Could not convert previous epoch state to integer\n")
                        t.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': avg_loss,
                        }, f'checkpoints\\checkpoint_epoch_{epoch+1}.pth')
                        start_from_saved=False
                    
                else:
                    t.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': avg_loss,
                    }, f'checkpoints\\checkpoint_epoch_{epoch+1}.pth')
            
        else:
            t.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss,
                }, f'checkpoints\\validation_loss_reached_epoch_{epoch+1}.pth')
            print(f"Validation loss target reached\nSaving model at epoch {epoch+1} and validation loss {val_loss:.6f}\n")
            return model, loss_list, val_loss_list, batch_loss_list
                
            
            
        
    return model, loss_list, val_loss_list, batch_loss_list

def validate_model(model, val_loader, criterion, device):
    """Validate the model on validation data"""
    model.eval()
    val_loss = 0.0
    
    with t.no_grad():
        for x_data, y_data in val_loader:
            x_data = x_data.to(device)
            y_data = y_data.to(device)
            
            outputs = model(x_data)
            loss = criterion(outputs, y_data)
            val_loss += loss.item()
    
    return val_loss / len(val_loader)

class LSTMClassificationModel(nn.Module):
    def __init__(self, num_x_data, num_classes, hidden_dim, num_hidden, dropout):
        super().__init__()
        
        hidden_layers = []
        num_classes = num_classes-1 #to compensate for indexing (num_classes = 3 , index is 0,1,2) 

        
        hidden_layers.append(nn.LSTM(num_x_data,hidden_dim,num_hidden, batch_first=True))
        hidden_layers.append(nn.Dropout(dropout))
        hidden_layers.append(nn.ReLU())
        hidden_layers.append(nn.Linear())

        self.lstm_model= nn.Sequential(*hidden_layers)
        self.attention = nn.Linear(hidden_dim,1)
        self.fc = nn.Linear(hidden_dim,num_classes)
    
    def forward(self,x):
        lstm_output,_ = self.lstm_model(x)
        attention_weights = t.softmax(self.attention(lstm_output,1))
        final = t.sum(attention_weights*lstm_output, dim=1)
        return self.fc(final)
        