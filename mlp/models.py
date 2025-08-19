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

def overfit_test(model,train_loader,device):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    x_data, y_data = next(iter(train_loader))
    x_data = x_data.to(device)
    y_data = y_data.to(device)

    for i in range(300):
        optimizer.zero_grad()
        outputs = model(x_data)
        loss = criterion(outputs, y_data)
        loss.backward()
        optimizer.step()

        if i % 50 == 0:
            preds = t.argmax(outputs, dim=1)
            acc = (preds == y_data).float().mean().item()
            print(f"Step {i}: Loss = {loss.item():.4f}, Accuracy = {acc:.4f}")



def train_model(model:nn.Module,
                train_loader,
                test_loader,
                epochs,
                learning_rate,
                device=None,
                start_from_saved=False,
                checkpoint_path:str=None,
                target_val_loss=0.005,
                buy_class_weights=1.6,
                sell_class_weights=1.6,
                holds_class_weights=1.0,
                val_loss_flag=True):
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
    if buy_class_weights is int:
        buy_class_weights = float(buy_class_weights)

    if sell_class_weights is int:
        sell_class_weights = float(sell_class_weights)
                                  
    if holds_class_weights is int:
        holds_class_weights = float(holds_class_weights)

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
    weights = t.Tensor([holds_class_weights, sell_class_weights, buy_class_weights]).to(device)

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
        if val_loss>=target_val_loss and val_loss_flag==True:
            model.train() 
            epoch_loss = 0.0
            start_time = time.time()
        
        
            for batch_idx, (x_data, y_data) in enumerate(train_loader):
                x_data = x_data.to(device)
                y_data = y_data.to(device)

                
                # Forward pass
                optimizer.zero_grad()
                outputs = model(x_data).to(device)
                
                # Calculate loss
                loss = criterion(outputs, y_data)
                
                # Backward pass and optimize
                loss.backward()
                
                # Gradient clipping
                t.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Adjust max_norm as needed
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
            
        elif val_loss<=target_val_loss and val_loss_flag==True:
            t.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss,
                }, f'checkpoints\\validation_loss_reached_epoch_{epoch+1}.pth')
            print(f"Validation loss target reached\nSaving model at epoch {epoch+1} and validation loss {val_loss:.6f}\n")
            return model, loss_list, val_loss_list, batch_loss_list
        
        else:
            model.train() 
            epoch_loss = 0.0
            start_time = time.time()
        
        
            for batch_idx, (x_data, y_data) in enumerate(train_loader):
                x_data = x_data.to(device)
                y_data = y_data.to(device)

                
                # Forward pass
                optimizer.zero_grad()
                outputs = model(x_data).to(device)
                
                # Calculate loss
                loss = criterion(outputs, y_data)
                
                # Backward pass and optimize
                loss.backward()
                
                # Gradient clipping
                t.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Adjust max_norm as needed
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

class LSTMWrapper(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)

    def forward(self, x):
        out, _ = self.lstm(x)
        return out  #Just return the output, drop hidden states

class LSTMClassificationModel(nn.Module):
    def __init__(self, num_x_data, num_classes, hidden_dim, num_hidden, dropout):
        super().__init__()
        
        hidden_layers = []
        
        hidden_layers.append(LSTMWrapper(num_x_data, hidden_dim, num_hidden))
        hidden_layers.append(nn.LayerNorm(hidden_dim*2))
        hidden_layers.append(nn.Dropout(dropout))
        hidden_layers.append(nn.ReLU())

        self.lstm_model= nn.Sequential(*hidden_layers)
        self.attention = nn.Linear(hidden_dim*2,1)
        self.fc = nn.Linear(hidden_dim*2,num_classes)
    
    def forward(self,x):
        lstm_output = self.lstm_model(x)
        
        #Attention
        scores = self.attention(lstm_output)
        attention_weights = t.softmax(scores,dim=1)
        weighted_output = attention_weights*lstm_output
        final = t.sum(weighted_output, dim=1)
        return self.fc(final)
        