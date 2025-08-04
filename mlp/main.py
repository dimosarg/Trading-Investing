import data_collection as data
import data_processing as proc
import data_to_torch as dtt
import models
import numpy as np
import torch as t
from torch.utils.data import DataLoader

#Orismata
stoch_rsi_period = 14
mfi_period = 14
bollinger_bands_sma_period = 20
ema_period_d12 = 14
ema_period_d26 = 26
ema_period_d20 = 20
ema_period_d50 = 50
days_after_small = 3
days_after_big = 7
sensitivity_small = 2
sensitivity_big = 3
sensitivity_buy = 3
sensitivity_sell = 2
data_file_name = "btc-usd.npy"
read_data_flag = True
batch_size = 32
num_classes = 3
hidden_dim = 32
num_hidden = 2
dropout = 0.2 
epochs = 10
learning_rate = 0.0001
train_test_split = 0.8
cpu_flag = False
start_from_saved = False
checkpoint_path = None
target_val_loss = 0.005
class_weights = 5
##Telos Orismaton

device = models.getdevice()

if read_data_flag:
    print("Reading data...\n")
    prices = data.read_data(data_file_name)
else:
    print("Searching for data...\n")
    prices = data.getdata("btc-usd",period="max")

print("Calculating indicators...\n")
ema12 = proc.ema(prices=prices, ema_period=ema_period_d12)
ema20 = proc.ema(prices=prices, ema_period=ema_period_d20)
ema26 = proc.ema(prices=prices, ema_period=ema_period_d26)
ema50 = proc.ema(prices=prices, ema_period=ema_period_d50)
rsi = proc.rsi(prices=prices, stoch_rsi_period=stoch_rsi_period)
stochastic = proc.stoch(prices=prices,stoch_rsi_period=stoch_rsi_period)
mfi = proc.mfi(prices=prices,mfi_period=mfi_period)
mid_bollinger_band, lower_bollinger_band, higher_bollinger_band = proc.bollinger_bands(prices=prices,bollinger_bands_sma_period=bollinger_bands_sma_period)
print("Done calculating indicators\n")

x_data = np.stack([ema12,ema20,ema26,ema50,rsi,stochastic,mfi,mid_bollinger_band,lower_bollinger_band,higher_bollinger_band],1)

num_x_data = x_data.shape[1]
print(f"x_data has {num_x_data} indicators\n")

print("Assigning labels...\n")
y_data = proc.labeling(prices=prices,
                       days_after_small=days_after_small,
                       days_after_big=days_after_big,
                       sensitivity_small=sensitivity_small,
                       sensitivity_big=sensitivity_big)

print("Creating dataset...\n")
dataset = dtt.LSTMClassificationDataset(x_data=x_data, y_data=y_data)
print(f"Splitting data into train and test with {train_test_split*100}% percentage...\n")
train_loader, test_loader = dtt.train_test_split(dataset,train_test_split=train_test_split,batch_size=batch_size)

print("Creating model...\n")
model = models.LSTMClassificationModel(num_x_data=num_x_data,
                                       num_classes=num_classes,
                                       hidden_dim=hidden_dim,
                                       num_hidden=num_hidden,
                                       dropout=dropout
                                       )

print(f"Training model with {device} as device\n")
trained_model, loss_list, val_loss_list, batch_loss_list = models.train_model(
    model=model,
    train_loader=train_loader,
    test_loader=test_loader,
    device=device,
    epochs=epochs,
    learning_rate=learning_rate,
    start_from_saved=start_from_saved,
    checkpoint_path=checkpoint_path,
    target_val_loss=target_val_loss
    )


print(prices)