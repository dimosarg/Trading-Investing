import data_collection as data
import data_processing as proc
import data_to_torch as dtt
import plots as plt
import backtesting as bck
import models
import numpy as np
import torch as t

#Orismata
ticker_symbol = "btc-usd"
stoch_rsi_period = 14
mfi_period = 14
bollinger_bands_sma_period = 20
ema_period_d12 = 14
ema_period_d26 = 26
ema_period_d20 = 20
ema_period_d50 = 50
days_after_small = 5
days_after_big = 21
same_indicator_days = 5
overbought_rsi = 20 #%
oversold_rsi = 80 #%
##labeling
slope_days_before=21
slope_days_after=slope_days_before 
slope_threshold=0.0
min_change=0.005
peak_distance=15
peak_prominence=0.005
##
##support resistance
s_r_slope_days_before=14
s_r_slope_days_after=s_r_slope_days_before 
s_r_slope_threshold=0.0
s_r_min_change=0.01
s_r_peak_distance=s_r_slope_days_before
s_r_peak_prominence=0.005
##
data_file_name = f"{ticker_symbol}.npy"
read_data_flag = True
batch_size = 512
num_classes = 3
hidden_dim = 256
num_hidden = 3
dropout = 0.3
epochs = 1
learning_rate = 0.001
train_test_split = 0.8
cpu_flag = False
start_from_saved = True
checkpoint_path = "checkpoints\\checkpoint_epoch_250.pth"
target_val_loss_flag = False
target_val_loss = 0.005
holds_class_weights = 1
alpha_class_weighting = 0.86#from 0 to 1
look_back_period = 14
cutoff_period = 51
overfit_test_flag = False

amount = 50
leverage = 2
stop_loss_percentage = 20
##Telos Orismaton

device = models.getdevice()

if read_data_flag:
    print("Reading data...\n")
    prices = data.read_data(data_file_name)
else:
    print("Searching for data...\n")
    prices = data.getdata(ticker_symbol,period="max")

_,closes,_,_,_ = proc.dataprocessing(prices)


print("Calculating indicators...\n")
ema12 = proc.normalize(proc.ema(prices=prices, ema_period=ema_period_d12))
ema20 = proc.normalize(proc.ema(prices=prices, ema_period=ema_period_d20))
ema26 = proc.normalize(proc.ema(prices=prices, ema_period=ema_period_d26))
ema50 = proc.normalize(proc.ema(prices=prices, ema_period=ema_period_d50))
rsi = proc.normalize(proc.rsi(prices=prices, stoch_rsi_period=stoch_rsi_period))
stochastic = proc.normalize(proc.stoch(prices=prices,stoch_rsi_period=stoch_rsi_period))
mfi = proc.normalize(proc.mfi(prices=prices,mfi_period=mfi_period))
macd = proc.normalize(proc.macd(prices=prices))
mid_bollinger_band, lower_bollinger_band, higher_bollinger_band, standard_deviation= proc.normalize(proc.bollinger_bands(prices=prices,bollinger_bands_sma_period=bollinger_bands_sma_period))
eng_candle = proc.engulfing_candle(prices=prices, stoch_rsi_period=stoch_rsi_period, overbought_rsi=overbought_rsi, oversold_rsi=oversold_rsi)
supp_res = proc.support_resistance(prices=prices,
                                   s_r_min_change=s_r_min_change,
                                   s_r_n_after=s_r_slope_days_after,
                                   s_r_n_before=s_r_slope_days_before,
                                   s_r_peak_distance=s_r_peak_distance,
                                   s_r_peak_prominence=s_r_peak_prominence,
                                   s_r_slope_threshold=s_r_slope_threshold)
print("Done calculating indicators\n")

print("Normalizing prices...\n")
opens,closes,highs,lows,volume = proc.dataprocessing(prices)

unnormalized_closes = closes
opens = proc.normalize(opens)
closes = proc.normalize(closes)
highs = proc.normalize(highs)
lows = proc.normalize(lows)
volume = proc.normalize(volume)

indicators = np.stack([ema12,ema20,ema26,eng_candle,supp_res,rsi,stochastic,mfi,macd,standard_deviation,mid_bollinger_band,higher_bollinger_band,lower_bollinger_band],1) #This works [eng_candle,supp_res,rsi,stochastic,mfi,standard_deviation]]

num_indic = indicators.shape[1]
print(f"x_data has {num_indic} indicators\n")

x_data = proc.x_data_making(indicators, look_back_period=look_back_period, cutoff_period=cutoff_period)

print("Assigning labels...\n")
y_data, sell_count, buy_count = proc.slope_labeling(prices=prices,
                                n_before=slope_days_before, 
                                n_after=slope_days_after,
                                slope_threshold=slope_threshold,
                                min_change=min_change,
                                peak_distance=peak_distance,
                                peak_prominence=peak_prominence)

print("Adjusting class weights...\n")
buy_prominence = buy_count/len(y_data)
sell_prominence = sell_count/len(y_data)

buy_class_weights = 1/(buy_prominence**alpha_class_weighting)
sell_class_weights = 1/(sell_prominence**alpha_class_weighting)

print(f"Class weights are {holds_class_weights} for holds, {buy_class_weights} for buys and {sell_class_weights} for sells\n")

print("Creating dataset...\n")
dataset = dtt.LSTMClassificationDataset(x_data=x_data, y_data=y_data)
print(f"Splitting data into train and test with {train_test_split*100}% percentage...\n")
train_loader, test_loader = dtt.train_test_split(dataset,train_test_split=train_test_split,batch_size=batch_size)

print("Creating model...\n")
model = models.LSTMClassificationModel(num_x_data=num_indic,
                                       num_classes=num_classes,
                                       hidden_dim=hidden_dim,
                                       num_hidden=num_hidden,
                                       dropout=dropout
                                       )

if overfit_test_flag:
    print("Performing overfitting test...")
    models.overfit_test(model.to(device),train_loader,device)

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
    target_val_loss=target_val_loss,
    buy_class_weights=buy_class_weights,
    sell_class_weights=sell_class_weights,
    holds_class_weights=holds_class_weights,
    val_loss_flag=target_val_loss_flag
    )


test_array = x_data[int(round(len(x_data)*train_test_split,0)):len(x_data)]
test_array_closes = closes[int(round(len(x_data)*train_test_split,0)):len(x_data)]
test_array_unnormalized_closes = unnormalized_closes[int(round(len(x_data)*train_test_split,0)):len(x_data)]
test_known_results = y_data[int(round(len(x_data)*train_test_split,0)):len(x_data)]

test_predictions = trained_model(t.FloatTensor(test_array).to(device))

test_predictions = t.argmax(test_predictions,dim=1).cpu().detach().numpy()

plt.plot_scatter(test_array_closes, test_predictions, "Prediction of price movement",num_classes)
plt.plot_scatter(test_array_closes,test_known_results,"Buy-Sell labels",num_classes)
plt.plot_line_diagram(val_loss_list,"Validation Loss","Validation Loss")

final_amount, profit_perc, profit_money = bck.with_holds(ypred=test_predictions, test_closes=test_array_unnormalized_closes, leverage=leverage, amount=amount)
#final_amount_sl, profit_perc_sl, profit_money_sl = bck.with_holds_stop_loss(ypred=test_predictions, test_closes=test_array_unnormalized_closes, leverage=leverage, amount=amount, stop_loss_percentage=stop_loss_percentage)
#final_amount_rev, profit_perc_rev, profit_money_rev = bck.with_holds_reverse(ypred=test_predictions, test_closes=test_array_unnormalized_closes, leverage=leverage, amount=amount)

print(f"Strategy:\nFinal amount: {final_amount}\nProfit percentage: {profit_perc}\nProfit: {profit_money}\n")
#print(f"Stop loss strategy:\nFinal amount: {final_amount_sl}\nProfit percentage: {profit_perc_sl}\nProfit: {profit_money_sl}\n")
#print(f"Reverse strategy:\nFinal amount: {final_amount_rev}\nProfit percentage: {profit_perc_rev}\nProfit: {profit_money_rev}")

