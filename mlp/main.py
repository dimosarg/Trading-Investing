import data_collection as data
import data_processing as proc
import numpy as np

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
##Telos Orismaton

if read_data_flag:
    prices = data.read_data(data_file_name)
else:
    prices = data.getdata("btc-usd",period="max")

ema12 = proc.ema(prices=prices, ema_period=ema_period_d12)
ema20 = proc.ema(prices=prices, ema_period=ema_period_d20)
ema26 = proc.ema(prices=prices, ema_period=ema_period_d26)
ema50 = proc.ema(prices=prices, ema_period=ema_period_d50)
rsi = proc.rsi(prices=prices, stoch_rsi_period=stoch_rsi_period)
stochastic = proc.stoch(prices=prices,stoch_rsi_period=stoch_rsi_period)
mfi = proc.mfi(prices=prices,mfi_period=mfi_period)
mid_bollinger_band, lower_bollinger_band, higher_bollinger_band = proc.bollinger_bands(prices=prices,bollinger_bands_sma_period=bollinger_bands_sma_period)

x_data = np.stack([ema12,ema20,ema26,ema50,rsi,stochastic,mfi,mid_bollinger_band,lower_bollinger_band,higher_bollinger_band],1)




print(prices)