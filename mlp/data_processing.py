import numpy as np
import pandas as pd
import torch as t

def dataprocessing(prices):
    """Διαχωρισμος του αρχικου dataframe στα βασικά επιμέρους στοιχεία του 
    που είναι οι στήλες open,close,high,low\n
    args:\n
    prices = what you get from yfinance
    """

    try:
        opens=prices.iloc[:,4].to_numpy()
        closes = prices.iloc[:,1].to_numpy()
        lows=prices.iloc[:,3].to_numpy()
        highs=prices.iloc[:,2].to_numpy()
        volume=prices.iloc[:,5].to_numpy()
    except ValueError:
        print("Check dataframe column naming")
    return opens,closes,highs,lows,volume



def stoch(prices,stoch_rsi_period:int):
    """Υπολογισμος του δείκτη Stochastic για τις τιμές\n
    args:\n
    prices = what you get from yfinance\n
    stoch_rsi_period = period for stochastic and rsi"""

    _,closes,highs,lows,_=dataprocessing(prices)

    stoch=np.zeros(len(prices))
    lowest=np.zeros(stoch_rsi_period)
    highest=np.zeros(stoch_rsi_period)

    for i in range(stoch_rsi_period,len(stoch)):
        for j in range(len(lowest)):
            lowest[j]=lows[i-stoch_rsi_period+j]
            highest[j]=highs[i-stoch_rsi_period+j]

        l14=np.min(lowest)
        h14=np.max(highest)

        stoch[i]=((closes[i-1]-l14)/(h14-l14))*100
    return stoch



def rsi(prices,stoch_rsi_period:int):
    """Υπολογισμος του δείκτη RSI για τις τιμές\n
    args:\n
    prices = what you get from yfinance\n
    stoch_rsi_period = period for stochastic and rsi
    """
    opens,closes,_,_,_=dataprocessing(prices)

    

    
    dist=opens-closes

    percdist=(dist/opens)*100

    rsi=np.zeros(len(opens))
    rsimidmat=np.zeros(stoch_rsi_period)

    for i in range(stoch_rsi_period-1,len(rsi)):
        pos=0
        neg=0
        counterpos=0
        counterneg=0
        for j in range(len(rsimidmat)):
            rsimidmat[j]=percdist[i-stoch_rsi_period+j]
            if rsimidmat[j]>0:
                pos=pos+rsimidmat[j]
                counterpos=counterpos+1
            elif rsimidmat[j]<=0:
                neg=neg+np.abs(rsimidmat[j])
                counterneg=counterneg+1
        
        avggain=pos/stoch_rsi_period
        avgloss=neg/stoch_rsi_period

        rs=avggain/avgloss

        rsi[i]=100/(1+rs)
    return rsi



def mfi(prices,mfi_period):
    """Money Flow Index\n
    args:\n
    prices = what you get from yfinance\n
    mfi_period = period for the mfi
    """
    _,closes,highs,lows, volume = dataprocessing(prices)

    typical_price = np.zeros(len(prices))
    raw_money_flow = np.zeros(len(prices))
    money_flow_ratio = np.zeros(len(prices))
    money_flow = np.zeros(len(prices))
    
    posmf=[]
    negmf=[]

    typical_price = (highs+lows+closes)/3
    raw_money_flow = typical_price*volume
        
    for i in range(mfi_period,len(prices)):
        posmf.clear()
        negmf.clear()
        for j in range(mfi_period):
            if raw_money_flow[i-mfi_period+j]-raw_money_flow[i-mfi_period+j-1]>=0:
                posmf.append(raw_money_flow[i+j-mfi_period])
            else:
                negmf.append(raw_money_flow[i+j-mfi_period])
        
        money_flow_ratio[i] = sum(posmf)/sum(negmf)
        
        money_flow[i] = 100-(100/(1+money_flow_ratio[i]))
        
    return money_flow



def bollinger_bands(prices,bollinger_bands_sma_period):
    """Bollinger Bands\n
    args:\n
    prices = what you get from yfinance\n
    bollinger_bands_sma_period = period for the middle band (SMA)
    """
    _,closes,_,_,_ = dataprocessing(prices)

    middle_band = np.zeros(len(prices))
    lower_band = np.zeros(len(prices))
    higher_band = np.zeros(len(prices))

    for i in range(bollinger_bands_sma_period,len(prices)):
        middle_band[i] = np.sum(closes[i-bollinger_bands_sma_period:i])/bollinger_bands_sma_period
        standard_deviation = np.std(closes[i-bollinger_bands_sma_period:i])
        lower_band[i] = middle_band[i]-(standard_deviation*2)
        higher_band[i] = middle_band[i]+(standard_deviation*2)
    
    return middle_band,lower_band,higher_band



def ema(prices,ema_period):
    """Exponential Moving Average of price
    args:\n
    prices = what you get from yfinance\n
    ema_period = period of the ema
    """
    _,closes,_,_,_ = dataprocessing(prices)

    smoothing_multiplier = 2/(ema_period+1)
    expo_moving_average = np.zeros(len(prices))

    expo_moving_average[ema_period] = np.sum(closes[0:ema_period-1])/ema_period

    for i in range(ema_period+1,len(prices)):
        expo_moving_average[i] = closes[i]*smoothing_multiplier+expo_moving_average[i-1]*(1-smoothing_multiplier)
        
    return expo_moving_average



def macd(prices):
    """MACD indicator (12day ema - 26day ema)\n
    args:\n
    prices = what you get from yfinance
    """

    macd_array = ema(prices,12) - ema(prices,26)

    return macd_array



def labeling(prices,days_after_small:int,days_after_big:int,sensitivity_small:int,sensitivity_big:int):

    opens,closes,_,_,_=dataprocessing(prices)

    bullish=np.zeros(len(prices))
    bearish=np.zeros(len(prices))
    neutral=np.zeros(len(prices))


    for i in range(len(bullish)):
        if opens[i]-closes[i]>0:
            bullish[i]=1
        elif opens[i]-closes[i]<0:
            bearish[i]=1
        elif opens[i]-closes[i]==0:
            neutral[i]=1

    indicator=np.zeros(len(bullish))

    for i in range(len(bullish)-days_after_big):
        #buy=2
        if closes[i+days_after_small]>=closes[i]+closes[i]*(sensitivity_small+0.003) or closes[i+days_after_big]>=closes[i]+closes[i]*(sensitivity_big+0.003):
            indicator[i]=2
        #sell=1
        elif closes[i+days_after_small]<=closes[i]-closes[i]*(sensitivity_small+0.002) or closes[i+days_after_big]<=closes[i]-closes[i]*(sensitivity_big+0.002):
            indicator[i]=1

    #Avoiding same actions for multiple days at a time
    
    # for i in range(len(bullish) - 7): 
    #     if np.any(indicator[i + 1:i + 2] == indicator[i]) and indicator[i]!= 0:
    #             indicator[i + 1:i + 2] = 0

    sells=np.zeros(len(indicator))
    buys=np.zeros(len(indicator))
    holds=np.zeros(len(indicator))

    for i in range(len(indicator)):
        if indicator[i]==1:
            sells[i]= closes[i]
        elif indicator[i]==2:
            buys[i]= closes[i]
        elif indicator[i]==0:
            holds[i]= closes[i]
    
    return indicator



def labelingbuysell(prices,days_after_small:int,days_after_big:int,sensitivity_small:int,sensitivity_big:int,sensitivity_buy:int,sensitivity_sell:int):
    """Βάζει τις ταμπέλες Buy και Sell ως όρισμα στα στοιχεία που δίνονται για training:
    args:
    Prices: Dataframe
    days_after_small: Μικρό περιθώριο ημερών για αλλαγή τιμής
    days_after_big: Μεγάλο περιθώριο ημερών για αλλαγή τιμής
    sensitivity_small: Μικρή ευαισθησία αλλαγής τιμης που βασίζεται στο μικρό περιθώριο ημερών
    sensitivity_big: Μεγάλη ευαισθησία αλλαγής τιμης που βασίζεται στο μεγάλο περιθώριο ημερών
    sensitivity_buy: Ευαισθησία αλλαγής τιμης που βασίζεται μόνο στις τιμές αγοράς
    sensitivity_sell: Ευαισθησία αλλαγής τιμης που βασίζεται μόνο στις τιμές πωλήσεων
    """

    opens,closes,_,_,_=dataprocessing(prices)

    bullish=np.zeros(len(prices))
    bearish=np.zeros(len(prices))

    for i in range(len(bullish)):
        if opens[i]-closes[i]>=0:
            bullish[i]=1
        elif opens[i]-closes[i]<0:
            bearish[i]=1

    indicator=np.zeros(len(bullish))

    for i in range(len(bullish)-days_after_big):
        #buy=2
        if closes[i+days_after_small]>=closes[i]+closes[i]*(sensitivity_small+sensitivity_buy) or closes[i+days_after_big]>=closes[i]+closes[i]*(sensitivity_big+sensitivity_buy):
            indicator[i]=1
        #sell=1
        elif closes[i+days_after_small]<=closes[i]-closes[i]*(sensitivity_small+sensitivity_sell) or closes[i+days_after_big]<=closes[i]-closes[i]*(sensitivity_big+sensitivity_sell):
            indicator[i]=0

    sells=np.zeros(len(indicator))
    buys=np.zeros(len(indicator))

    for i in range(len(indicator)):
        if indicator[i]==0:
            sells[i]= closes[i]
        elif indicator[i]==1:
            buys[i]= closes[i]
    
    return indicator