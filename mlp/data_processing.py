import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.signal import find_peaks

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
    standard_deviation = np.zeros(len(prices))

    for i in range(bollinger_bands_sma_period,len(prices)):
        middle_band[i] = np.sum(closes[i-bollinger_bands_sma_period:i])/bollinger_bands_sma_period
        standard_deviation[i] = np.std(closes[i-bollinger_bands_sma_period:i])
        lower_band[i] = middle_band[i]-(standard_deviation[i]*2)
        higher_band[i] = middle_band[i]+(standard_deviation[i]*2)
    
    return middle_band,lower_band,higher_band,standard_deviation



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



def support_resistance(prices,
                    s_r_n_before=5, 
                    s_r_n_after=5, 
                    s_r_slope_threshold=0.0, 
                    s_r_min_change=0.005,
                    s_r_peak_distance=5, 
                    s_r_peak_prominence=0.005):

    _,closes,_,_,_=dataprocessing(prices)
    indicator = np.zeros(len(closes))

    # -------- Slope-based turning points --------
    slope_turns = np.zeros(len(closes))
    for i in range(s_r_n_before, len(closes) - s_r_n_after):
        # slope before
        x_before = np.arange(s_r_n_before).reshape(-1, 1)
        y_before = closes[i-s_r_n_before:i]
        slope_before = LinearRegression().fit(x_before, y_before).coef_[0]

        # slope after
        x_after = np.arange(s_r_n_after).reshape(-1, 1)
        y_after = closes[i+1:i+1+s_r_n_after]
        slope_after = LinearRegression().fit(x_after, y_after).coef_[0]

        # detect reversal
        if slope_before < -s_r_slope_threshold and slope_after > s_r_slope_threshold:
            slope_turns[i] = 1  # Buy
        elif slope_before > s_r_slope_threshold and slope_after < -s_r_slope_threshold:
            slope_turns[i] = 0.5  # Sell

    # -------- Peak/trough confirmation --------
    peaks, _ = find_peaks(closes, distance=s_r_peak_distance, prominence=s_r_peak_prominence)
    troughs, _ = find_peaks(-closes, distance=s_r_peak_distance, prominence=s_r_peak_prominence)

    peak_points = set(peaks)
    trough_points = set(troughs)

    # -------- Combine slope and peak info --------
    for i in range(len(closes)):
        if slope_turns[i] == 1 and i in trough_points:
            # confirm BUY and filter by min change
            if (closes[i+s_r_n_after] - closes[i]) / closes[i] > s_r_min_change:
                indicator[i] = 1
        elif slope_turns[i] == 0.5 and i in peak_points:
            # confirm SELL and filter by min change
            if (closes[i] - closes[i+s_r_n_after]) / closes[i] > s_r_min_change:
                indicator[i] = 0.5

    return indicator



def engulfing_candle(prices,stoch_rsi_period,overbought_rsi,oversold_rsi):
    opens,closes,highs,lows,_=dataprocessing(prices)
    rsi_vals=rsi(prices,stoch_rsi_period)
    overbought_rsi=overbought_rsi/100
    oversold_rsi=oversold_rsi/100
    eng_candle = np.zeros(len(closes))
    
    #bullish engulfing candle
    for i in range(1,len(closes)):
        day_before = closes[i-1]-opens[i-1]
        today = closes[i]-opens[i]
        #if rsi_vals[i]<oversold_rsi:
        if day_before<0 and today>0:
            if closes[i-1]>opens[i] and closes[i]>opens[i-1]:
                eng_candle[i] = 1

    #bearish engulfing candle
        #if rsi_vals[i]>overbought_rsi:
        if day_before>0 and today<0:
            if closes[i-1]<opens[i] and closes[i]<opens[i-1]:
                eng_candle[i] = 0.5

    return eng_candle
 


def labeling(prices,
             days_after_small:int,
             days_after_big:int,
             sensitivity_small,
             sensitivity_big,
             cutoff_period,
             same_indicator_days):
    """Βάζει τις ταμπέλες Buy και Sell ως όρισμα στα στοιχεία που δίνονται για training\n
        args:\n
        Prices = Dataframe\n
        days_after_small = Μικρό περιθώριο ημερών για αλλαγή τιμής\n
        days_after_big = Μεγάλο περιθώριο ημερών για αλλαγή τιμής\n
        sensitivity_small = Μικρή ευαισθησία αλλαγής τιμης που βασίζεται στο μικρό περιθώριο ημερών\n
        sensitivity_big = Μεγάλη ευαισθησία αλλαγής τιμης που βασίζεται στο μεγάλο περιθώριο ημερών\n
        cutoff_period = Number of days to ignore in the beginning of the indicator array (because they are 0s)\n
        same_indicator_days = Number of days where if the indicator is same as the previous day's, it becomes hold
        """
    _,closes,_,_,_=dataprocessing(prices)

    indicator=np.zeros(len(prices))

    for i in range(len(prices)-days_after_big):
        big_buy_closes = closes[i]*(1+sensitivity_big)
        small_buy_closes = closes[i]*(1+sensitivity_small)
        big_sell_closes = closes[i]*(1-sensitivity_big)
        small_sell_closes = closes[i]*(1-sensitivity_small)
        closes[i]=closes[i]

        #buy=2
        if closes[i+days_after_small]>small_buy_closes or closes[i+days_after_big]>big_buy_closes:
            indicator[i]=2
        #sell=1
        elif closes[i+days_after_small]<small_sell_closes or closes[i+days_after_big]<big_sell_closes:
            indicator[i]=1

    # Avoiding same actions for multiple days at a time
    for i in range(same_indicator_days,len(prices)):
        if np.any(indicator[i-same_indicator_days:i] == indicator[i]) and indicator[i]!= 0:
                indicator[i-same_indicator_days:i] = 0

    sells=np.zeros(len(indicator)-cutoff_period)
    buys=np.zeros(len(indicator)-cutoff_period)
    holds=np.zeros(len(indicator)-cutoff_period)

    for i in range(cutoff_period, len(sells)-1):
        if indicator[i]==1:
            sells[i]= closes[i]
        elif indicator[i]==2:
            buys[i]= closes[i]
        elif indicator[i]==0:
            holds[i]= closes[i]
    
    return indicator



def slope_labeling(prices,
                    n_before=5, 
                    n_after=5, 
                    slope_threshold=0.0, 
                    min_change=0.005,
                    peak_distance=5, 
                    peak_prominence=0.005):
    """
    Hybrid slope + peak/trough detection labeling.

    Labels:
    0 = Hold
    1 = Sell
    2 = Buy
    """

    _,closes,_,_,_=dataprocessing(prices)
    indicator = np.zeros(len(closes))

    # -------- Slope-based turning points --------
    slope_turns = np.zeros(len(closes))
    for i in range(n_before, len(closes) - n_after):
        # slope before
        x_before = np.arange(n_before).reshape(-1, 1)
        y_before = closes[i-n_before:i]
        slope_before = LinearRegression().fit(x_before, y_before).coef_[0]

        # slope after
        x_after = np.arange(n_after).reshape(-1, 1)
        y_after = closes[i+1:i+1+n_after]
        slope_after = LinearRegression().fit(x_after, y_after).coef_[0]

        # detect reversal
        if slope_before < -slope_threshold and slope_after > slope_threshold:
            slope_turns[i] = 2  # Buy
        elif slope_before > slope_threshold and slope_after < -slope_threshold:
            slope_turns[i] = 1  # Sell

    # -------- Peak/trough confirmation --------
    peaks, _ = find_peaks(closes, distance=peak_distance, prominence=peak_prominence)
    troughs, _ = find_peaks(-closes, distance=peak_distance, prominence=peak_prominence)

    peak_points = set(peaks)
    trough_points = set(troughs)

    # -------- Combine slope and peak info --------
    buy_count = 0
    sell_count = 0
    for i in range(len(closes)):
        if slope_turns[i] == 2 and i in trough_points:
            # confirm BUY and filter by min change
            if (closes[i+n_after] - closes[i]) / closes[i] > min_change:
                indicator[i] = 2
                buy_count = buy_count+1
        elif slope_turns[i] == 1 and i in peak_points:
            # confirm SELL and filter by min change
            if (closes[i] - closes[i+n_after]) / closes[i] > min_change:
                indicator[i] = 1
                sell_count = sell_count+1

    return indicator, sell_count, buy_count



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



def normalize(data):
    """Normalizes data within a range of 0 - 1\n
    args:\n
    data = Array object with indicators
    """
    if np.any(element < 0 for element in data):
        datanew = data + np.min(data)
    else:
        datanew = data
    
    dist = np.max(data) - np.min(data)
    
    datafinal = datanew/dist

    return datafinal



def x_data_making(data:np, look_back_period, cutoff_period):
    """Prepares the indicators given for training and outputs a 3d array
    with stacked 2d arrays of the indicators for look_back_period number of days\n
    args:\n
    data = Indicators\n
    look_back_period = Number of days to look back\n
    cutoff_period = Number of days to ignore in the beginning of the indicator array (because they are 0s)
    """

    x_data = []

    for i in range(cutoff_period,len(data)):
        x_data.append(data[i-look_back_period:i,:])

    x_data_np = np.array(x_data)

    return x_data_np