import yfinance as yf
import os
from dotenv import load_dotenv
from pathlib import Path
from datetime import date as dt
from dateutil.relativedelta import relativedelta as rd


def getkeys():
    dotenv_path=Path('C:\\Users\\dimos\\Desktop\\MyDesktopFolders\\TradingWithML\\.env')
    load_dotenv(dotenv_path=dotenv_path)

    public=os.getenv('PUBLIC_ALPACA')
    secret=os.getenv('SECRET_ALPACA')
    return public,secret

def getdata(tickers,start,end,interval):

    data = yf.download(tickers, start=start, end=end, interval=interval)

    print(data)

    return data.to_numpy()

def datatrimming(data):
    closes = data[:,0]
    opens = data[:,3]
    return closes, opens

def allatonce(data,amount_monthly):
    closes, opens = datatrimming(data)
    total_amount_invested = amount_monthly*len(closes)

    stock_amount = total_amount_invested/opens[0]
    cash_pool = stock_amount * closes[len(closes)-1]

    total_pnl = cash_pool - total_amount_invested
    total_pnl_percentage = ((cash_pool - total_amount_invested)/total_amount_invested)*100

    return cash_pool, total_amount_invested, total_pnl, total_pnl_percentage

def DCA(data,amount_monthly):
    total_months = len(data)
    cash_pool = 0

    closes, opens = datatrimming(data)

    for i in range(total_months):
        money = cash_pool + amount_monthly
        price_fluctuation = ((opens[i] - closes[i]))/opens[i]
        cash_pool = money * (1-price_fluctuation)
        total_pnl = (cash_pool) - (amount_monthly * (i+1))
    
    total_contribution=total_months*amount_monthly

    total_pnl_percentage = ((cash_pool-total_contribution)/total_contribution)*100

    return cash_pool,total_contribution,total_pnl,total_pnl_percentage

def main(ticker,years_ago,days_ago,interval,amount_monthly,dca_flag,allatonce_flag):
    
    time_after = dt.today() - rd(days=days_ago)
    time_before = time_after - rd(years=years_ago)

    data = getdata(ticker,time_before,time_after,interval)
    
    if dca_flag==True:
        final_money_dca, total_contribution_dca, total_pnl_dca, total_pnl_percentage_dca = DCA(data,amount_monthly)
        ## DCA method
        print("DCA Method for ---> "+str(ticker)+"\n\n"+"My contribution: " + str(total_contribution_dca))
        print("Final money pool: " + str(round(final_money_dca,2)))
        if total_pnl_dca < 0:
            print("My losses: "+str(round(total_pnl_dca,2)))
            print("My losses percentage: "+str(round(total_pnl_percentage_dca,2)))
        else:
            print("My gains: "+str(round(total_pnl_dca,2)))
            print("My gains percentage: "+str(round(total_pnl_percentage_dca,2))+"\n\n")

    if allatonce_flag==True:
        final_money_allatonce, total_contribution_allatonce, total_pnl_allatonce, total_pnl_percentage_allatonce = allatonce(data,amount_monthly)

        ## All at once
        print("All at once Method\n\n"+"My contribution: " + str(total_contribution_allatonce))
        print("Final money pool: " + str(round(final_money_allatonce,2)))
        if total_pnl_dca < 0:
            print("My losses: "+str(round(total_pnl_allatonce,2)))
            print("My losses percentage: "+str(round(total_pnl_percentage_allatonce,2)))
        else:
            print("My gains: "+str(round(total_pnl_allatonce,2)))
            print("My gains percentage: "+str(round(total_pnl_percentage_allatonce,2)))



if __name__=="__main__":
    main(
        ticker="SOL-USD",
        years_ago=3,
        days_ago=0,
        interval="1mo",
        amount_monthly=100.0,
        dca_flag=True,
        allatonce_flag=False
        )