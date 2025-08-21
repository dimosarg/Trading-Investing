import numpy as np
from sklearn import metrics

def strategy(ypred:np,closes:np,amount,leverage):
    """Δέχεται τις προβλέψεις του μοντέλου και μοντελοποιεί στα ιστορικά δεδομένα του test την
    στρατηγική του να αγοράζεις και να πουλάς όταν αλλάζει η πρόβλεψη:
    
    args:
    
    Ypred = Πίνακας προβλέψεων του μοντέλου ως τένσορα
    Closes = Όλες οι τιμές κλεισίματος που αντλούνται από την yfinance
    Amount = Ποσό το οποίο επενδύεται
    Leverage = Συντελεστής πολλαπλασιασμού του ποσού που επενδύεται"""

    originalamount=amount
    ypred=ypred[:len(ypred)]
    closes=closes[(len(closes)-len(ypred)):len(closes)]
    profit=0
    positionperc=amount/closes[0]

    for i in range(1,len(ypred)):
        try:
            if ypred[i]!=ypred[i-1]:
                positionperc=amount/closes[i]
            else:
                if ypred[i]==1:
                    profit=positionperc*leverage*((closes[i]-closes[i-1]))
                    amount=amount+profit
                else:
                    profit=positionperc*leverage*((closes[i-1]-closes[i]))
                    amount=amount+profit
        except IndexError:
            break

    profitperc=((amount-originalamount)/originalamount)*100
    profitmoney=amount-originalamount

    return profitperc,profitmoney

def truefalse(ypred:np,closes:np):
    closes=closes[(len(closes)-len(ypred)):len(closes)]
    truelabels=np.zeros(len(ypred))

    for i in range(1,len(ypred)):
        if closes[i]-closes[i-1]>=0:
            truelabels[i]=1
        else:
            truelabels[i]=0

    confusion_matrix=metrics.confusion_matrix(truelabels,ypred.cpu())

    corrects=confusion_matrix[0][0]+confusion_matrix[1][1]
    incorrects=confusion_matrix[0][1]+confusion_matrix[1][0]

    correctperc=(corrects/(corrects+incorrects))*100
    incorrectperc=(incorrects/(corrects+incorrects))*100

    print("Ποσοστό σωστών προβλέψεων: "+str(round(correctperc,2))+"%")
    print("Ποσοστό εσφαλμένων προβλέψεων: "+str(round(incorrectperc,2))+"%")

    cm=metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = ["Sell", "Buy"])

    return cm

def outliers(ypred:np,closes:np,amount,leverage,outliers):
    """Δέχεται τις προβλέψεις του μοντέλου και μοντελοποιεί στα ιστορικά δεδομένα του test την
    στρατηγική του να αγοράζεις και να πουλάς όταν αλλάζει η πρόβλεψη αγνωόντας τα outliers δηλαδή μεμονωμένες αλλαγές της πρόβλεψης:
    
    args:
    
    Ypred = Πίνακας προβλέψεων του μοντέλου ως τένσορα
    Closes = Όλες οι τιμές κλεισίματος που αντλούνται από την yfinance
    Amount = Ποσό το οποίο επενδύεται
    Leverage = Συντελεστής πολλαπλασιασμού του ποσού που επενδύεται"""

    originalamount=amount
    closes=closes[(len(closes)-len(ypred)):len(closes)]
    profit=0
    positionperc=amount/closes[0]

    for i in range(1,len(ypred)):
        try:
            if ypred[i]!=ypred[i-1] and sum(ypred[i:i+outliers])==outliers:
                positionperc=amount/closes[i]
            else:
                if ypred[i]==1:
                    profit=positionperc*leverage*((closes[i]-closes[i-1]))
                    amount=amount+profit
                else:
                    profit=positionperc*leverage*((closes[i-1]-closes[i]))
                    amount=amount+profit
        except IndexError:
            break

    profitperc=((amount-originalamount)/originalamount)*100
    profitmoney=amount-originalamount

    return profitperc,profitmoney

def reversestrategy(ypred:np,closes:np,amount,leverage):
    """Δέχεται τις προβλέψεις του μοντέλου και μοντελοποιεί στα ιστορικά δεδομένα του test την
    στρατηγική του να αγοράζεις και να πουλάς όταν αλλάζει η πρόβλεψη !!!ΠΡΟΣΟΧΗ!!! αυτή η στρατηγική εξυπηρετεί ΜΟΝΟ
    σκοπούς calibration στην πρώτη στρατηγική. Αυτή πρέπει κάθε φορά να βγαίνει αρνητική.
    
    args:
    
    Ypred = Πίνακας προβλέψεων του μοντέλου ως τένσορα
    Closes = Όλες οι τιμές κλεισίματος που αντλούνται από την yfinance
    Amount = Ποσό το οποίο επενδύεται
    Leverage = Συντελεστής πολλαπλασιασμού του ποσού που επενδύεται"""

    originalamount=amount
    closes=closes[(len(closes)-len(ypred)):len(closes)]
    profit=0
    positionperc=amount/closes[0]

    for i in range(1,len(ypred)):
        try:
            if ypred[i]!=ypred[i-1]:
                positionperc=amount/closes[i]
            else:
                if ypred[i]==1:
                    profit=positionperc*leverage*((closes[i-1]-closes[i]))
                    amount=amount+profit
                else:
                    profit=positionperc*leverage*((closes[i]-closes[i-1]))
                    amount=amount+profit
        except IndexError:
            break

    profitperc=((amount-originalamount)/originalamount)*100
    profitmoney=amount-originalamount

    return profitperc,profitmoney


def strategynumdays(ypred:np,closes:np,amount,leverage,numdays):
    """Δέχεται τις προβλέψεις του μοντέλου και μοντελοποιεί στα ιστορικά δεδομένα του test την
    στρατηγική του να αγοράζεις και να πουλάς όταν αλλάζει η πρόβλεψη:
    
    args:
    
    Ypred = Πίνακας προβλέψεων του μοντέλου ως τένσορα
    Closes = Όλες οι τιμές κλεισίματος που αντλούνται από την yfinance
    Amount = Ποσό το οποίο επενδύεται
    Leverage = Συντελεστής πολλαπλασιασμού του ποσού που επενδύεται"""

    originalamount=amount
    ypred=ypred[:len(ypred)-numdays]
    closes=closes[(len(closes)-len(ypred)):len(closes)]
    profit=0
    positionperc=amount/closes[0]

    for i in range(1,len(ypred)):
        try:
            if ypred[i]!=ypred[i-1]:
                positionperc=amount/closes[i]
            else:
                if ypred[i]==1:
                    profit=positionperc*leverage*((closes[i]-closes[i-1]))
                    amount=amount+profit
                else:
                    profit=positionperc*leverage*((closes[i-1]-closes[i]))
                    amount=amount+profit
        except IndexError:
            break

    profitperc=((amount-originalamount)/originalamount)*100
    profitmoney=amount-originalamount

    return profitperc,profitmoney

def truefalsenumdays(ypred:np,closes:np,numdays):
    closes=closes[(len(closes)-len(ypred)):len(closes)]
    truelabels=np.zeros(len(ypred)-numdays)

    for i in range(1,len(ypred)-numdays):
        if closes[i+numdays]-closes[i]>=0:
            truelabels[i]=1
        else:
            truelabels[i]=0

    confusion_matrix=metrics.confusion_matrix(truelabels,ypred[:len(ypred)-numdays].cpu())

    corrects=confusion_matrix[0][0]+confusion_matrix[1][1]
    incorrects=confusion_matrix[0][1]+confusion_matrix[1][0]

    correctperc=(corrects/(corrects+incorrects))*100
    incorrectperc=(incorrects/(corrects+incorrects))*100

    print("Ποσοστό σωστών προβλέψεων: "+str(round(correctperc,2))+"%")
    print("Ποσοστό εσφαλμένων προβλέψεων: "+str(round(incorrectperc,2))+"%")

    cm=metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = ["Sell", "Buy"])

    return cm

def waitstrategy(ypred:np,closes:np,amount,leverage,waitdays):
    """Δέχεται τις προβλέψεις του μοντέλου και μοντελοποιεί στα ιστορικά δεδομένα του test την
    στρατηγική του να αγοράζεις και να πουλάς όταν αλλάζει η πρόβλεψη:
    
    args:
    
    Ypred = Πίνακας προβλέψεων του μοντέλου ως τένσορα
    Closes = Όλες οι τιμές κλεισίματος που αντλούνται από την yfinance
    Amount = Ποσό το οποίο επενδύεται
    Leverage = Συντελεστής πολλαπλασιασμού του ποσού που επενδύεται"""

    originalamount=amount
    ypred=ypred[:len(ypred)]
    closes=closes[(len(closes)-len(ypred)):len(closes)]
    profit=0
    positionperc=amount/closes[0]

    for i in range(waitdays,len(ypred)):
        try:
            if ypred[i]!=ypred[i-waitdays]:
                positionperc=amount/closes[i]
            else:
                if ypred[i]==1:
                    profit=positionperc*leverage*((closes[i]-closes[i-1]))
                    amount=amount+profit
                else:
                    profit=positionperc*leverage*((closes[i-1]-closes[i]))
                    amount=amount+profit
        except IndexError:
            break

    profitperc=((amount-originalamount)/originalamount)*100
    profitmoney=amount-originalamount

    return profitperc,profitmoney

# def with_holds(ypred:np, test_closes:np, amount, leverage):
#     original_amount = amount
#     current_position = 0
#     position_perc=[0]
#     profit = 0

#     for i in range(1,len(ypred)):
#         if ypred[i] != ypred[i-1] and ypred[i] != 0:
#             position_perc.append(amount/test_closes[i])
#             position_perc.append(test_closes[i])
#             if current_position == 1:
#                 profit =  position_perc[-2]*leverage*(position_perc[-1]-test_closes[i])
#             elif current_position == 2:
#                 profit =  position_perc[-2]*leverage*(test_closes[i]-position_perc[-1])
                

#             amount = amount+profit

#             if ypred[i] == 1:
#                 current_position = 1
#             elif ypred[i] == 2:
#                 current_position = 2
    
#     final_amount = amount
#     profitperc=((amount-original_amount)/original_amount)*100
#     profitmoney=amount-original_amount

#     return final_amount, profitperc, profitmoney
            
            
        
def with_holds(ypred, test_closes, amount, leverage):
    """Whenever the prediction changes, the algorythm closes the previous position and opens another one with
    oposite direction\n
    
    args:\n

    ypred = Predictions (Numpy array)\n
    test_closes = Close prices of the test data (Numpy array)\n
    amount = Amount invested (float)\n
    leverage = Leverage (int)
    """
    original_amount = amount
    current_position = [0,0]
    position_perc=[0,0]
    profit = 0

    for i in range(1,len(ypred)):
        if ypred[i]!=0:
            current_position.append(int(ypred[i]))

        if ypred[i] != ypred[i-1] and ypred[i] != 0 and current_position[-1] != current_position[-2]:
            
            #calculate profits of previous position
            if current_position[-2] == 1:
                profit =  position_perc[-2]*leverage*(position_perc[-1]-test_closes[i])
            elif current_position[-2] == 2:
                profit =  position_perc[-2]*leverage*(test_closes[i]-position_perc[-1])
            
            #close previous position
            amount = amount+profit

            #start new position
            position_perc.append(amount/test_closes[i])
            position_perc.append(test_closes[i])

    final_amount = amount
    profitperc=((amount-original_amount)/original_amount)*100
    profitmoney=amount-original_amount

    return final_amount, profitperc, profitmoney
            
            
        
def with_holds_reverse(ypred, test_closes, amount, leverage):
    """Basically whenever the prediction is to buy I sell and 
    whenever it says to sell I buy\n
    
    args:\n

    ypred = Predictions (Numpy array)\n
    test_closes = Close prices of the test data (Numpy array)\n
    amount = Amount invested (float)\n
    leverage = Leverage (int)
    """
    original_amount = amount
    current_position = [0,0]
    position_perc=[0,0]
    profit = 0

    for i in range(1,len(ypred)):
        current_position.append(int(ypred[i]))
        if ypred[i] != ypred[i-1] and ypred[i] != 0 and current_position[-1] != current_position[-2]:
            
            #calculate profits of previous position
            if current_position[-2] == 1:
                profit =  position_perc[-2]*leverage*(test_closes[i]-position_perc[-1])
            elif current_position[-2] == 2:
                profit =  position_perc[-2]*leverage*(position_perc[-1]-test_closes[i])
            
            #close previous position
            amount = amount+profit

            #start new position
            position_perc.append(amount/test_closes[i])
            position_perc.append(test_closes[i])

    final_amount = amount
    profitperc=((amount-original_amount)/original_amount)*100
    profitmoney=amount-original_amount

    return final_amount, profitperc, profitmoney
            
            
        
