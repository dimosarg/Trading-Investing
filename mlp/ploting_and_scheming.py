import matplotlib.pyplot as plt
from numpy import zeros
from  matplotlib.colors import LinearSegmentedColormap

def listToPlot(y:any)->dict:
    """
    Δοθείσας μια προσπέλασιμης δομή η οποία έχει το αποτέλεσμα/στοχο του μοντέλου για κάθε χρονική στιγμή 
    δημιουργούμε ένα λεξικό όπου έχει ως κλειδι τον αριθμό των κλάσεων και ώς τιμή μια λίστα με τις χρονικές στιγμές
    της αντιστοιχής κλάσης
    args:
        y μια προσπελάσιμη δομή
    return:
        dict
    """
    dict={}
    for index,value in enumerate(y):
        if value not in dict:
            dict[value]=[index]
        else:
            dict[value].append(index)
    return dict


def plotPriceAndModelAction(ytest,prices,labels):
    """
    Η συναρτηση αυτη χρησιμοποιείται για την απεικόνηση της πραγματικής τιμής της μετοχής 
    και την χρήση διαφορετικών χρωμάτων για την απεικόνηση της
    ώς ενδείξη της προβλέψης του μοντέλου για την αντίστοιχη μέρα.

    args:
        ytest η πρόβλεψη του μοντέλου σε ordinal μορφή (δηλαδή μονοδιάστατο πίνακα)
        prices οι πραγματικές τιμές της μετοχές σε παραλλήρισμο με τις μέρες που το μοντέλο κάνει της προβλέψεις
    """
   
    _, axes = plt.subplots(figsize=(15, 6))
    
    y = ytest.cpu().detach().numpy()
    print(prices.shape)
    dict=listToPlot(y)
    keys=dict.keys()
    color_map = LinearSegmentedColormap.from_list('gr',["g", "r"], N=len(keys))
    try:
        for i,label in enumerate(keys):
            if dict.get(label):
                axes.scatter(dict[label], prices[dict[label]], color=color_map(i), label=labels[label],s=5)  
        axes.set_xlabel('Index')
        axes.set_ylabel('Prices')
        axes.legend()
    
    except Exception as e:
        print(f"An error occurred: {e}")
    plt.title('MY DARK AND TWISTED FANTASY')
    plt.xlabel('Time')
    plt.ylabel('Value of index')

    #plt.show()

def plotBuySellAction(ytest,prices):
    """
    Η συναρτηση αυτη χρησιμοποιείται για την απεικόνηση της πραγματικής τιμής της μετοχής 
    και την χρήση διαφορετικών χρωμάτων για την απεικόνηση της
    ώς ενδείξη της προβλέψης του μοντέλου για την αντίστοιχη μέρα.

    args:
        ytest η πρόβλεψη του μοντέλου σε ordinal μορφή (δηλαδή μονοδιάστατο πίνακα)
        prices οι πραγματικές τιμές της μετοχές σε παραλλήρισμο με τις μέρες που το μοντέλο κάνει της προβλέψεις
    """
   
    _, axes = plt.subplots(figsize=(15, 6))
    
    y = ytest.cpu().detach().numpy()
    print(prices.shape)
    dict=listToPlot(y)
    keys=dict.keys()
    color_map = LinearSegmentedColormap.from_list('gr',["g","r"], N=len(keys))
    # labels=dict.get(label)
    # for i in range(len(keys)):
    #     if dict.get(label)=='0':
    #         labels=='sell'
    #     elif dict.get(label)=='1':
    #         labels=='buy'

    try:
        for i,label in enumerate(keys):
            if dict.get(label):
                axes.scatter(dict[label], prices[dict[label]], color=color_map(i), label=label,marker='.')  
        axes.set_xlabel('Index')
        axes.set_ylabel('Prices')
        axes.legend()
    
    except Exception as e:
        print(f"An error occurred: {e}")
    plt.title('MY DARK AND TWISTED FANTASY')
    plt.xlabel('Time')
    plt.ylabel('Value of index')

    plt.show()

def plotlabels(sells,buys,holds):
    """Πλοτάρει τα σημεία που θα πρέπει να πουλάει και αγοράζει το πρόγραμμα
    args:
        Dataframe for Sells
        Dataframe for Buys
        Dataframe for Holds
    """
    xvals=zeros(len(sells))
    
    for i in range(1,len(xvals)):
        xvals[i]=xvals[i-1]+1

    fig1=plt.figure(1)
    plt.scatter(xvals,sells,label='Sell',color='r')
    plt.scatter(xvals,buys,label='Buy',color='g')
    plt.scatter(xvals,holds,label='Hold',color='b',marker='.')
    plt.legend()
    plt.title("Where to bet")
    plt.show()

def plotlabelsbuysell(sells,buys):
    """Πλοτάρει τα σημεία που θα πρέπει να πουλάει και αγοράζει το πρόγραμμα
    args:
        Dataframe for Sells
        Dataframe for Buys
    """
    xvals=zeros(len(sells))
    
    for i in range(1,len(xvals)):
        xvals[i]=xvals[i-1]+1
    fig1=plt.figure(1)
    plt.scatter(xvals,sells,label='Sell',color='r',marker='.')
    plt.scatter(xvals,buys,label='Buy',color='g',marker='.')
    plt.legend()
    plt.title("Where to bet")
    #plt.show()

# def plotActualAndPredictedPrices(ytrue,ypred):
#     """
#     Δίνουμε τους στόχους και τις προβλέψεις και η συνάρτηση τις πλοτάρει για κάθε μέρα.
#     """
#     _, axes = plt.subplots(figsize=(15, 6))
#     axes.plot(range(len(ytrue)), ytrue, color = 'red', label = 'Real Stock Price')
#     axes.plot(range(len(ytrue)),ypred, color = 'blue', label = 'Predicted Stock Price')
#     plt.title('MY DARK AND TWISTED FANTASY')
#     plt.xlabel('Time')
#     plt.ylabel('Value of index')
#     plt.legend()
#     plt.show()

def confusionmatrix(confusion_matrix):
    
    confusion_matrix.plot()
def plotTraingModel(training_data:list,testing_data:list=[],training_data_label:str='TrainData',testing_data_label:str='TestData',xlabel='epoch',ylabe='accuracy'):
    _, axes = plt.subplots(figsize=(15, 6))
    axes.plot(range(len(training_data)), training_data, color = 'red', label = training_data_label)
    axes.plot(range(len(testing_data)),testing_data, color = 'blue', label = testing_data_label)
    plt.title('MY DARK AND TESTED FANTASY')
    plt.xlabel(xlabel)
    plt.ylabel(ylabe)
    plt.legend()
    plt.show()
def show():
    plt.show()