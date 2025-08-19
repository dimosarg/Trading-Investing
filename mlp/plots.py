import matplotlib.pyplot as plt
import numpy as np
from  matplotlib.colors import LinearSegmentedColormap



def plot_line_diagram(y:list,title,label):
    """Λαμβάνει λίστα και δίνει το 2D διάγραμμα της με άξονα x το μήκος της\n\n
    Ορίσματα:\n

    x= Λίστα σημείων, αριθμών, κτλ.\n
    title= Τίτλος διαγράμματος\n
    label= Τί δείχνει το διάγραμμα
    """

    x=np.arange(1,int(len(y)+1))
    plt.figure(1)
    plt.plot(x,y,color="b",linestyle="-",label=label,linewidth=1)
    plt.title(title)
    plt.xlabel("Iterations")
    plt.ylabel(label)
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.close()

def plot_scatter(prices,y,title,num_classes):
    color_map = {-1:"b" ,0:"k", 1:"r", 2:"g"}
    x=np.arange(len(prices))
    plt.figure(1)
    for class_label in range(num_classes):
        idx = y == class_label
        plt.scatter(x[idx], prices[idx], color=color_map[class_label], label=f'Class {class_label}',marker=".")
    plt.legend()
    plt.title(title)
    plt.xlabel("Iterations")
    plt.show()
    plt.close()









def save_line_diagram(y:list,title,label):
    """Λαμβάνει λίστα και δίνει το 2D διάγραμμα της με άξονα x το μήκος της\n\n
    Ορίσματα:\n

    x= Λίστα σημείων, αριθμών, κτλ.\n
    title= Τίτλος διαγράμματος\n
    label= Τί δείχνει το διάγραμμα
    """

    x=np.arange(1,int(len(y)+1))
    plt.figure(1)
    plt.plot(x,y,color="b",linestyle="-",label=label,linewidth=1)
    plt.title(title)
    plt.xlabel("Iterations")
    plt.ylabel(label)
    plt.legend()
    plt.grid(True)
    plt.savefig("diagrams\\"+title+".png")
    plt.close()