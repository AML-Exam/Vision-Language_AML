import numpy as np 
from matplotlib import pyplot as plt 

if __name__ == '__main__':
    history=[]
    i=1

    with open("./record/domain_disentangle_photo/log.txt","r") as file:
        lines=file.readlines()
        for line in lines:
            if line.find("Accuracy") != -1:
                history.append((i, float(line.split("Accuracy: ")[1])))
                i+=1

    history = np.array(history)
    plt.title("Training results") 
    plt.xlabel("Epochs (every 50)") 
    plt.ylabel("Accuracy") 
    plt.plot(history[:,0],history[:,1]) 
    plt.show()