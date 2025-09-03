# First we will implement Linear regression using Python 
import numpy as np
import matplotlib.pyplot as plt
def estimate_coef(x,y):
    n= np.size(x)
    m_x=np.mean(x)
    m_y=np.mean(y)
    # Calculating cross-deviation and deviation about x
    SS_xy=np.sum(x*y)- n*m_y*m_x
    SS_xx=np.sum(x*x)- n*m_x*m_x
    #Calculating Regression Cofficient
    b1=SS_xy/SS_xx
    b0=m_y+b1*m_x
    return (b0,b1)

def plot_reg_line(x,y,b):
    #Plotting actual points
    plt.scatter(x,y,color='m',marker='o',s=30)
    y_pred=b[0]+b[1]*x
    #Plotting regression line
    plt.plot(x,y_pred,color='g')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

# Now we will define main function
def main():
    np.random.seed(42)
    x=np.random.randint(20,size=8)
    y=np.random.randint(20,size=8)  
    #estimating cofficients 
    b = estimate_coef(x,y)
    print(b)
    plot_reg_line(x,y,b)

if __name__ == "__main__":
    main()

