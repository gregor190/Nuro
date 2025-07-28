import math

def relu(x):
    return max(0, x)
#relu end
def relu_derivative(x):
    return 1 if x > 0 else 0
#relu derivitive end
def sigmoid(x):
    return 1 / (1 + math.exp(-x))
#sigmoid end    
def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)    
#sigmoid derivitevi end
def tanh(x):
    return math.tanh(x)
#tanh end
def tanh_derivative(x):
    return 1 - math.tanh(x) ** 2
#tanh derivitev end   
