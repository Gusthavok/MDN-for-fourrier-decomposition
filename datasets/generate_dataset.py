import numpy as np
import csv

def generate_data(x_of_inputs, random_inputs_function, size, path):
    """
    x_of_inputs : return a list representing x
    random_inputs_function : return a random input
    size : size of the dataset created
    """
    data=[]
    for _ in range(size):
        inputs = random_inputs_function()
        output = x_of_inputs(inputs)
        data.append(inputs+output)
    
    tableau = np.array(data)
    with open(path+'tensor_output.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(tableau)
        

if __name__=="__main__":
    from linear_dependant_gaussians.function import random_inputs_function, x_of_inputs
    
    generate_data(x_of_inputs, random_inputs_function, 100, path='./linear_dependant_gaussians/')
    