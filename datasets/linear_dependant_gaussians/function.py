from numpy import random

def random_inputs_function():
    return [random.random(), random.random()]

def x_of_inputs(params): # renvoie un output dans ]0, 1[
    a, b = params
    centre = a*.6+.2 # centre entre .2 et .8
    ecart_type = .05+b/5 # variance entre .05 et .25
    
    value = -1
    while value<0 or value>1:
        value = random.normal(centre, scale=ecart_type)
    
    return [value]