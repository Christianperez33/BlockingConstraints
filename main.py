import pickle
from generadorFinal import JSP

dir = "./INSTANCES/Problema2/"

with open(f'{dir}{dir.split("/")[-2]}','rb') as file:
        problema:JSP = pickle.load(file)
        print(problema.features())
