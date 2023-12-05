import dill as pickle

with open('models/healingyuk.pkl', 'rb') as file:
    model = pickle.load(file)
    print(model(-7.016901205125758, 110.46777246024239))