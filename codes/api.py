import pandas as pd
from fastapi import FastAPI
import dill
import traceback
import uvicorn




app = FastAPI()

with open('../models/vot.joblib','rb') as model_file:
    model = dill.load(model_file)

with open('../models/features.joblib','rb') as features_file:
    features = dill.load(features_file)



@app.post('/prediction')
def predict(Pclass: int , Name: str, Sex: str, Age:int, Ticket:str, Fare:float, Cabin:str, Embarked:str ,Relatives:int):
    """
    Class - Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd) |
    Relatives - Number of Relatves Aboard |
    Ticket - Ticket Number |
    Fare - Passenger Fare |
    Embarked - Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)
    """
    if model:
        try:
            query = {'Pclass':Pclass,'Name':Name, 'Sex':Sex, 'Age':Age, 'Ticket':Ticket, 'Fare':Fare, 'Cabin':Cabin, 'Embarked':Embarked,'Relatives':Relatives} 
            query = pd.DataFrame(query,index=[0])
            query = query.reindex(columns=features, fill_value=0)
            predict = model.predict(query)

            return {'prediction':str(predict)}
        except:
            return {'trace': traceback.format_exc()}
    else:
        print ('Model is not good')
        return ('Model is not good')

        
if __name__ == '__main__':
    uvicorn.run(app)
