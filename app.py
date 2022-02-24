#!/usr/bin/env python
# coding: utf-8

# In[1]:


from flask import Flask


# In[2]:


app = Flask(__name__)


# In[3]:


from flask import request, render_template
from keras.models import load_model
import pickle

@app.route("/", methods=["GET","POST"])
def index():
    if request.method=="POST":
        Income = request.form.get("Income")
        Age = request.form.get("Age")
        Loan = request.form.get("Loan")
        print(Income, Age, Loan)
        
        modelLR = pickle.load(open("LRmodel.sav", 'rb'))
        predLR = modelLR.predict([[float(Income), float(Age), float(Loan)]])
        sLR = "Predicted default on credit card (Logistic Regression): " + str(predLR)
        
        modelDT = pickle.load(open("DTmodel.sav", 'rb'))
        predDT = modelDT.predict([[float(Income), float(Age), float(Loan)]])
        sDT = "Predicted default on credit card (Decision Tree): " + str(predDT)
    
        modelRF = pickle.load(open("RFmodel.sav", 'rb'))
        predRF = modelRF.predict([[float(Income), float(Age), float(Loan)]])
        sRF = "Predicted default on credit card (Random Forest): " + str(predRF)
    
        modelXG = pickle.load(open("XGBmodel.sav", 'rb'))
        predXG = modelXG.predict([[float(Income), float(Age), float(Loan)]])
        sXG = "Predicted default on credit card (XGBoost): " + str(predXG)
    
        modelNN = load_model("NNKmodel")
        predNN = modelNN.predict([[float(Income), float(Age), float(Loan)]])
        sNN = "Predicted default on credit card (Neural Network-Keras): " + str(predNN)
        
        modelMLP = pickle.load(open("NNmodel.sav", 'rb'))
        predMLP = modelMLP.predict([[float(Income), float(Age), float(Loan)]])
        sMLP = "Predicted default on credit card (Neural Network-MLPClassifier): " + str(predMLP)
        
        return(render_template("index.html", result1 = sLR, result2 = sDT, result3 = sRF, result4 = sXG, result5 = sNN, result6 = sMLP))
    
    else: 
        s = "."
        return(render_template("index.html", result1 = s, result2 = s, result3 = s, result4 = s, result5 = s, result6 = s))


# In[ ]:


if __name__ == "__main__":
    app.run()

