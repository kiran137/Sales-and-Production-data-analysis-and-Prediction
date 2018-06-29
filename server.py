from flask import Flask, request
from flask_cors import CORS, cross_origin
from flask_restful import Resource, Api
from json import dumps
from flask_jsonpify import jsonify

import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
from matplotlib.pylab import rcParams
from datetime import datetime
app = Flask(__name__)
api = Api(app)

from sklearn.externals import joblib

CORS(app)

@app.route("/get_",methods=["POST"])
def get_():
    #date_=request.form['date_']
    #tons=request.form['tons']
    #return json.dumps({'status':'OK'});  
    data = request.get_json(force=True)

    date_= datetime.strptime(data['date_'], '%Y-%m-%d').toordinal()
    
    qty=float(data["tons"])
    lin_reg = joblib.load("regression_model.pkl")
    dat= lin_reg.predict(np.array([[qty,date_]]))
    dat=np.round(dat,2)
   
    dat=dat.tolist()  
    return jsonify(dat) 

##api.add_resource(Employees_Name, '/employees/<employee_id>') # Route_3
@app.route("/get1_",methods=["POST"])

def get1_():
    data1=request.get_json(force=True)
    date_=datetime.strptime(data1['date_'],'%Y-%m-%d').toordinal()
    
    qty=float(data1["tons"])
  
    lin_reg1 = joblib.load("regression_model1.pkl")
    dat1= lin_reg1.predict(np.array([[date_,qty]]))
    dat1=dat1.tolist()

    return jsonify(dat1) 

#@app.route("/get2_",methods=["POST"])

#def get2_():
 #   data2=request.get_json(force=True)
  #  date_=datetime.strptime(data2['date_'],'%Y-%m-%d').toordinal()
    
   # qty=float(data2["tons"])
    #print(date_,qty)
    #lin_reg2 = joblib.load("regression_model2.pkl")
    #dat2= lin_reg2.predict(np.array([[date_,qty]]))
    
    #dat2=dat2.tolist()

    #return jsonify(dat2)
    


if __name__ == '__main__':
     app.run(port=8080)