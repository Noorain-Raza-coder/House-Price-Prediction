from flask import Flask ,render_template , request
import pandas as pd
import pickle
import sklearn
import numpy as np
from sklearn.preprocessing import StandardScaler


app = Flask(__name__)
df = pd.read_csv('1. Regression - Module - (Housing Prices).csv')
model = pickle.load(open("LinearRegression.pkl",'rb'))

scalar = StandardScaler()


@app.route('/')
def index ():
    return render_template('index.html')

@app.route('/getvalue', methods = ['POST','GET'])
def getvalue ():
    # flatarea = request.form['flatarea']
    # bathrooms = request.form['bathrooms']
    # print(flatarea , bathrooms)

    # input = pd.DataFrame([[flatarea,bathrooms]],columns = ['Flat Area (in Sqft)','No of Bedrooms'])
    # prediction = model.predict(input)[0]

    data = [float(x) for x in request.form.values()]
    print(data)
    print(len(data))
    # 1 step ## >> reshape = (np.array(data).reshape(1,-1))
    # final_data = scalar.fit_transform(reshape)


    # final_data = scalar.fit_transform(np.array(data).reshape(1, -1))

    # good result> final_data = np.array(data).reshape(1, -1)
    # prediction = model.predict(final_data)[0] >

    reshape = np.array(data).reshape(1, -1)
    ## >>>>>>>>>> final_data = scalar.fit_transform(reshape) <<<<<<<<<<<<< imp


    prediction = model.predict(reshape)
    return render_template('pass.html',pred = str(prediction))

if __name__ == '__main__':
    app.run(debug=True ,port=5001)

    # data = scalar.fit_transform(data)
    # final_data = pd.DataFrame([[data]] , columns = ['No of Bedrooms', 'No of Bathrooms', 'Lot Area (in Sqft)','No of Floors', 'Overall Grade','Area of the House from Basement (in Sqft)', 'Basement Area (in Sqft)','Age of House (in Years)', 'Latitude', 'Longitude','Living Area after Renovation (in Sqft)','Lot Area after Renovation (in Sqft)', 'Year Since Renovation','Waterfront_View_Yes', 'No_of_Times_Visited_Once','No_of_Times_Visited_Thrice', 'No_of_Times_Visited_Twice','Condition_of_the_House_Excellent', 'Condition_of_the_House_Good','Condition_of_the_House_Okay', 'Ever_Renovated_Yes','Zipcode_grp_Zipcode_grp_1', 'Zipcode_grp_Zipcode_grp_2','Zipcode_grp_Zipcode_grp_3', 'Zipcode_grp_Zipcode_grp_4','Zipcode_grp_Zipcode_grp_5', 'Zipcode_grp_Zipcode_grp_6','Zipcode_grp_Zipcode_grp_7', 'Zipcode_grp_Zipcode_grp_8','Zipcode_grp_Zipcode_grp_9'])
