from flask import Flask, redirect, url_for, render_template, request, flash, Markup, copy_current_request_context
import io
import base64
import os
import matplotlib.pyplot as plt
import json
import shap
import numpy as np
import pandas as pd
from seldon_core.seldon_client import SeldonClient
import keras_fun
from flask import Flask, jsonify
from getdata import get_val
import os
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array

shap.initjs()

server = "ab18fab544741415b8b57b7ab4bc8532-1539125363.ap-south-1.elb.amazonaws.com"
port_id = "80"
ns = "kubeflow-seldon"

features = ['Ave_Flot_Air_Flow', 'Ave_Flot_Level', '% Iron Feed', 'Starch Flow',
            'Amina Flow', 'Ore Pulp Flow', 'Ore Pulp pH', 'Ore Pulp Density']


def load_input(csv_path, row):

    df1 = pd.read_csv(csv_path).set_index('NewDateTime')
    row_sample = df1.iloc[[row]]
    cols = df1.columns.to_list()[:-1]
    dataframe = pd.DataFrame(row_sample, columns=cols)

    return dataframe, cols


def predictor_inference(payload, host, port, namespace, deployment_name):

    sc = SeldonClient(gateway_endpoint=host + ":" + port,
                      namespace=namespace, deployment_name=deployment_name)
    client_prediction = sc.predict(
        data=payload.values,
        names=features,
        gateway="ambassador",
        payload_type="ndarray",
        transport="rest")

    return client_prediction


def explainer_inference(payload, host, port, namespace, deployment_name):

    sc = SeldonClient(gateway_endpoint=host + ":" + port,
                      namespace=namespace, deployment_name=deployment_name)
    client_prediction = sc.predict(
        data=payload,
        names=[],
        gateway="ambassador",
        payload_type="ndarray",
        transport="rest")

    return client_prediction


def shapley_force_plot(image_name, exp_value, shap_values, sample, columns):
    return shap.force_plot(exp_value,
                           np.array(shap_values[0]),
                           sample,
                           columns,
                           show=False,
                           matplotlib=True).savefig(image_name,
                                                    format="png",
                                                    dpi=150,
                                                    bbox_inches='tight')



#shapley_force_plot('first_plot.png', expected_value, shapley_values, instance, features)

app = Flask(__name__, static_url_path='/static')

df = pd.read_csv('clean_data.csv')
del df['NewDateTime']


df = pd.read_csv('clean_data.csv')
df['index'] = df.index
df.set_index(["index"], inplace=False, drop=True)
first_column = df.pop('index')
df.insert(0, 'index', first_column)
new_df = df.iloc[:5, :]


data, features = load_input('clean_data.csv', 3)
data1 = data.reset_index()

colList = data1.columns.values
colname = colList
row_val = data1.iloc[0]
print(row_val)
fe_df = row_val.to_frame(name='Feature_value').reset_index().rename(
    columns={'index': 'Feature Name'})

ti_tle = fe_df.columns.values

model_plot = ''

@app.route("/", methods=("POST", "GET"))
def landing_page():
    return render_template('index.html')

@app.route("/navigation-page", methods=("POST", "GET"))
def hello_world():
    return render_template(
        'navigation-page.html'
    )


@app.route('/pass_val', methods=['POST', 'GET'])
def pass_val():
    clicked=0
    f =0
    if request.method == "POST":
        clicked=request.form['data']
        t = int(clicked) or 1
        print("data : ", t )
        print("type", type(t))
        data, features = load_input('clean_data.csv', t)
        data1 = data.reset_index()
        
        colList = data1.columns.values
        #colname = colList
        row_val = data1.iloc[0]
        print(row_val)
        fe_df = row_val.to_frame(name='Feature_value').reset_index().rename(
            columns={'index': 'Feature Name'})


        predictor_response = predictor_inference(
            data, server, port_id, ns, deployment_name="predictor")
        transformed_input = json.loads(
            predictor_response.response["data"]["ndarray"][0])[0]
        tr1 = transformed_input.copy()
        tr1.insert(0, 'Not Passed')
        fe_df["Scaled Value"] = tr1
        silica_concentrate = json.loads(
            predictor_response.response["data"]["ndarray"][1])[0]
        predt = silica_concentrate
        print(f"Transformed Input = {transformed_input }")
        print(f"Silica Concentration = {silica_concentrate} %")
        instance = np.array(transformed_input)
        explainer_response = explainer_inference(instance.reshape(
            1, -1), server, port_id, ns, deployment_name="explainer")
        shapley_values = json.loads(
            explainer_response.response["data"]["ndarray"][0])[0]
        expected_value = json.loads(
            explainer_response.response["data"]["ndarray"][1])[0]
        print(f"Shapley Values = {shapley_values}")
        print(f"Expected Value = {expected_value} %")
        #expected_value_rounded = list(map(lambda x: round(x, ndigits=2), expected_value[0]))
        shapley_values_rounded = list(map(lambda x: round(x, ndigits=2), shapley_values[0]))
        shapley_values_1 = []
        shapley_values_1.append(shapley_values_rounded)
        title = fe_df.columns.values
        shapley_force_plot('static\images\plot.png',expected_value, shapley_values, np.round(instance, 2),features)

        predt=round(predt, 2)
        
        return jsonify(predt)
    # colname = colList
    model_plot = 'static\images\plot.png'
    print(f)
    
    return render_template('workspace.html', 
            mod=model_plot,
            tables=[new_df.to_html(classes='data')],
            cols=colname,
            titles=ti_tle,
            intit=ti_tle)



if __name__ == '__main__':
    app.run(debug=True, threaded=True)
