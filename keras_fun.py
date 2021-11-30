from seldon_core.seldon_client import SeldonClient
import pandas as pd
import requests as request
from flask import Flask, jsonify

host = "ab18fab544741415b8b57b7ab4bc8532-1539125363.ap-south-1.elb.amazonaws.com"
port = "80"  # Make sure you use the port above
deployment_name = "production-optimization"
namespace = "kubeflow-seldon"
x = 0


def inference(data_path, row_index_number):

    df1 = pd.read_csv(data_path, decimal=",", parse_dates=[
                      "date"], infer_datetime_format=True, error_bad_lines=False, engine="python")
    df1.drop(labels='% Silica Concentrate', axis=1, inplace=True)
    features = ['date', '% Iron Feed', '% Silica Feed', 'Starch Flow', 'Amina Flow', 'Ore Pulp Flow', 'Ore Pulp pH', 'Ore Pulp Density',
                'Flotation Column 01 Air Flow', 'Flotation Column 02 Air Flow', 'Flotation Column 03 Air Flow',
                'Flotation Column 04 Air Flow', 'Flotation Column 05 Air Flow', 'Flotation Column 06 Air Flow', 'Flotation Column 07 Air Flow',
                'Flotation Column 01 Level', 'Flotation Column 02 Level', 'Flotation Column 03 Level',
                'Flotation Column 04 Level', 'Flotation Column 05 Level', 'Flotation Column 06 Level',
                'Flotation Column 07 Level', '% Iron Concentrate']

    df1['date'] = df1['date'].astype(str)
    data = df1.iloc[[row_index_number]]
    df = pd.DataFrame(data, columns=features)

    sc = SeldonClient(deployment_name=deployment_name,
                      gateway_endpoint=host + ":" + port, namespace=namespace)
    client_prediction = sc.predict(
        data=df.values,
        names=df.columns.to_list(),
        gateway="ambassador",
        payload_type="ndarray",
        transport="rest")

    return client_prediction


def gi_df(row_index_number):
    resp = inference('min_flow.csv', row_index_number)
    pred1 = resp.response['data']['ndarray'][0]

    dr = resp.__dict__
    fe_names = dr['request'].data.names
    fe_list = dr['request'].data.ndarray.values[0].list_value.values
    list1 = []
    for t in range(len(fe_list)):
        if t == 0:
            val = fe_list[t].string_value
        else:
            val = fe_list[t].number_value
        list1.append(val)
    feature_df = pd.DataFrame(list(zip(fe_names, list1)), columns=[
                              'Features_name', 'Values'])
    return feature_df, pred1
