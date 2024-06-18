from flask import Flask, render_template, request
import os
import subprocess
import sys
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
from flask import Flask,request,render_template
import pickle
import requests, json
import csv
import joblib
from geopy.geocoders import Nominatim
from flask import Flask, render_template, request
from flask import Flask,render_template,request,redirect,flash
from werkzeug.utils import secure_filename
from main import getPrediction
import os
import pytest
from selenium import webdriver
import time
import pandas as pd
UPLOAD_FOLDER = 'static/images/'

app = Flask(__name__,static_folder="static")

app.secret_key = "secret key"

app.config['UPLOAD_FOLDER']= UPLOAD_FOLDER
loaded_model = joblib.load('D://Major Project App//saved models//yield.pkl')
data = pd.read_csv('D://Major Project App//saved models//area.csv')

def mol_per_m2_to_ppb(concentration_mol_per_m2, troposphere_height_m):
    # Convert mol/m^2 to mol/m^3
    concentration_mol_per_m3 = concentration_mol_per_m2 / troposphere_height_m
    
    # Convert mol/m^3 to ppb
    molar_volume_air_STP = 22.414  # m^3/mol
    ppb = (concentration_mol_per_m3 * 1e9) / molar_volume_air_STP
    
    return ppb
def infer_air_quality(ppb_value):
        if ppb_value < 20:
            return ("Good air quality for agriculture. The concentration of nitrogen dioxide (NO2) is low, "
                    "indicating minimal pollution in the air. This level of air quality is beneficial for "
                    "crop growth and farm operations. Low levels of NO2 help maintain soil fertility and "
                    "promote healthy plant development. Farmers can proceed with agricultural activities "
                    "without significant concerns about air pollution affecting crop yields or soil health.")
        elif 20 <= ppb_value < 50:
            return ("Moderate air quality for agriculture. The concentration of nitrogen dioxide (NO2) is "
                    "moderate, suggesting some level of pollution in the air. While moderate levels of NO2 "
                    "may not directly impact crop growth, farmers should remain vigilant. Prolonged exposure "
                    "to moderate levels of air pollution can affect plant health and reduce agricultural yields. "
                    "It's advisable to monitor air quality regularly and take preventive measures to protect "
                    "crops from potential adverse effects.")
        else:
            return ("High air pollution level for agriculture. The concentration of nitrogen dioxide (NO2) is "
                    "high, indicating significant pollution in the air. High levels of NO2 pose a serious threat "
                    "to agricultural productivity and farm sustainability. Exposure to high levels of air pollution "
                    "can lead to reduced crop yields, soil degradation, and damage to plant health. Farmers should "
                    "take immediate action to minimize exposure by implementing air quality monitoring systems, "
                    "adopting pollution control measures, and considering alternative farming practices to mitigate "
                    "the adverse effects of air pollution on crops and soil health.")

@app.route('/')
def home():
    return render_template('home.html')
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 00:42:12 2024

@author: saira
"""

@app.route('/geevis',methods=['get'])
def geevis():
    return render_template('gee-visualisation.html')
@app.route('/leaf',methods=['get'])
def leaf():
    return render_template('leaf-disease.html')
@app.route('/ycsv',methods=['get'])
def ycsv():
    return render_template('yield-csv.html')
@app.route('/ygee',methods=['get'])
def ygee():
    return render_template('yield-gee.html')

@app.route('/geevisuali')
def geevisuali():
    image1_path = 'static/images/ndmi.jpg'
    image2_path = 'static/images/ndvi.jpg'
    image3_path = 'static/images/clus.png'
    mean_no2_mol_per_m2 = data['Mean_NO2_Value'].iloc[0]
    mean_no2_ppb = mol_per_m2_to_ppb(mean_no2_mol_per_m2,1000)
    air_quality_statement = infer_air_quality(mean_no2_ppb)
    return render_template('gee-visualisation.html', data=data, image1=image1_path, image2=image2_path, image3=image3_path, air_quality=air_quality_statement)

@app.route('/gee')
def gee():
    options = webdriver.FirefoxOptions()
    driver = webdriver.Firefox(options=options)
    driver.get("https://code.earthengine.google.com/b890ee6e2b9b4334cc480df7e6314497?noload=true")

    try:
        # Wait until the tab is closed
        while True:
            time.sleep(20)  # Check every second if the tab is closed
            if not driver.current_url.startswith("https://code.earthengine.google.com"):
                # If the current URL is not the Earth Engine URL, assume tab is closed
                break
    finally:
            driver.quit()

@app.route('/colab')
def colab():
    options = webdriver.FirefoxOptions()
    driver = webdriver.Firefox(options=options)
    driver.get("https://colab.research.google.com/drive/1O_tH72epXMWJdEdy8dyrsaUmBuErLKIk#scrollTo=MXgF-zxcfAc_")
    try:
        # Wait until the tab is closed
        while True:
            time.sleep(20)  # Check every second if the tab is closed
            if not driver.current_url.startswith("https://colab.research.google.com/"):
            # If the current URL is not the Earth Engine URL, assume tab is closed
                break
    finally:
        driver.quit()

@app.route('/submitt', methods=['POST'])
def submitt():
    if request.method == 'POST':
        # Extract form data
        #annual_rainfall = request.form['annual_rainfall']
       # pesticide = request.form['pesticide']
        #fertilizer = request.form['fertilizer']
        area = request.form['area']
        #production = request.form['production']
        season = request.form['season']
        state = request.form['state']
        crop = request.form['crop']
        
        # Load fertilizer CSV file
        fertilizer_csv_path = "D:\\Major Project App\\saved models\\fertilizer.csv"
        fertilizer_data = pd.read_csv(fertilizer_csv_path)
        mod=joblib.load('D:\\Major Project App\\saved models\\production.pkl')
        model=mod['model']
        lab_enc=mod['label_encoder']

        numeric_features = ['Area']
        categorical_features = ['State_Name','Season','Crop']

        input_data = pd.DataFrame({'State_Name': [state], 'Season': [season], 'Crop': [crop], 'Area': [area]})

        inpdata = lab_enc.transform(input_data)

        prediction = model.predict(inpdata)
        print(prediction[0])
        production=prediction[0]

        print(f'Predicted Target for User Input: {prediction[0]}')

        # Preprocess state values in the fertilizer CSV
        fertilizer_data['State'] = fertilizer_data['State'].str.lower().str.replace(' ', '')

        # Load pesticide CSV file
        pesticide_csv_path = "D:\\Major Project App\\saved models\\pesticide.csv"  # Update this with the path to your pesticide CSV file
        pesticide_data = pd.read_csv(pesticide_csv_path)

        # Preprocess state values in the pesticide CSV
        pesticide_data['State'] = pesticide_data['State'].str.lower().str.replace(' ', '')

        # Example usage
        state_value = state.lower().replace(' ', '')  # Replace "Your_State_Value" with the actual state value
        fertilizer=0
        pesticide=0
        # Fetch fertilizer value
        fertilizer_state_data = fertilizer_data[fertilizer_data['State'] == state_value]
        if not fertilizer_state_data.empty:
            fertilizer_value = fertilizer_state_data['Fertilizer'].values[0]  # Assuming only one value per state
            print(f"Fertilizer value for {state_value}: {fertilizer_value}")
        else:
            print(f"No fertilizer value found for {state_value}")

        # Fetch pesticide value
        pesticide_state_data = pesticide_data[pesticide_data['State'] == state_value]
        if not pesticide_state_data.empty:
            pesticide_value = pesticide_state_data['Pesticide'].values[0]  # Assuming only one value per state
            print(f"Pesticide value for {state_value}: {pesticide_value}")
        else:
            print(f"No pesticide value found for {state_value}")
            
        
        
        fertilizer=fertilizer_value
        pesticide=pesticide_value
        
        
        
        
        
        
        
        def search_annual_rainfall(state):
            # Load the CSV file into a DataFrame
            file_path = "D:\\Major Project App\\saved models\\ar.csv"
            data = pd.read_csv(file_path)
        
            # Convert the 'State' column to lowercase and remove leading/trailing spaces
            data['State'] = data['State'].str.lower().str.strip()
        
            # Convert the annual rainfall data to a dictionary
            state_annual_rainfall = dict(zip(data['State'], data['AR']))
        
            # Search for the annual rainfall of the given state
            annual_rainfall = state_annual_rainfall.get(state.lower().strip())
        
            return float(annual_rainfall.replace(',', ''))
        
        # Example usage
        
        annual_rainfall = search_annual_rainfall(state)
        if annual_rainfall is not None:
            print(f"The annual rainfall of {state.strip()} is {annual_rainfall}")
        else:
            print(f"No data found for {state.strip()}")

        numeric_features = ['Area','Production','Annual_Rainfall','Fertilizer','Pesticide']
        categorical_features = ['Crop','Season','State']      
        model=loaded_model['model']
        label_encoders= loaded_model['label_encoders']
        user_input={}
        user_input['Crop']=label_encoders['Crop'].transform([crop])[0]
        user_input['Season']=label_encoders['Season'].transform([season])[0]
        user_input['State']=label_encoders['State'].transform([state])[0]
        user_input['Area']=[float(area)]
        user_input['Production']=[float(production)]
        user_input['Annual_Rainfall']=[float(annual_rainfall)]
        user_input['Fertilizer']=[float(fertilizer)]
        user_input['Pesticide']=[float(pesticide)]
        
        

        # Create a DataFrame from user input
        user_df = pd.DataFrame(user_input)

        # Predict using the BaggingRegressor
        user_prediction = model.predict(user_df)
        print(f'Predicted Target for User Input: {user_prediction[0]}')

         #Process the data (you can perform any further processing here)
        # For demonstration purposes, let's print the data
        print("Annual Rainfall:", annual_rainfall)
        print("Pesticide:", pesticide)
        print("Fertilizer:", fertilizer)
        print("Area:", area)
        print("Production:", production)
        print("Season:", season)
        print("State:", state)
        print("Crop:", crop)
      
        
        
        
        #prediction=rf.predict(features)
        #print(prediction)
        # Return a response (you can customize this as needed)
        
        return render_template('yield-csv.html', predicted_value=user_prediction[0])
@app.route('/disc',methods=['POST'])
def submit_file():
    if request.method=='POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file=request.files['file']
        if file.filename=='':
            flash('No file selected for uploading')
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)  #Use this werkzeug method to secure filename. 
            file.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))
            #getPrediction(filename)
            label = getPrediction(filename)
            print(label) 
            name=label
            flash("The Identified Disease is " + label )
            #flash("The possible pesticides are " + rec)
            full_filename = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            flash(full_filename)
            return redirect('/leaf')
if __name__ == '__main__':
    app.run()
