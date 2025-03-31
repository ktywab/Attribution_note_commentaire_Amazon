import joblib
from flask import Flask, render_template, request, url_for

#Création d'objet de la classe Flask
app = Flask(__name__)

# Chargement du modèle
model = joblib.load('modele.pkl')


#Première route
@app.route('/')
def index():
    return render_template('index.html')

#Deuxième route
@app.route('/predict',methods=["post"])
def predict():
    # Récupérer les données du formulaire
    #request.form permet de recupérer les données du formulaires sous formre de dictionnaire
    features = [[
    int(request.form.get('Code_commune', 0)),
    int(request.form.get('Code_departement', 0)),
    int(request.form.get('Nbre_lots', 0)),
    float(request.form.get('Code_Nature', 0.0)),#1 pour terrain agricole(Prés) et 0 pour terrain occupé par des batiments(Sols)
    float(request.form.get('Surface_terrain', 0.0)),
    float(request.form.get('Longitude', 0.0))
    ]]

    prediction = model.predict(features)[0]
    prediction_proba = model.predict_proba(features)[0][1]#Proba de tombé sur 1

    return render_template('predict.html',prediction_classe = prediction,prob_defaut = prediction_proba)



if __name__=='__main__' :
    app.run(debug=True)