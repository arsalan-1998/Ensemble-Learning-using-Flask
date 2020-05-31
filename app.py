from flask import Flask, flash, render_template, request
import pymongo
import math
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.exceptions import ConvergenceWarning
from warnings import simplefilter

simplefilter(action='ignore', category = FutureWarning)
simplefilter(action='ignore',category=ConvergenceWarning)
csv_data = pd.read_csv('heart.csv')

def model_svm(data, udata):
    u_test = udata
    data = data 
    target = data['target'] 
    data = data.drop(['target'],axis=1) 
    kfolds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1) 
    pipeline_svm = make_pipeline(SVC(probability=True, kernel='linear', class_weight='balanced')) 
    grid_svm = GridSearchCV(pipeline_svm, 
    param_grid = {'svc__C': [0.01, 0.1, 1]}, 
    cv = kfolds, 
    verbose=1, 
    n_jobs=-1) 
    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=10) 
    grid_svm.fit(x_train, y_train) 
    accuracy = grid_svm.score(x_test, y_test)
    joblib.dump(grid_svm, "heart_disease.pkl") 
    model_grid_svm = joblib.load('heart_disease.pkl')
    res = model_grid_svm.predict(u_test)
    p = model_grid_svm.predict_proba(u_test)[:, 1] 
    p_no = model_grid_svm.predict_proba(u_test)[:, 0] 
    return res, grid_svm, p, p_no, accuracy 

def model_mnb(data, udata):
    u_test = udata
    data = data
    target=data['target']
    data = data.drop(['target'],axis=1)
    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=10)
    classifierNB=MultinomialNB()
    classifierNB.fit(x_train,y_train)
    accuracy = classifierNB.score(x_test, y_test)
    joblib.dump(classifierNB, 'heart_disease_mnb.pkl')
    model_classifierNB = joblib.load('heart_disease_mnb.pkl' )
    res = model_classifierNB.predict(u_test)
    p1 = model_classifierNB.predict_proba(u_test)[:, 1]
    p1_no = model_classifierNB.predict_proba(u_test)[:, 0]
    return res, classifierNB, p1, p1_no, accuracy
 
def model_lr(data, udata):
    u_test = udata
    data = data
    target=data['target']
    data = data.drop(['target'],axis=1)
    data.head()
    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=10)
    classifierLR=LogisticRegression()
    classifierLR.fit(x_train,y_train)
    accuracy = classifierLR.score(x_test, y_test)
    joblib.dump(classifierLR, 'heart_disease_lr.pkl')
    model_classifierLR = joblib.load('heart_disease_lr.pkl' )
    res = model_classifierLR.predict(u_test)
    p2 = model_classifierLR.predict_proba(u_test)[:, 1]
    p2_no = model_classifierLR.predict_proba(u_test)[:, 0]
    return res, classifierLR, p2, p2_no, accuracy

def model_dt(data,udata):
    u_test = udata
    data = data
    target=data['target']
    data = data.drop(['target'],axis=1)
    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=10)
    classifierDT=DecisionTreeClassifier(criterion='gini', random_state=50, max_depth=3, min_samples_leaf=5)
    classifierDT.fit(x_train,y_train)
    accuracy = classifierDT.score(x_test, y_test)
    joblib.dump(classifierDT, 'heart_disease_dt.pkl')
    model_classifierDT = joblib.load('heart_disease_dt.pkl' )
    res = model_classifierDT.predict(u_test)
    p3 = model_classifierDT.predict_proba(u_test)[:, 1]
    p3_no = model_classifierDT.predict_proba(u_test)[:, 0] 
    return res, classifierDT, p3, p3_no, accuracy

def model_rf(data, udata):
    u_test = udata
    data = data
    target=data['target']
    data = data.drop(['target'],axis=1)
    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=10)
    classifierRF=RandomForestClassifier()
    classifierRF.fit(x_train,y_train)
    accuracy = classifierRF.score(x_test, y_test)
    joblib.dump(classifierRF, 'heart_disease_logice.pkl')
    model_classifierRF = joblib.load('heart_disease_logice.pkl' )
    res = model_classifierRF.predict(u_test)
    p4 = model_classifierRF.predict_proba(u_test)[:, 1]
    p4_no = model_classifierRF.predict_proba(u_test)[:, 0] 
    return res, classifierRF, p4, p4_no, accuracy

data = csv_data # reference of the csv file stored in the data variable for ensemble model
target = data['target'] # seggregate the target column 
data = data.drop(['target'],axis=1) # delete the target colum from the dataset
x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=10) # train and test split using sklearn

def ensemble_model(udata, grid_svm, classifierNB, classifierLR, classifierDT, classifierRF): #ensemble model takes 6 parameters ( udata=user-data, grid_svm=svm-classifier, classifierNB = Multinomial Classifier, classifierLR = Logistic Rgression, classifierDT = Decision Tree, classifierRF = Random FOrest )
    u_test = udata #user-data is stored in udata
    estimators=[('svm', grid_svm), ('nb', classifierNB), ('lr', classifierLR), ('dt', classifierDT),('rf', classifierRF)] # creates a models list of tuples with {key:value} pairs
    # Hard Voting Classifier : Aggregate predections of each classifier and predict the class that gets most votes. This is called as “majority – voting”
    majority_voting = VotingClassifier(estimators, voting='soft') #Voting Classifier with voting as soft. In soft voting, we predict the class labels by averaging the class-probabilities (only recommended if the classifiers are well-calibrated).
    majority_voting.fit(x_train, y_train)# train the model 
    accuracy = majority_voting.score(x_test, y_test)# get the accuracy of the model
    joblib.dump(majority_voting, 'heart_disease_ensemble.pkl') # Serialize the Object
    model_max_v = joblib.load('heart_disease_ensemble.pkl') # DeSerialize the Object
    v_prob = model_max_v.predict_proba(u_test)[:, 1] # predicts the positive probability
    v_prob_no = model_max_v.predict_proba(u_test)[:, 0] # predicts the Negative Probability
    res = model_max_v.predict(u_test) # predicts the user-data
    return res, v_prob, v_prob_no, accuracy # returns the predicted results and probabilities.

app = Flask(__name__) # Creates an instance of Flask
app.config['SECRET_KEY'] = 'hello_flask_app' # defines the secret key for the app inorder to use flash messages
mongo = pymongo.MongoClient() # creates a connection to MongoDB by default no params are required

@app.route('/') #defines a root route
def hello_world(): # method to handle at the root route
    return render_template('index.html') #if the server responds (200) status code the it returns the html page to user pyt
@app.route('/signup', methods=['GET','POST']) #create a /signup route which    
def signup():
    db = mongo.heart.users
    if request.method == 'GET':
        return render_template("index.html")
    else:
        data = [ request.form['user_email'], request.form['user_password'], request.form['re_password'] ]  
        user_obj = {
            "email" : data[0],
            "psw" : data[1],
            "re_psw" : data[2]
        } 
        user = db.find()   
        if (user.count() ==0): 
            if (user_obj['psw'] == user_obj['re_psw']) :
                val = db.save(user_obj) 
                success = 'User Registered'
                flash('You have Successfully Registered', 'success') 
                isLoggedin = True
                return render_template('index.html', data=success, isLoggedin=True)
            else:
                success = 'Error'
                flash('Please check your Password is matching', 'error') 
                isLoggedin = True
                return render_template('index.html', data=success, isLoggedin=True)
        else:
            for u in user:
                if ((u['email'] == user_obj['email'])) :
                    if ((user_obj['psw'] == user_obj['re_psw'])):
                        success = 'User Exists'
                        flash('Please Sign-in with your Account', 'error') 
                        isLoggedin = True  
                        return render_template('index.html', data=success, isLoggedin=True)
                    else: 
                        success = 'Error'
                        flash('Please check your email / password', 'error') 
                        isLoggedin = True  
                        return render_template('index.html', data=success, isLoggedin=True)
        val = db.save(user_obj)
        success = 'Success' 
        flash('You have Registered Successfully', 'success') 
        isLoggedin = True
        return render_template('index.html', data=success, isLoggedin=True)
            

@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'GET':
        return render_template("index.html")
    else:
        db = mongo.heart.users 
        data = [ request.form['user_email'], request.form['user_password'] ] 
        user = db.find() 
        if (user.count() == 0):
            success = 'Please Create an Account'
            flash('Sorry! No such Account Registered with us', 'error')
            return render_template('index.html', data=success)
        else:
            for u in user:
                if ((u['email'] == data[0])) :
                    if ((u['psw'] == data[1])):
                        return render_template('user-data.html')
                    else:
                        success = 'Sorry'
                        flash('Please check email / password', 'error') 
                        isLoggedin = True  
                        return render_template('index.html', data=success, isLoggedin=True)
            else:
                success = 'Sorry'
                flash('Please Create an Account', 'error') 
                isLoggedin = True  
                return render_template('index.html', data=success, isLoggedin=True)

@app.route('/contact', methods=['GET','POST'])
def contact():
    if request.method == 'GET':
        return render_template("index.html")
    else:
        data = {
            "name" : request.form['name'],
            "email" : request.form['email'],
            "subject" : request.form['subject'],
            "message" : request.form['message']    
        }
        return render_template('thank-you.html', data=data)

@app.route('/user-data', methods=['POST'])
def process_user_data():
    data = {
        "age": request.form['age'],
        "sex": request.form['sex'],
        "cp": request.form['cp'],
        "trestbps": request.form['trestbps'],
        "chol": request.form['chol'],
        "fbs": request.form['fbs'],
        "restecg": request.form['restecg'],
        "thalach": request.form['thalach'],
        "exang": request.form['exang'],
        "oldpeak": request.form['oldpeak'],
        "slope": request.form['slope'],
        "ca": request.form['ca'],
        "thal": request.form['thal']
    } 
    data = pd.DataFrame(data, index=[0])
    for v in data['sex']:
        if( v == 'male' ):
            data['sex'] = 1
        if( v == 'Female' ):
            data['sex'] = 0 

    val = model_mnb(data = csv_data, udata = data) 
    val2 = model_dt(data = csv_data, udata = data)
    val3 = model_lr(data = csv_data, udata = data)
    val4 = model_rf(data = csv_data, udata = data)
    val5 = model_svm(data = csv_data, udata = data)
    fin_model = ensemble_model(udata = data, grid_svm = val5[1], classifierNB = val[1], 
    classifierLR = val3[1], classifierDT = val2[1], classifierRF = val4[1])
    data = {
        "SVM_Result" : val5[0], 
        "Random_Forest" : val4[0],  
        "Logistic_Regression" : val3[0], 
        "Decision_Tree_Algorithm" : val2[0],  
        "Multinomial_nb" : val[0],
        "ensemble_model" : fin_model[0]
    }

    accuracy = {
        "svm_accuracy" : math.ceil(val5[4]*100),
        "random_forest_accuracy" : math.ceil(val4[4]*100),
        "logistic_regression_accuracy" : math.ceil(val3[4]*100),
        "decision_tree_accuracy": math.ceil(val2[4]*100),
        "multinomialnb_accuracy": math.ceil(val[4]*100), 
        "ensemble_accuracy" : math.ceil(fin_model[3]*100)

    }
    
    if((val[0] == 1) and (val2[0] == 1) and (val3[0] == 1) and (val4[0] == 1) and (val5[0] == 1) and (fin_model[0] == 1)):
        print('Iam here')
        prob = {
        "prob_1" : val[2]*100, 
        "prob_2" : val2[2]*100,
        "prob_3" : val3[2]*100,
        "prob_4" : val4[2]*100,
        "prob_5" : val5[2]*100,
        "v_prob" : fin_model[1]*100
    }
        return render_template('result.html',data=data, data1=prob, data2=accuracy)
    elif((val[0] == 1) and (val2[0] == 1) and (val3[0] == 1) and (val4[0] == 1) and (val5[0] == 1) and (fin_model[0] == 0)):
        prob = {
        "prob_1" : val[2]*100,
        "prob_2" : val2[2]*100,
        "prob_3" : val3[2]*100,
        "prob_4" : val4[2]*100,
        "prob_5" : val5[2]*100,
        "v_prob" : fin_model[2]*100
    }
        return render_template('result.html',data=data, data1=prob, data2=accuracy)
    elif((val[0] == 1) and (val2[0] == 1) and (val3[0] == 1) and (val4[0] == 1) and (val5[0] == 0) and (fin_model[0] == 1)):
        prob = {
        "prob_1" : val[2]*100,
        "prob_2" : val2[2]*100,
        "prob_3" : val3[2]*100,
        "prob_4" : val4[2]*100,
        "prob_5" : val5[3]*100,
        "v_prob" : fin_model[1]*100
    }
        return render_template('result.html',data=data, data1=prob, data2=accuracy)
    elif((val[0] == 1) and (val2[0] == 1) and (val3[0] == 1) and (val4[0] == 1) and (val5[0] == 0) and (fin_model[0] == 0)):
        prob = {
        "prob_1" : val[2]*100,
        "prob_2" : val2[2]*100,
        "prob_3" : val3[2]*100,
        "prob_4" : val4[2]*100,
        "prob_5" : val5[3]*100,
        "v_prob" : fin_model[1]*100
    }
        return render_template('result.html',data=data, data1=prob, data2=accuracy)
    elif((val[0] == 1) and (val2[0] == 1) and (val3[0] == 1) and (val4[0] == 0) and (val5[0] == 1) and (fin_model[0] == 1)):
        prob = {
        "prob_1" : val[2]*100,
        "prob_2" : val2[2]*100,
        "prob_3" : val3[2]*100,
        "prob_4" : val4[3]*100,
        "prob_5" : val5[2]*100,
        "v_prob" : fin_model[1]*100
    }
        return render_template('result.html',data=data, data1=prob, data2=accuracy)
    elif((val[0] == 1) and (val2[0] == 1) and (val3[0] == 1) and (val4[0] == 0) and (val5[0] == 1) and (fin_model[0] == 0)):
        prob = {
        "prob_1" : val[2]*100,
        "prob_2" : val2[2]*100,
        "prob_3" : val3[2]*100,
        "prob_4" : val4[3]*100,
        "prob_5" : val5[2]*100,
        "v_prob" : fin_model[2]*100
    }
        return render_template('result.html',data=data, data1=prob, data2=accuracy)
    elif((val[0] == 1) and (val2[0] == 1) and (val3[0] == 1) and (val4[0] == 0) and (val5[0] == 0) and (fin_model[0] == 1)):
        prob = {
        "prob_1" : val[2]*100,
        "prob_2" : val2[2]*100,
        "prob_3" : val3[2]*100,
        "prob_4" : val4[3]*100,
        "prob_5" : val5[3]*100,
        "v_prob" : fin_model[1]*100
    }
        return render_template('result.html',data=data, data1=prob, data2=accuracy)
    elif((val[0] == 1) and (val2[0] == 1) and (val3[0] == 1) and (val4[0] == 0) and (val5[0] == 0) and (fin_model[0] == 0)):
        prob = {
        "prob_1" : val[2]*100,
        "prob_2" : val2[2]*100,
        "prob_3" : val3[2]*100,
        "prob_4" : val4[3]*100,
        "prob_5" : val5[3]*100,
        "v_prob" : fin_model[2]*100
    }
        return render_template('result.html',data=data, data1=prob, data2=accuracy)
    elif((val[0] == 1) and (val2[0] == 1) and (val3[0] == 0) and (val4[0] == 1) and (val5[0] == 1) and (fin_model[0] == 1)):
        prob = {
        "prob_1" : val[2]*100,
        "prob_2" : val2[2]*100,
        "prob_3" : val3[3]*100,
        "prob_4" : val4[2]*100,
        "prob_5" : val5[2]*100,
        "v_prob" : fin_model[1]*100
    }
        return render_template('result.html',data=data, data1=prob, data2=accuracy)
    elif((val[0] == 1) and (val2[0] == 1) and (val3[0] == 0) and (val4[0] == 1) and (val5[0] == 0) and (fin_model[0] == 1)):
        prob = {
        "prob_1" : val[2]*100,
        "prob_2" : val2[2]*100,
        "prob_3" : val3[3]*100,
        "prob_4" : val4[2]*100,
        "prob_5" : val5[3]*100,
        "v_prob" : fin_model[1]*100
    }
        return render_template('result.html',data=data, data1=prob, data2=accuracy)
    elif((val[0] == 1) and (val2[0] == 1) and (val3[0] == 0) and (val4[0] == 1) and (val5[0] == 0) and (fin_model[0] == 0)):
        prob = {
        "prob_1" : val[2]*100,
        "prob_2" : val2[2]*100,
        "prob_3" : val3[3]*100,
        "prob_4" : val4[2]*100,
        "prob_5" : val5[3]*100,
        "v_prob" : fin_model[2]*100
    }
        return render_template('result.html',data=data, data1=prob, data2=accuracy)
    elif((val[0] == 1) and (val2[0] == 1) and (val3[0] == 0) and (val4[0] == 0) and (val5[0] == 1) and (fin_model[0] == 1)):
        prob = {
        "prob_1" : val[2]*100,
        "prob_2" : val2[3]*100,
        "prob_3" : val3[3]*100,
        "prob_4" : val4[3]*100,
        "prob_5" : val5[2]*100,
        "v_prob" : fin_model[1]*100
    }
        return render_template('result.html',data=data, data1=prob, data2=accuracy)
    elif((val[0] == 1) and (val2[0] == 1) and (val3[0] == 0) and (val4[0] == 0) and (val5[0] == 1) and (fin_model[0] == 0)):
        prob = {
        "prob_1" : val[2]*100,
        "prob_2" : val2[2]*100,
        "prob_3" : val3[3]*100,
        "prob_4" : val4[3]*100,
        "prob_5" : val5[2]*100,
        "v_prob" : fin_model[2]*100
    }
        return render_template('result.html',data=data, data1=prob, data2=accuracy)
    elif((val[0] == 1) and (val2[0] == 1) and (val3[0] == 0) and (val4[0] == 0) and (val5[0] == 0) and (fin_model[0] == 1)):
        prob = {
        "prob_1" : val[2]*100,
        "prob_2" : val2[2]*100,
        "prob_3" : val3[3]*100,
        "prob_4" : val4[3]*100,
        "prob_5" : val5[3]*100,
        "v_prob" : fin_model[1]*100
    }
        return render_template('result.html',data=data, data1=prob, data2=accuracy)
    elif((val[0] == 1) and (val2[0] == 1) and (val3[0] == 0) and (val4[0] == 0) and (val5[0] == 0) and (fin_model[0] == 0)):
        prob = {
        "prob_1" : val[2]*100,
        "prob_2" : val2[2]*100,
        "prob_3" : val3[3]*100,
        "prob_4" : val4[3]*100,
        "prob_5" : val5[3]*100,
        "v_prob" : fin_model[2]*100
    }
        return render_template('result.html',data=data, data1=prob, data2=accuracy)
    elif((val[0] == 1) and (val2[0] == 0) and (val3[0] == 1) and (val4[0] == 1) and (val5[0] == 1) and (fin_model[0] == 1)):
        prob = {
        "prob_1" : val[2]*100,
        "prob_2" : val2[3]*100,
        "prob_3" : val3[2]*100,
        "prob_4" : val4[2]*100,
        "prob_5" : val5[2]*100,
        "v_prob" : fin_model[1]*100
    }
        return render_template('result.html',data=data, data1=prob, data2=accuracy)
    elif((val[0] == 1) and (val2[0] == 0) and (val3[0] == 1) and (val4[0] == 1) and (val5[0] == 1) and (fin_model[0] == 0)):
        prob = {
        "prob_1" : val[2]*100,
        "prob_2" : val2[3]*100,
        "prob_3" : val3[2]*100,
        "prob_4" : val4[2]*100,
        "prob_5" : val5[2]*100,
        "v_prob" : fin_model[2]*100
    }
        return render_template('result.html',data=data, data1=prob, data2=accuracy)
    elif((val[0] == 1) and (val2[0] == 0) and (val3[0] == 1) and (val4[0] == 1) and (val5[0] == 0) and (fin_model[0] == 1)):
        prob = {
        "prob_1" : val[2]*100,
        "prob_2" : val2[3]*100,
        "prob_3" : val3[2]*100,
        "prob_4" : val4[2]*100,
        "prob_5" : val5[3]*100,
        "v_prob" : fin_model[1]*100
    }
        return render_template('result.html',data=data, data1=prob, data2=accuracy)
    elif((val[0] == 1) and (val2[0] == 0) and (val3[0] == 1) and (val4[0] == 0) and (val5[0] == 1) and (fin_model[0] == 1)):
        prob = {
        "prob_1" : val[2]*100,
        "prob_2" : val2[3]*100,
        "prob_3" : val3[2]*100,
        "prob_4" : val4[3]*100,
        "prob_5" : val5[2]*100,
        "v_prob" : fin_model[1]*100
    }
        return render_template('result.html',data=data, data1=prob, data2=accuracy)
    elif((val[0] == 1) and (val2[0] == 0) and (val3[0] == 1) and (val4[0] == 0) and (val5[0] == 1) and (fin_model[0] == 0)):
        prob = {
        "prob_1" : val[2]*100,
        "prob_2" : val2[3]*100,
        "prob_3" : val3[2]*100,
        "prob_4" : val4[3]*100,
        "prob_5" : val5[2]*100,
        "v_prob" : fin_model[2]*100
    }
        return render_template('result.html',data=data, data1=prob, data2=accuracy)
    elif((val[0] == 1) and (val2[0] == 0) and (val3[0] == 1) and (val4[0] == 0) and (val5[0] == 0) and (fin_model[0] == 1)):
        prob = {
        "prob_1" : val[2]*100,
        "prob_2" : val2[3]*100,
        "prob_3" : val3[2]*100,
        "prob_4" : val4[3]*100,
        "prob_5" : val5[3]*100,
        "v_prob" : fin_model[1]*100
    }
        return render_template('result.html',data=data, data1=prob, data2=accuracy)
    elif((val[0] == 1) and (val2[0] == 0) and (val3[0] == 1) and (val4[0] == 0) and (val5[0] == 0) and (fin_model[0] == 0)):
        prob = {
        "prob_1" : val[2]*100,
        "prob_2" : val2[3]*100,
        "prob_3" : val3[2]*100,
        "prob_4" : val4[3]*100,
        "prob_5" : val5[3]*100,
        "v_prob" : fin_model[2]*100
    }
        return render_template('result.html',data=data, data1=prob, data2=accuracy)
    elif((val[0] == 1) and (val2[0] == 0) and (val3[0] == 0) and (val4[0] == 1) and (val5[0] == 1) and (fin_model[0] == 1)):
        prob = {
        "prob_1" : val[2]*100,
        "prob_2" : val2[3]*100,
        "prob_3" : val3[3]*100,
        "prob_4" : val4[2]*100,
        "prob_5" : val5[2]*100,
        "v_prob" : fin_model[1]*100
    }
        return render_template('result.html',data=data, data1=prob, data2=accuracy)
    elif((val[0] == 1) and (val2[0] == 0) and (val3[0] == 0) and (val4[0] == 1) and (val5[0] == 1) and (fin_model[0] == 0)):
        prob = {
        "prob_1" : val[2]*100,
        "prob_2" : val2[3]*100,
        "prob_3" : val3[3]*100,
        "prob_4" : val4[2]*100,
        "prob_5" : val5[2]*100,
        "v_prob" : fin_model[2]*100
    }
        return render_template('result.html',data=data, data1=prob, data2=accuracy)
    elif((val[0] == 1) and (val2[0] == 0) and (val3[0] == 0) and (val4[0] == 1) and (val5[0] == 0) and (fin_model[0] == 1)):
        prob = {
        "prob_1" : val[2]*100,
        "prob_2" : val2[3]*100,
        "prob_3" : val3[3]*100,
        "prob_4" : val4[2]*100,
        "prob_5" : val5[3]*100,
        "v_prob" : fin_model[1]*100
    }
        return render_template('result.html',data=data, data1=prob, data2=accuracy)
    elif((val[0] == 1) and (val2[0] == 0) and (val3[0] == 0) and (val4[0] == 1) and (val5[0] == 0) and (fin_model[0] == 0)):
        prob = {
        "prob_1" : val[2]*100,
        "prob_2" : val2[3]*100,
        "prob_3" : val3[3]*100,
        "prob_4" : val4[2]*100,
        "prob_5" : val5[3]*100,
        "v_prob" : fin_model[2]*100
    }
        return render_template('result.html',data=data, data1=prob, data2=accuracy)
    elif((val[0] == 1) and (val2[0] == 0) and (val3[0] == 0) and (val4[0] == 0) and (val5[0] == 1) and (fin_model[0] == 1)):
        prob = {
        "prob_1" : val[2]*100,
        "prob_2" : val2[3]*100,
        "prob_3" : val3[3]*100,
        "prob_4" : val4[3]*100,
        "prob_5" : val5[2]*100,
        "v_prob" : fin_model[1]*100
    }
        return render_template('result.html',data=data, data1=prob, data2=accuracy)
    elif((val[0] == 1) and (val2[0] == 0) and (val3[0] == 0) and (val4[0] == 0) and (val5[0] == 1) and (fin_model[0] == 0)):
        prob = {
        "prob_1" : val[2]*100,
        "prob_2" : val2[3]*100,
        "prob_3" : val3[3]*100,
        "prob_4" : val4[3]*100,
        "prob_5" : val5[2]*100,
        "v_prob" : fin_model[2]*100
    }
        return render_template('result.html',data=data, data1=prob, data2=accuracy)
    elif((val[0] == 1) and (val2[0] == 0) and (val3[0] == 0) and (val4[0] == 0) and (val5[0] == 0) and (fin_model[0] == 1)):
        prob = {
        "prob_1" : val[2]*100,
        "prob_2" : val2[3]*100,
        "prob_3" : val3[3]*100,
        "prob_4" : val4[3]*100,
        "prob_5" : val5[3]*100,
        "v_prob" : fin_model[1]*100
    }
        return render_template('result.html',data=data, data1=prob, data2=accuracy)
    elif((val[0] == 1) and (val2[0] == 0) and (val3[0] == 0) and (val4[0] == 0) and (val5[0] == 0) and (fin_model[0] == 0)):
        prob = {
        "prob_1" : val[2]*100,
        "prob_2" : val2[3]*100,
        "prob_3" : val3[3]*100,
        "prob_4" : val4[3]*100,
        "prob_5" : val5[3]*100,
        "v_prob" : fin_model[2]*100
    }
        return render_template('result.html',data=data, data1=prob, data2=accuracy)
    elif((val[0] == 0) and (val2[0] == 1) and (val3[0] == 1) and (val4[0] == 1) and (val5[0] == 1) and (fin_model[0] == 1)):
        prob = {
        "prob_1" : val[3]*100,
        "prob_2" : val2[2]*100,
        "prob_3" : val3[2]*100,
        "prob_4" : val4[2]*100,
        "prob_5" : val5[2]*100,
        "v_prob" : fin_model[1]*100
    }
        return render_template('result.html',data=data, data1=prob, data2=accuracy)
    elif((val[0] == 0) and (val2[0] == 1) and (val3[0] == 1) and (val4[0] == 1) and (val5[0] == 1) and (fin_model[0] == 0)):
        prob = {
        "prob_1" : val[3]*100,
        "prob_2" : val2[2]*100,
        "prob_3" : val3[2]*100,
        "prob_4" : val4[2]*100,
        "prob_5" : val5[2]*100,
        "v_prob" : fin_model[2]*100
    }
        return render_template('result.html',data=data, data1=prob, data2=accuracy)
    elif((val[0] == 0) and (val2[0] == 1) and (val3[0] == 1) and (val4[0] == 1) and (val5[0] == 0) and (fin_model[0] == 1)):
        prob = {
        "prob_1" : val[3]*100,
        "prob_2" : val2[2]*100,
        "prob_3" : val3[2]*100,
        "prob_4" : val4[2]*100,
        "prob_5" : val5[3]*100,
        "v_prob" : fin_model[1]*100
    }
        return render_template('result.html',data=data, data1=prob, data2=accuracy)
    elif((val[0] == 0) and (val2[0] == 1) and (val3[0] == 1) and (val4[0] == 1) and (val5[0] == 0) and (fin_model[0] == 0)):
        prob = {
        "prob_1" : val[3]*100,
        "prob_2" : val2[2]*100,
        "prob_3" : val3[2]*100,
        "prob_4" : val4[2]*100,
        "prob_5" : val5[3]*100,
        "v_prob" : fin_model[2]*100
    }
        return render_template('result.html',data=data, data1=prob, data2=accuracy)
    elif((val[0] == 0) and (val2[0] == 1) and (val3[0] == 1) and (val4[0] == 0) and (val5[0] == 1) and (fin_model[0] == 1)):
        prob = {
        "prob_1" : val[3]*100,
        "prob_2" : val2[2]*100,
        "prob_3" : val3[2]*100,
        "prob_4" : val4[3]*100,
        "prob_5" : val5[2]*100,
        "v_prob" : fin_model[1]*100
    }
        return render_template('result.html',data=data, data1=prob, data2=accuracy)
    elif((val[0] == 0) and (val2[0] == 1) and (val3[0] == 1) and (val4[0] == 0) and (val5[0] == 1) and (fin_model[0] == 0)):
        prob = {
        "prob_1" : val[3]*100,
        "prob_2" : val2[2]*100,
        "prob_3" : val3[2]*100,
        "prob_4" : val4[3]*100,
        "prob_5" : val5[2]*100,
        "v_prob" : fin_model[2]*100
    }
        return render_template('result.html',data=data, data1=prob, data2=accuracy)
    elif((val[0] == 0) and (val2[0] == 1) and (val3[0] == 1) and (val4[0] == 0) and (val5[0] == 0) and (fin_model[0] == 1)):
        prob = {
        "prob_1" : val[3]*100,
        "prob_2" : val2[2]*100,
        "prob_3" : val3[2]*100,
        "prob_4" : val4[3]*100,
        "prob_5" : val5[3]*100,
        "v_prob" : fin_model[1]*100
    }
        return render_template('result.html',data=data, data1=prob, data2=accuracy)
    elif((val[0] == 0) and (val2[0] == 1) and (val3[0] == 1) and (val4[0] == 0) and (val5[0] == 0) and (fin_model[0] == 0)):
        prob = {
        "prob_1" : val[3]*100,
        "prob_2" : val2[2]*100,
        "prob_3" : val3[2]*100,
        "prob_4" : val4[3]*100,
        "prob_5" : val5[3]*100,
        "v_prob" : fin_model[2]*100
    }
        return render_template('result.html',data=data, data1=prob, data2=accuracy)
    elif((val[0] == 0) and (val2[0] == 1) and (val3[0] == 0) and (val4[0] == 1) and (val5[0] == 1) and (fin_model[0] == 1)):
        prob = {
        "prob_1" : val[3]*100,
        "prob_2" : val2[2]*100,
        "prob_3" : val3[3]*100,
        "prob_4" : val4[2]*100,
        "prob_5" : val5[2]*100,
        "v_prob" : fin_model[1]*100
    }
        return render_template('result.html',data=data, data1=prob, data2=accuracy)
    elif((val[0] == 0) and (val2[0] == 1) and (val3[0] == 0) and (val4[0] == 1) and (val5[0] == 1) and (fin_model[0] == 0)):
        prob = {
        "prob_1" : val[3]*100,
        "prob_2" : val2[2]*100,
        "prob_3" : val3[3]*100,
        "prob_4" : val4[2]*100,
        "prob_5" : val5[2]*100,
        "v_prob" : fin_model[2]*100
    }
        return render_template('result.html',data=data, data1=prob, data2=accuracy)
    elif((val[0] == 0) and (val2[0] == 1) and (val3[0] == 0) and (val4[0] == 1) and (val5[0] == 0) and (fin_model[0] == 1)):
        prob = {
        "prob_1" : val[3]*100,
        "prob_2" : val2[2]*100,
        "prob_3" : val3[3]*100,
        "prob_4" : val4[2]*100,
        "prob_5" : val5[3]*100,
        "v_prob" : fin_model[1]*100
    }
        return render_template('result.html',data=data, data1=prob, data2=accuracy)
    elif((val[0] == 0) and (val2[0] == 1) and (val3[0] == 0) and (val4[0] == 1) and (val5[0] == 0) and (fin_model[0] == 0)):
        prob = {
        "prob_1" : val[3]*100,
        "prob_2" : val2[2]*100,
        "prob_3" : val3[3]*100,
        "prob_4" : val4[2]*100,
        "prob_5" : val5[3]*100,
        "v_prob" : fin_model[2]*100
    }
        return render_template('result.html',data=data, data1=prob, data2=accuracy)
    elif((val[0] == 0) and (val2[0] == 1) and (val3[0] == 0) and (val4[0] == 0) and (val5[0] == 1) and (fin_model[0] == 1)):
        prob = {
        "prob_1" : val[3]*100,
        "prob_2" : val2[2]*100,
        "prob_3" : val3[3]*100,
        "prob_4" : val4[3]*100,
        "prob_5" : val5[2]*100,
        "v_prob" : fin_model[1]*100
    }
        return render_template('result.html',data=data, data1=prob, data2=accuracy)
    elif((val[0] == 0) and (val2[0] == 1) and (val3[0] == 0) and (val4[0] == 0) and (val5[0] == 1) and (fin_model[0] == 0)):
        prob = {
        "prob_1" : val[3]*100,
        "prob_2" : val2[2]*100,
        "prob_3" : val3[3]*100,
        "prob_4" : val4[3]*100,
        "prob_5" : val5[2]*100,
        "v_prob" : fin_model[2]*100
    }   
        return render_template('result.html',data=data, data1=prob, data2=accuracy)
    elif((val[0] == 0) and (val2[0] == 1) and (val3[0] == 0) and (val4[0] == 0) and (val5[0] == 0) and (fin_model[0] == 1)):
        prob = {
        "prob_1" : val[3]*100,
        "prob_2" : val2[2]*100,
        "prob_3" : val3[3]*100,
        "prob_4" : val4[3]*100,
        "prob_5" : val5[3]*100,
        "v_prob" : fin_model[1]*100
    }
        return render_template('result.html',data=data, data1=prob, data2=accuracy)
    elif((val[0] == 0) and (val2[0] == 1) and (val3[0] == 0) and (val4[0] == 0) and (val5[0] == 0) and (fin_model[0] == 0)):
        prob = {
        "prob_1" : val[3]*100,
        "prob_2" : val2[2]*100,
        "prob_3" : val3[3]*100,
        "prob_4" : val4[3]*100,
        "prob_5" : val5[3]*100,
        "v_prob" : fin_model[2]*100
    }
        return render_template('result.html',data=data, data1=prob, data2=accuracy)
    elif((val[0] == 0) and (val2[0] == 0) and (val3[0] == 1) and (val4[0] == 1) and (val5[0] == 1) and (fin_model[0] == 1)):
        prob = {
        "prob_1" : val[3]*100,
        "prob_2" : val2[3]*100,
        "prob_3" : val3[2]*100,
        "prob_4" : val4[2]*100,
        "prob_5" : val5[2]*100,
        "v_prob" : fin_model[1]*100
    }
        return render_template('result.html',data=data, data1=prob, data2=accuracy)
    elif((val[0] == 0) and (val2[0] == 0) and (val3[0] == 1) and (val4[0] == 1) and (val5[0] == 1) and (fin_model[0] == 0)):
        prob = {
        "prob_1" : val[3]*100,
        "prob_2" : val2[3]*100,
        "prob_3" : val3[2]*100,
        "prob_4" : val4[2]*100,
        "prob_5" : val5[2]*100,
        "v_prob" : fin_model[2]*100
    }
        return render_template('result.html',data=data, data1=prob, data2=accuracy)
    elif((val[0] == 0) and (val2[0] == 0) and (val3[0] == 1) and (val4[0] == 1) and (val5[0] == 0) and (fin_model[0] == 1)):
        prob = {
        "prob_1" : val[3]*100,
        "prob_2" : val2[3]*100,
        "prob_3" : val3[2]*100,
        "prob_4" : val4[2]*100,
        "prob_5" : val5[3]*100,
        "v_prob" : fin_model[1]*100
    }
        return render_template('result.html',data=data, data1=prob, data2=accuracy)
    elif((val[0] == 0) and (val2[0] == 0) and (val3[0] == 1) and (val4[0] == 1) and (val5[0] == 0) and (fin_model[0] == 0)):
        prob = {
        "prob_1" : val[3]*100,
        "prob_2" : val2[3]*100,
        "prob_3" : val3[2]*100,
        "prob_4" : val4[2]*100,
        "prob_5" : val5[3]*100,
        "v_prob" : fin_model[2]*100
    }
        return render_template('result.html',data=data, data1=prob, data2=accuracy)
    elif((val[0] == 0) and (val2[0] == 0) and (val3[0] == 1) and (val4[0] == 0) and (val5[0] == 1) and (fin_model[0] == 1)):
        prob = {
        "prob_1" : val[3]*100,
        "prob_2" : val2[3]*100,
        "prob_3" : val3[2]*100,
        "prob_4" : val4[3]*100,
        "prob_5" : val5[2]*100,
        "v_prob" : fin_model[1]*100
    }
        return render_template('result.html',data=data, data1=prob, data2=accuracy)
    elif((val[0] == 0) and (val2[0] == 0) and (val3[0] == 1) and (val4[0] == 0) and (val5[0] == 1) and (fin_model[0] == 0)):
        prob = {
        "prob_1" : val[3]*100,
        "prob_2" : val2[3]*100,
        "prob_3" : val3[2]*100,
        "prob_4" : val4[3]*100,
        "prob_5" : val5[2]*100,
        "v_prob" : fin_model[2]*100
    }
        return render_template('result.html',data=data, data1=prob, data2=accuracy)
    elif((val[0] == 0) and (val2[0] == 0) and (val3[0] == 1) and (val4[0] == 0) and (val5[0] == 0) and (fin_model[0] == 1)):
        prob = {
        "prob_1" : val[3]*100,
        "prob_2" : val2[3]*100,
        "prob_3" : val3[2]*100,
        "prob_4" : val4[3]*100,
        "prob_5" : val5[3]*100,
        "v_prob" : fin_model[1]*100
    }
        return render_template('result.html',data=data, data1=prob, data2=accuracy)
    elif((val[0] == 0) and (val2[0] == 0) and (val3[0] == 1) and (val4[0] == 0) and (val5[0] == 0) and (fin_model[0] == 0)):
        prob = {
        "prob_1" : val[3]*100,
        "prob_2" : val2[3]*100,
        "prob_3" : val3[2]*100,
        "prob_4" : val4[3]*100,
        "prob_5" : val5[3]*100,
        "v_prob" : fin_model[2]*100
    }
        return render_template('result.html',data=data, data1=prob, data2=accuracy)
    elif((val[0] == 0) and (val2[0] == 0) and (val3[0] == 0) and (val4[0] == 1) and (val5[0] == 1) and (fin_model[0] == 1)):
        prob = {
        "prob_1" : val[3]*100,
        "prob_2" : val2[3]*100,
        "prob_3" : val3[3]*100,
        "prob_4" : val4[2]*100,
        "prob_5" : val5[2]*100,
        "v_prob" : fin_model[1]*100
    }
        return render_template('result.html',data=data, data1=prob, data2=accuracy)
    elif((val[0] == 0) and (val2[0] == 0) and (val3[0] == 0) and (val4[0] == 1) and (val5[0] == 1) and (fin_model[0] == 0)):
        prob = {
        "prob_1" : val[3]*100,
        "prob_2" : val2[3]*100,
        "prob_3" : val3[3]*100,
        "prob_4" : val4[2]*100,
        "prob_5" : val5[2]*100,
        "v_prob" : fin_model[2]*100
    }
        return render_template('result.html',data=data, data1=prob, data2=accuracy)
    elif((val[0] == 0) and (val2[0] == 0) and (val3[0] == 0) and (val4[0] == 1) and (val5[0] == 0) and (fin_model[0] == 1)):
        prob = {
        "prob_1" : val[3]*100,
        "prob_2" : val2[3]*100,
        "prob_3" : val3[3]*100,
        "prob_4" : val4[2]*100,
        "prob_5" : val5[3]*100,
        "v_prob" : fin_model[1]*100
    }
        return render_template('result.html',data=data, data1=prob, data2=accuracy)
    elif((val[0] == 0) and (val2[0] == 0) and (val3[0] == 0) and (val4[0] == 1) and (val5[0] == 0) and (fin_model[0] == 0)):
        prob = {
        "prob_1" : val[3]*100,
        "prob_2" : val2[3]*100,
        "prob_3" : val3[3]*100,
        "prob_4" : val4[2]*100,
        "prob_5" : val5[3]*100,
        "v_prob" : fin_model[2]*100
    }
        return render_template('result.html',data=data, data1=prob, data2=accuracy)
    elif((val[0] == 0) and (val2[0] == 0) and (val3[0] == 0) and (val4[0] == 0) and (val5[0] == 1) and (fin_model[0] == 1)):
        prob = {
        "prob_1" : val[3]*100,
        "prob_2" : val2[3]*100,
        "prob_3" : val3[3]*100,
        "prob_4" : val4[3]*100,
        "prob_5" : val5[2]*100,
        "v_prob" : fin_model[1]*100
    }
        return render_template('result.html',data=data, data1=prob, data2=accuracy) 
    elif((val[0] == 0) and (val2[0] == 0) and (val3[0] == 0) and (val4[0] == 0) and (val5[0] == 1) and (fin_model[0] == 0)):
        prob = {
        "prob_1" : val[3]*100,
        "prob_2" : val2[3]*100,
        "prob_3" : val3[3]*100,
        "prob_4" : val4[3]*100,
        "prob_5" : val5[2]*100, 
        "v_prob" : fin_model[2]*100
    }
        return render_template('result.html',data=data, data1=prob, data2=accuracy)
    elif((val[0] == 0) and (val2[0] == 0) and (val3[0] == 0) and (val4[0] == 0) and (val5[0] == 0) and (fin_model[0] == 1)):
        prob = {
        "prob_1" : val[3]*100,
        "prob_2" : val2[3]*100,
        "prob_3" : val3[3]*100,
        "prob_4" : val4[3]*100,
        "prob_5" : val5[3]*100,
        "v_prob" : fin_model[1]*100
    }
        return render_template('result.html',data=data, data1=prob, data2=accuracy)
    elif((val[0] == 0) and (val2[0] == 0) and (val3[0] == 0) and (val4[0] == 0) and (val5[0] == 0) and (fin_model[0] == 0)):
        prob = {
        "prob_1" : val[3]*100,
        "prob_2" : val2[3]*100,
        "prob_3" : val3[3]*100,
        "prob_4" : val4[3]*100,
        "prob_5" : val5[3]*100,
        "v_prob" : fin_model[2]*100
    }
        return render_template('result.html',data=data, data1=prob, data2=accuracy)

    return render_template('result.html',data=data) 
    

@app.route('/logout')
def logout():
    isLoggedin = False
    return render_template('index.html', isLoggedin=isLoggedin)

app.run(debug=True) 
