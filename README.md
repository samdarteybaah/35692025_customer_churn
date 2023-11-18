# 35692025_customer_churn
 This project is an AI model used to predict possible churn in the a telecommunication firm
 This AI model can be used to predict whether a customer will churn or not, which uses certain characteristics to determine this probability.
 It uses MLP to train, GridSearchCV to tune the model. These are the functionalities

 A model is trained, tuned and deployed with streamlit to build a web app

The web app takes in inputs from the necessary features: ['MonthlyCharges', 'TotalCharges', 'tenure', 'Contract', 'PaymentMethod_encoded', 'InternetService'] from the user.

It then predicts whether a customer would churn or not based on the answered features and the probability or likelihood of correctness

streamlit run app.py should be run to in app.py terminal 


 video demonstration of model in action: https://drive.google.com/file/d/1oL1QI52Ez9Ls-QF6aaR3TYEOb_KaAGCV/view?usp=sharing
 
 picture demonstration: https://drive.google.com/drive/folders/1rqtq0nQpI18CgLQr7jd-3oqNc3bBG0Bp?usp=sharing
