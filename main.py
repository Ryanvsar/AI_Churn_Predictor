"""
------------------------------------------------------
Author: Ryan Vandersar
Date (DD-MM-YYYY): 22-12-2025
------------------------------------------------------
Program Title: Churn Prediction AI
------------------------------------------------------
File Description: Main file to run code
------------------------------------------------------
"""
from saveToDb import save_to_db
from collectAndPredict import runModel

users, churn_user_ids = save_to_db()
runModel(churn_user_ids)

