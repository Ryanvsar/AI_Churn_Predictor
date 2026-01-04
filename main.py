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
from saveToDb2 import save_to_db2
from collectAndPredict import runModel


# * * Run usageSimulation(Small - 500 users) * *
# _, churn_user_ids = save_to_db()
# runModel(churn_user_ids)

# * * Run usageSimulation2(Medium - 5000 users) * *
_, churn_user_ids, _ = save_to_db2()
runModel(churn_user_ids)
