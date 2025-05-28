Formula 1 Prediction The project shows in the UI the predictions of Monaco GP of Carlos Sainz, Alexander Albon, and Williams constructor.

Also you can see different stadistics that allow to me to have the predictions as for example:

Evolution of the points in the time of the drivers and Williams constructor.

Qualifyng
Wins
Lap Times Outliers
Pit Stops Outliers
Turns
Performance
Weather
Precision
Accuracy
Recovery
Pit Stops
Lap Times
The project is trained with 6 different models, 3 (RandomForestClassifier, SVC, KNeigborsClassifier) for driver predictions and 3 (RandomForestRegressor, SVR, KneighborsRegressor) for Williams constructor points.

How to Run? Install the necessary libraries using:

pip install -r requirements.txt run app.py

it executes two files app.py and templates/index.html
