import pandas as pd
data = pd.read_csv('/Users/mayagonzalez/Desktop/Academic/ML_HW/athens_www2_weather.csv')
numRows = len(data.index) -1 # should be 358
print(numRows)