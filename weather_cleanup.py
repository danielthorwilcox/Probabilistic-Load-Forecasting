import pandas as pd
import matplotlib.pyplot as plt
from functools import reduce

data = pd.read_csv('demand_generation\energy_dataset.csv')
weather = pd.read_csv('demand_generation\weather_features.csv')
weather = weather.drop_duplicates(subset=['dt_iso', 'city_name'])
cities = weather['city_name'].unique()
# print(cities)
city = []
for name in cities:
    city.append(weather.loc[weather['city_name'] == name])
cities[3] = 'Barcelona'

for i in range(len(cities)):
    city[i].columns= ['dt_iso', 'city_name_'+cities[i], 'temp_'+cities[i], 'temp_min_'+cities[i], 'temp_max_'+cities[i], 'pressure_'+cities[i], 'humitidy_'+cities[i], 'wind_speed_'+cities[i], 'wind_deg_'+cities[i], 'rain_1h_'+cities[i], 'rain_3h_'+cities[i], 'snow_3h_'+cities[i], 'clouds_all_'+cities[i], 'weather_id_'+cities[i], 'weather_main_'+cities[i], 'weather_description_'+cities[i], 'weather_icon_'+cities[i]]
weather_total = reduce(lambda  left,right: pd.merge(left,right,on=['dt_iso'], how='outer'), city)
for i in range(len(cities)):
    weather_total = weather_total.drop(columns=['city_name_'+cities[i]])
print(weather_total)
weather_total.to_csv("demand_generation/weather_clean.csv", index = False)