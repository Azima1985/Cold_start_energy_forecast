# Cold_start_energy_forecast
Energy consumption prediction - Cold start problem

https://www.drivendata.org/competitions/55/schneider-cold-start/

Power Laws: Cold Start Energy Forecasting
Building energy forecasting has gained momentum with the increase of building energy efficiency research and solution development. Indeed, forecasting the global energy consumption of a building can play a pivotal role in the operations of the building. It provides an initial check for facility managers and building automation systems to mark any discrepancy between expected and actual energy use. Accurate energy consumption forecasts are also used by facility managers, utility companies and building commissioning projects to implement energy-saving policies and optimize the operations of chillers, boilers and energy storage systems.

Usually, forecasting algorithms use historical information to compute their forecast. Most of the time, the bigger the historic dataset, the more accurate the forecast. This requirement presents a big challenge: how can we make accurate predictions for new buildings, which don't have a long consumption history?

The goal of this challenge is to build an algorithm which provides an accurate forecast from the very start of a building's instrumentation.

Available Data:
Datasets for 758 buildings including hourly energy consumption for 28 days (672 time steps).

Goal:
Predict energy consumption trends for buildings with little historical data available (1 day - 2 weeks of data).

Method:
Implement Reptile meta learning.
