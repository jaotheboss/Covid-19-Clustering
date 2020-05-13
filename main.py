#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 22:59:02 2020

@author: jaoming

To do:
       - Get education data

Total data used:
- Density
- Flights
- Spending
- Earning
- SPI
- Education?
"""

import os
os.chdir('/Users/jaoming/Documents/Active Projects/Coronavirus Prediction (Kaggle)')

import pandas as pd
import numpy as np
import datetime
import requests
from bs4 import BeautifulSoup
import re
# from PyPDF2 import PdfFileReader is for manipulating the pdf (rotating, merging and what not)
import camelot

data = pd.read_csv('covidcases.csv')

# Cleaning of the data
unique_provinces = list(filter(lambda x: pd.notna(x), data['Province_State'].unique()))
unique_provinces.sort()

## briefly looking at the provinces we notice that Hong Kong is not considered a country
## shifting Hong Kong into a country
data.loc[data['Province_State'] == 'Hong Kong', 'Country_Region'] = ['Hong Kong']*len(data.loc[data['Province_State'] == 'Hong Kong', 'Country_Region'].index)

# drop province and id columns
data.drop(['Id', 'Province_State'], axis = 1, inplace = True)

"""# convert string date to actualy datetime object
data['Date'] = [datetime.datetime.strptime(x, '%Y-%m-%d') for x in data['Date']]

# we want to cut group the dates together
len(data['Date'].unique()) # 97 days altogether
## we'll group them by the week
countries = list(data['Country_Region'].unique())
timeseries_data = pd.DataFrame()
for country in countries:
       country_data = data.loc[data['Country_Region'] == country, ['Date', 'ConfirmedCases', 'Fatalities']]
       country_data = country_data.groupby(pd.Grouper(key = 'Date', freq = '7D')).mean().reset_index()
       country_data['Country'] = [country]*len(country_data.index)
       country_data = country_data.reindex(columns = ['Country', 'Date', 'ConfirmedCases', 'Fatalities'])
       
       timseries_data = timeseries_data.append(country_data)

timeseries_data = timeseries_data.melt(id_vars = ['Country', 'Date'], 
                               value_vars = ['ConfirmedCases', 'Fatalities'], 
                               var_name = 'Status', 
                               value_name = 'Value')

### Example of using time series data
sg_data = timeseries_data.loc[timeseries_data['Country'] == 'Singapore', :]

import seaborn as sns

plot = sns.relplot(y = 'Value', x = 'Date', data = sg_data, col = 'Status', markers = True)
plot.set(xlim = [sg_data['Date'].unique()[0], sg_data['Date'].unique()[-1]])
for ax in plot.axes.flat:
       for label in ax.get_xticklabels():
              label.set_rotation(30)"""
              
# now we only take the latest confirmed case values
current_data = data.groupby('Country_Region').max()
current_data.reset_index(inplace = True)
current_data.drop(['Date'], axis = 1, inplace = True)

# rectifying data
current_data.loc[91, 'Country_Region'] = 'South Korea'
current_data.loc[164, 'Country_Region'] = 'Taiwan'

############### DATA SELECTION AND PREPROCESSING ###############
## We'll find various variables to cluster upon

## Economic
### GDP per capita
gdp_html = requests.get('https://www.worldometers.info/gdp/gdp-per-capita/').content
parser2 = BeautifulSoup(gdp_html, features = 'html.parser')
country_gdp = []
for row in parser2.findAll('tr')[1:]:
       if re.sub('[$,]', '', row.contents[5].contents[0][1:]) == 'N.A. ':
              break
       country, gdp = row.contents[3].contents[0].contents[0], int(re.sub('[$,]', '', row.contents[5].contents[0][1:]))
       country_gdp.append([country, gdp])
country_gdp = pd.DataFrame(country_gdp, columns = ['Country', 'GDP'])

### Healthcare Spending per capita
country_healthcare_exp = pd.read_excel('Healthcare Expenditure Per Capita.xlsx')
country_healthcare_exp.columns = ['Country', 'Healthcare Spending']

## Geographical
### Density (per km)
density_html = requests.get('https://worldpopulationreview.com/countries/countries-by-density/').content
parser = BeautifulSoup(density_html, features = 'html.parser')
country_density = []
for row in parser.findAll('tr')[1:]:
       country, density = row.contents[1].contents[0].contents[0], int(re.sub(',', '', row.contents[2].contents[0][:-4]))
       country_density.append([country, density])
country_density = pd.DataFrame(country_density, columns = ['Country', 'Density'])

### Flights Traffic
"""def extract_information(pdf_file_name):
       with open(pdf_file_name, 'rb') as pdf:
              pdf = PdfFileReader(pdf)
              page = pdf.getPage(31)
              text = page.extractText() # only useful for scraping text
       return text"""
flight_raw = camelot.read_pdf('Flight Movements.pdf', pages = '33-35', flavor = 'stream')
country_flight = pd.DataFrame()
for i in range(3):
       temp = flight_raw[i].df
       if i == 0:
              temp.drop([1, 3], axis = 1, inplace = True)
              temp.drop([0, 1, 2, 3, 4], axis = 0, inplace = True)
              temp.columns = ['Country', 'Flight Count']
              temp = temp.loc[temp['Flight Count'] != '', :]
              country_flight = country_flight.append(temp)
       else:
              temp.drop(2, axis = 1, inplace = True)
              temp.drop([0, 1, 2], axis = 0, inplace = True)
              temp.columns = ['Country', 'Flight Count']
              temp = temp.loc[temp['Country'] != '', :]
              country_flight = country_flight.append(temp)
country_flight = country_flight.loc[country_flight['Flight Count'] != '', :]
country_flight['Flight Count'] = country_flight['Flight Count'].apply(lambda x: int(re.sub(',', '', x)))
country_flight.sort_values(by = 'Flight Count', inplace = True, ascending = False)
country_flight.reset_index(inplace = True, drop = True)

# rectifying names
country_flight.loc[0, 'Country'] = 'US'
country_flight.loc[1, 'Country'] = 'China'
country_flight.loc[11, 'Country'] = 'South Korea'
country_flight.loc[19, 'Country'] = 'Taiwan'
country_flight.loc[22, 'Country'] = 'Hong Kong'
country_flight.loc[46, 'Country'] = 'Iran'
country_flight.loc[84, 'Country'] = 'Bolivia'


## Social
### Social Progress Index (SPI)
"""
SPI measures the extent to which countries provide for the social and environmental needs of their citizens.
"""
spi_raw = camelot.read_pdf('Social Progress Index.pdf', flavor = 'stream')[0].df
country_spi = pd.DataFrame()
for i in range(4, 17, 4):
       temp = spi_raw.iloc[:, (i - 4):i]
       temp.columns = temp.iloc[0, :]
       temp.drop(0, inplace = True)
       temp.drop(['', 'RANKING'], axis = 1, inplace = True)
       country_spi = country_spi.append(temp)
country_spi.reset_index(inplace = True, drop = True)
country_spi = country_spi.loc[country_spi['COUNTRY'] != '', :]
country_spi.columns = ['Country', 'SPI']

### Education Index (EI)
country_ei = pd.read_csv('Education Index.csv')
country_ei = country_ei.iloc[:189, :]
country_ei['EI'] = [float(i) for i in country_ei['EI']]
country_ei.sort_values(by = 'EI', inplace = True, ascending = False)
country_ei.reset_index(inplace = True, drop = True)

# rectifying names
country_ei.loc[11, 'Country'] = 'US'
country_ei.loc[23, 'Country'] = 'South Korea'
country_ei.loc[24, 'Country'] = 'Hong Kong'
country_ei.loc[32, 'Country'] = 'Russia'
country_ei.loc[63, 'Country'] = 'Iran'
country_ei.loc[77, 'Country'] = 'Moldova'
country_ei.loc[83, 'Country'] = 'Venezuela'
country_ei.loc[90, 'Country'] = 'Bolivia'
country_ei.loc[123, 'Country'] = 'Micronesia'
country_ei.loc[136, 'Country'] = 'Eswatini'
country_ei.loc[149, 'Country'] = 'Congo'
country_ei.loc[165, 'Country'] = 'Tanzania'

############### DATA TRANSFORMATION ###############
current_data.columns = ['Country', 'Cases', 'Fatalities']

merged = pd.merge(left = current_data, right = country_ei, on = 'Country', how = 'left')
merged = pd.merge(left = merged, right = country_spi, on = 'Country', how = 'left')
merged = pd.merge(left = merged, right = country_flight, on = 'Country', how = 'left')
merged = pd.merge(left = merged, right = country_density, on = 'Country', how = 'left')
merged = pd.merge(left = merged, right = country_gdp, on = 'Country', how = 'left')
merged = pd.merge(left = merged, right = country_healthcare_exp, on = 'Country', how = 'left')
merged.dropna(inplace = True)
merged.reset_index(inplace = True, drop = True)

result_df = merged.copy()

from sklearn.preprocessing import StandardScaler
standardizer = StandardScaler()
for label in merged.columns[1:]:
       merged[label] = standardizer.fit_transform(np.array(merged[label]).reshape(-1, 1)).reshape(1, -1)[0]

############### DATA MINING ###############
from scipy.cluster.hierarchy import dendrogram
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer
from sklearn.cluster import AgglomerativeClustering, KMeans

## visualizing with heirarchal clustering to determine how many clusters there should be
hclust = AgglomerativeClustering(n_clusters = None, distance_threshold = 0)
hclust.fit(merged.loc[:, merged.columns != 'Country'])

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

plot_dendrogram(hclust)

## getting labels
hclust = AgglomerativeClustering(n_clusters = 5, distance_threshold = None)
hclust_labels = hclust.fit_predict(merged.loc[:, merged.columns != 'Country'])

## visualizing with k means elbow and silhouette plots to determine how many clusters there should be
kmeans = KMeans()
visualizer = KElbowVisualizer(kmeans, k = (2, 10))
visualizer.fit(merged.loc[:, merged.columns != 'Country'])

kmeans = KMeans(n_clusters = 10)
visualizer = SilhouetteVisualizer(kmeans, colors = 'yellowbrick')
visualizer.fit(merged.loc[:, merged.columns != 'Country'])

## getting labels
kmeans = KMeans(n_clusters = 5, n_jobs = -1)
kmean_labels = kmeans.fit_predict(merged.loc[:, merged.columns != 'Country'])

############### INFERENCING AND EVALUATION ###############
import seaborn as sns

result_df['Cluster'] = hclust_labels
result_df.groupby('Cluster').mean()

sns.catplot(x = 'Cluster', y = 'Fatalities', data = result_df)
# we observe that Singapore is an anomaly
# when looking at the data to reconcile this, we try to spot the variation in the patterns like
# 1. 3rd most in number of cases, lowest in number of fatalities, given that it has relatively high fligh count, high density and high everything else. 
# whether this is good or not, it's not for me to say. but all i can tell from the data is that Singapore is an anomaly and can't be fitted into 4 main other clusters
# meaning they do not confirm to the patterns of the rest of the countries within those other clusters.

# let's remove Singapore and continue the analysis
u_df = result_df.loc[result_df['Country'] != 'Singapore', :]
u_df.drop('Cluster', axis = 1, inplace = True)

## refitting the data
ufit_df = u_df.copy()
for label in ufit_df.columns[1:]:
       ufit_df[label] = standardizer.fit_transform(np.array(ufit_df[label]).reshape(-1, 1)).reshape(1, -1)[0]

hclust = AgglomerativeClustering(n_clusters = None, distance_threshold = 0)
hclust.fit(ufit_df.loc[:, ufit_df.columns != 'Country'])
plot_dendrogram(hclust)

hclust = AgglomerativeClustering(n_clusters = 5, distance_threshold = None)
hclust_labels = hclust.fit_predict(ufit_df.loc[:, ufit_df.columns != 'Country'])

## do the same with the kmeans data
kmeans = KMeans()
visualizer = KElbowVisualizer(kmeans, k = (2, 10))
visualizer.fit(ufit_df.loc[:, ufit_df.columns != 'Country'])

kmeans = KMeans(n_clusters = 5)
visualizer = SilhouetteVisualizer(kmeans, colors = 'yellowbrick')
visualizer.fit(ufit_df.loc[:, ufit_df.columns != 'Country'])

# Infering Again
u_df['Cluster'] = hclust_labels
u_df.groupby('Cluster').mean()

uscaled_df = ufit_df.copy()
uscaled_df['Cluster'] = hclust_labels

sns.catplot(x = 'Cluster', y = 'Fatalities', data = uscaled_df.query('Cluster != 0'), kind = 'box')

## Transforming Data for Plot
utransform_df = u_df.melt(id_vars = ['Country', 'Cluster'], var_name = 'Indicator', value_name = 'Value')
utransform_df['Value'] = [float(i) for i in utransform_df['Value']]

plot = sns.FacetGrid(utransform_df, col = 'Indicator', col_wrap = 4, sharey = False)
plot.map(sns.barplot, 'Cluster', 'Value')

# uscaled_df.to_csv('data.csv', index = False)

