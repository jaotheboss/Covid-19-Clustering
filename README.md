# Covid-19-Clustering
Analysis and clustering of some Covid-19 datasets that I have found and scraped from online


## Objective:
To simply explore the Covid-19 dataset and implement a clustering algorithm to further dig out interesting insights.

## Datasets:

  1. Covid-19 Dataset (kaggle)
  2. Education Index (.csv)
  3. Healthcare Expenditure (.csv)
  4. Social Progress Index (pdf scraping)
  5. Flight Traffic (pdf scraping)
  6. Population Density (web scraping)
  7. Economic Income/GDP per Capita (web scraping)

## Skills Used:

  1. Data Wrangling (Preprocessing and transformation of the dataset)
  2. Web Scraping (Collection of the data)
  3. PDF Scraping (Collection of the data)
  4. K-Means Clustering (For EDA)
  5. Hierarchal Clustering (For EDA)
  6. Visualisations (For EDA)

## Process and Findings:
My first intuition is to compare the fatality and case count of each country with reference to other variables. 
Hence, there was a need to search for various indicators. 
Through my search, I have decided to settle for 6 different indicators:
  1. Education
  
      - Using the [Education Index (EI)](https://en.wikipedia.org/wiki/Education_Index) which is a part of the Human Development Index (HDI) that is measured by combining average adult years of schooling with expected years of schooling for children, each receiving 50% weighting
  
  2. Healthcare Spending
  
      - The total amount of money spent on healthcare as a whole (investments in research, procurement of tools, etc)
  
  3. Country's Income
  
      - Measured by Gross Domestic Product per capita
  
  4. Flight Traffic and Movement
  
      - The amount of human transactions made via Flight. ie people flying in or out and those in transit and making their way into the country
  
  5. Population Density
  
      - The number of people per square kilometer
  
  6. Social Progression
      
      - Measured using the [Social Progress Index (SPI)](https://en.wikipedia.org/wiki/Social_Progress_Index) which measures the extent to which countries provide for the social and environmental needs of their citizens
  
This process took up the bulk of my time as I was new to scraping data off PDFs. Something I considered when I was looking for data was how up to date they were. The main Covid-19 dataset listed the final count of Confirmed Cases and Fatalities at the end of **April 2020**, and therefore I had to make sure the data collected were still relevant. 


One thing I noticed throughout all the datasets is that the **namings of the countries were inconsistent**.
Not only that but there are some datasets that label countries different or do not even include some countries.
For example, if you did not know, there is a huge debate on whether Taiwan and Hong Kong are [considered part of China](https://www.scmp.com/news/china/society/article/2164126/why-are-taiwan-and-hong-kong-separate-china-chinese-raise-ruckus). 
Although they may be part of China in legal terms, they are not in almost every other way. 
In order to make this consistent for me, I have converted Hong Kong and Taiwan into their individual countries for this project.

However, there are still certain countries that are missing and therefore have to admit that we won't have all the countries in the resultant dataset.
After merging all the data together, we were only left with slightly under 100 countries. Albeit all of them had full data for each variable. 

After completing the dataset, I proceeded with clustering the data to find out if it was possible to group countries together based on the variables I have tagged them with. Since I will be using clustering, it is imperative for the model that the numerical distance between the points are consistent. For example, values vary widely for `Population Density` and `GDP`. One of them being able to reach past tens of thousands while the other can go as low as a single digit. Therefore, to reconcile this, I [standardized](https://www.statisticshowto.com/standardized-values-examples/) each of the variables. 

After standardizing all the variables, we can confirm that the values will vary and centralize at the 0 point on the axis. Thus, making it easier for the clustering algorithm to detect the actual closeness of the countries with regard to those variables. 

To find out how many clusters to group the countries in, I first do a hierarchal clustering and to iteratively run a K-Means test with clusters 2 to 9, to determine at which number of clusters would yield the smallest intra-variance cluster, and I will also run a sillhouete plot to determine the data distribution within each cluster.


Below shows the first hierarchal plot and as I observe it, I would say that there are around 5 clusters (looking at around the 11 mark on the y-axis):
![Hierarchal Plot 1](https://github.com/jaotheboss/Covid-19-Clustering/blob/master/Visualisations/Hierarchal%20Plot_v1.png)

This is further substantiated when I plotted the various K-Means intra-variance graph, as can be seen below:
![K-Means Graph 1](https://github.com/jaotheboss/Covid-19-Clustering/blob/master/Visualisations/Elbow%20Plot_v1.png)

However, when looking at the sillhouete plot, show below, we notice that there are some data points that could have been placed in the wrong cluster:
![Sillhouete Plot 1](https://github.com/jaotheboss/Covid-19-Clustering/blob/master/Visualisations/Sillhouete%20Plot_v1.png)


I decide to dive into the dataset to find out more about this. Turns out I did find one thing that is **interesting** to note.

  • **Singapore** is the **only country** that is placed in cluster 3. Although there is not definitive reasoning behind this, I suspect that this owes to the fact that Singapore has a low fatality count given its other variables. I say this because there are other countries like [Qatar](https://www.google.com/search?q=qatar+covid+19&oq=qatar+cov&aqs=chrome.1.0l2j69i57j0l4.4547j0j7&sourceid=chrome&ie=UTF-8) that have low fatality counts with high case counts. However, what is different is the fact that Singapore has an extremely high density and that could be one of the odds in dataset. That given how dense Singapore is, there should be a higher fatality count. There are probably other ways to reconcile for this observation, as it is not easy to cluster countries together given how unique all of them are. 

I, therefore, continue the exploration without Singapore in the dataset. I proceed with plotting out the various graphs and plots I have done before on this newer dataset and these are the results. 

From what we can see below, it still seems that 4-5 clusters are the way to go:
![Hierarchal Plot 2](https://github.com/jaotheboss/Covid-19-Clustering/blob/master/Visualisations/Hierarchal%20Plot_v2.png)

Similar to the case above, the K-Means intra-variance graph substantiates having 4 clusters, as shown below:
![K-Means Graph 2](https://github.com/jaotheboss/Covid-19-Clustering/blob/master/Visualisations/Elbow%20Plot_v2.png)

We can also see that if there are 5 clusters, some clusters might not have enough statistical significance to be in their own cluster:
![Sillhouete Plot 2](https://github.com/jaotheboss/Covid-19-Clustering/blob/master/Visualisations/Sillhouete%20Plot%20with%205_v2.png)

Whereas, when we have 4 clusters, all of them have statistically enough countries within each cluster to be legitimate while only having to deal with a few countries that could possibly be placed in the wrong cluster:
![Sillhouete Plot 3](https://github.com/jaotheboss/Covid-19-Clustering/blob/master/Visualisations/Sillhouete%20Plot%20with%204_v2.png)

Now we can place the labels on the countries and plot out some distribution charts. Just for fun, I plotted out for scenarios of having both 4 and 5 clusters.

Plot for **4 clusters**:
![Facet Grid for 4](https://github.com/jaotheboss/Covid-19-Clustering/blob/master/Visualisations/Facetgrid%20Plot%20for%204.png)

Plot for **5 clusters**:
![Facet Grid for 5](https://github.com/jaotheboss/Covid-19-Clustering/blob/master/Visualisations/Facetgrid%20Plot%20for%205.png)

Since the clustering is done, I have decided to look into the statistical significance of each of the variables with regards to how they affect the Confirmed Cases and Fatality counts. For this scenario, I have ported my workings to another programming language: R. 

Using R, I was able to do an [analysis of variance (ANOVA)](https://www.statisticshowto.com/probability-and-statistics/hypothesis-testing/anova/) to determine how each of the variables would affect the variance of the model that was fitted with the given dataset. 

Here are my findings for how the variables have affected the Confirmed Cases counts:
![Confirmed Cases](https://github.com/jaotheboss/Covid-19-Clustering/blob/master/Visualisations/ANOVA%20for%20Cases.png)

And here are my findings for how the varialbes have affected the Fatality counts:
![Fatality](https://github.com/jaotheboss/Covid-19-Clustering/blob/master/Visualisations/ANOVA%20for%20Fatalities.png)

It can be seen in both cases at the Social Progress Index (SPI), Education Index (EI) and Flight Counts have a p-value of < 0.05 and therefore imply that these variables are statistically significant in affecting the Confirmed Cases and Fatality counts. 

However, what is even more interesting to me is the fact that GDP and Healthcare Expenditure has a p-value of around 0.9. This implies that those variables have no statistically relationship with the response variables (the Confirmed Cases and Fatality counts). Meaning to say that regardless of how much money you are earning or how much money you are pumping into the healthcare industry, the coronavirus situation would still spread as it did, statistically speaking. 

## Conclusion:

I think it would be more interesting to further explore this dataset using a more sociological and psychological perspective. 
Given that the indicators that have statistical significance to the cases are those that have ties with social aspects. Something interesting to explore as well would be the political variables of a country. Maybe how fond they are of their government, what kind of political philosophy a country is following, etc. 
