#1. Filter the data to include only weekdays (Monday to Friday) and
#plot a line graph showing the pedestrian counts for each day of the
#week.
import pandas as pd
import matplotlib.pyplot as plt

# Read the dataset
url = "https://data.cityofnewyork.us/api/views/6fi9-q3ta/rows.csv?accessType=DOWNLOAD"
df = pd.read_csv(url)

df['hour_begining'] = pd.to_datetime(df['hour_begining'])

weekdays_df = df[df['day'].isin(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'])]
weekday_pedestrian_counts = weekdays_df.groupby('day')['Pedestrians'].sum()
plt.figure(figsize=(10,6))
weekday_pedestrian_counts.plot(kind='line', marker='o')
plt.title('Pedestrian Counts for Weekdays (Monday to Friday)')
plt.xlabel('Day of the Week')
plt.ylabel('Total Pedestrian Count')
plt.grid(True)
plt.show()

#----- Write your code below this after running above above code-----------

#2. Track pedestrian counts on the Brooklyn Bridge for the year 2019
#and analyze how different weather conditions influence pedestrian
#activity in that year. Sort the pedestrian count data by weather
#summary to identify any correlations( with a correlation matrix)
#between weather patterns and pedestrian counts for the selected year.

#-This question requires you to show the relationship between a
#numerical feature(Pedestrians) and a non-numerical feature(Weather
#Summary). In such instances we use Encoding. Each weather condition
#can be encoded as numbers( 0,1,2..). This technique is called One-hot
#encoding.

#-Correlation matrices may not always be the most suitable
#visualization method for relationships involving categorical
#datapoints, nonetheless this was given as a question to help you
#understand the concept better.

df_encoded = pd.get_dummies(df, columns=['Weather_Summary'], prefix='Weather')
correlation_matrix = df_encoded.corr()
print(correlation_matrix)

#3. Implement a custom function to categorize time of day into morning,
#afternoon, evening, and night, and create a new column in the
#DataFrame to store these categories. Use this new column to analyze
#pedestrian activity patterns throughout the day.

def categorize_time_of_day(hour):
    if 5 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 17:
        return 'Afternoon'
    elif 17 <= hour < 21:
        return 'Evening'
    else:
        return 'Night'

df['Time_of_Day'] = df['hour_begining'].dt.hour.apply(categorize_time_of_day)

pedestrian_activity_by_time = df.groupby('Time_of_Day')['total_pedestrian_count'].mean()
