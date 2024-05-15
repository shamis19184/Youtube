#!/usr/bin/env python
# coding: utf-8

# # Exploratory Data Analysis on YouTube data.
# # Domain: Social media
# # Context and Content: In a fairly recent move by Youtube, it announced the decision to hide the number of dislikes from users around November 2021. However, the official YouTube Data API allowed you to get information about dislikes until December 13, 2021. Doing an EDA-exercise can help to draw some unseen insights from this dataset.
# # Learning Outcome:
# # ● Exploratory Data Analysis using Pandas.
# # Objective:
# # To do data analysis and explore the youtube dislikes dataset using numpy and pandas libraries and drive meaningful insights by performing Exploratory data analysis.
# # Data Description:
# # YouTube Dislikes Dataset:
# # ● This dataset contains information about trending YouTube videos from August 2020 to December 2021 for the USA, Canada, and Great Britain.
# # ● This dataset contains the latest possible information about dislikes,likes,views and more which was collected just before December 13. The information was collected by videos that had been trending in the USA, Canada, and Great Britain for a year prior.
# # ● Dataset link: https://www.kaggle.com/datasets/dmitrynikolaev/youtube-dislikes-dataset

# # Attribute Information:
# 
# 1. Video ID - Unique video id.
# 2. Title - Video title.
# 3. Channel ID - Id of the channel.
# 4. Channel Title - Title of the channel.
# 5. Published at - Video publication date.
# 6. View count - Number of views.
# 7. Likes - Number of likes.
# 8. Dislikes - Number of dislikes.
# 9. Comment Count - Number of comments.
# 10. Tags - Tags (in one string).
# 11. Description - Video description.
# 12. Comments-  20 Video comments (in one string)
# 

# # 1. Import required libraries and read the provided dataset (youtube_dislike_dataset.csv) and retrieve top 5 and bottom 5 records.
# 

# In[98]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[99]:


df = pd.read_csv('youtube_dislike_dataset.csv')


# In[100]:


df.head()


# In[101]:


df.tail()


# # 2. Check the info of the dataframe and write your inferences on data types and shape of the dataset.
# 

# In[102]:


df.info()


# # - The dataset has a total of  37422 rows and 17 columns.
# # - There are columns with data types like int64 - object.
# # - There is some missing value in the comments column.

# # 3. Check for the Percentage of the missing values and drop or impute them.

# In[103]:


df.isnull().mean()*100


# In[104]:


df1 = df.dropna()


# In[105]:


df1.isnull().mean()*100


# # we dropped the missing values and saved in other variable

# # 4. Check the statistical summary of both numerical and categorical columns and write your inferences.

# In[106]:


df1.describe().T


# In[107]:


df1.describe(include = 'object').T


# # - Summary statistics for numerical columns.
# # - Count, unique, top, and freq for categorical columns.

# # 5. Convert datatype of column published_at from object to pandas datetime.

# In[108]:


df1['published_at'] = pd.to_datetime(df1['published_at'])
df1['published_at']


# # 6. Create a new column as 'published_month' using the column published_at (display the months only)

# In[109]:


df1['published_month'] = df1['published_at'].dt.month
df1['published_month']


# # 7. Replace the numbers in the column published_month as names of the months i,e., 1 as 'Jan', 2 as 'Feb' and so on.....
# 

# In[110]:


month_mapping = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'} 
df1['published_month'] = df1['published_month'].apply(lambda x: month_mapping.get(x, 'Unknown'))
df1['published_month']


# # 8. Find the number of videos published each month and arrange the months in a decreasing order based on the video count.

# In[111]:


df1['published_month'].value_counts().sort_index(ascending=False)


# # 9. Find the count of unique video_id, channel_id and channel_title.

# In[112]:


print(df1[['video_id', 'channel_id', 'channel_title']].nunique())


# # 10. Find the top10 channel names having the highest number of videos in the dataset and the bottom 10 having lowest number of videos.
# 

# In[113]:


df1['channel_title'].value_counts().sort_values(ascending=False).head(10)


# In[114]:


df1['channel_title'].value_counts().sort_values().head(10)


# # 11. Find the title of the video which has the maximum number of likes and the title of the video having minimum likes and write your inferences

# In[118]:


max_likes_title = df1.loc[df1['likes'].idxmax()]['title']
print(f"Video with maximum likes: {max_likes_title}")


# In[119]:


min_likes_title = df1.loc[df1['likes'].idxmin()]['title']
print(f"Video with minimum likes: {min_likes_title}")


# # 12. Find the title of the video which has the maximum number of dislikes and the title of the video having minimum dislikes and write your inferences.

# In[121]:


max_dislikes_title = df1.loc[df1['dislikes'].idxmax()]['title']
print(f"Video with maximum dislikes: {max_dislikes_title}")


# In[123]:


min_dislikes_title = df1.loc[df1['dislikes'].idxmin()]['title']
print(f"Video with minimum dislikes: {min_dislikes_title}")


# # 13. Does the number of views have any effect on how many people disliked the video? Support your answer with a metric and a plot.
# 

# In[125]:


correlation_views_dislikes = df1['view_count'].corr(df1['dislikes'])
print(f"Correlation between views and dislikes: {correlation_views_dislikes}")


# In[129]:


plt.scatter(df1['view_count'], df1['dislikes'] )
plt.title('Relationship between Views and Dislikes')
plt.xlabel('Views')
plt.ylabel('Dislikes')
plt.show()


# # 14. Display all the information about the videos that were published in January, and mention the count of videos that were published in January.

# In[138]:


df1[df1['published_month'] == 'Jan']


# In[141]:


df1[df1['published_month'] == 'Jan'].nunique()


# In[140]:


len(df1[df1['published_month'] == 'Jan'])

