%%configure -f
{"executorMemory": "2G","driverMemory":"1G","executorCores": 1,"numExecutors": 3, 
 "conf": {"spark.dynamicAllocation.enabled": "false", 
          "spark.sql.parquet.enableVectorizedReader": "false", 
          "spark.pyspark.python": "python3",
          "spark.pyspark.virtualenv.enabled": "true",
          "spark.pyspark.virtualenv.type": "native",
          "spark.pyspark.virtualenv.bin.path": "/usr/bin/virtualenv"}}

sc.install_pypi_package("pandas")
sc.install_pypi_package("matplotlib")
sc.install_pypi_package("scikit-learn")
sc.install_pypi_package("seaborn")
sc.install_pypi_package("numpy")
sc.install_pypi_package("ipywidgets")

# Importing required packages

from pyspark.sql.functions import col, year, avg, when, length, expr, isnan
from pyspark.sql.types import StructType, StructField, IntegerType, FloatType
from pyspark.sql.types import IntegerType
import matplotlib.pyplot as plt
from pyspark.sql.functions import lit
import seaborn as sns
import numpy as np
import pandas as pd
from pyspark.sql import functions as F
from functools import reduce
from pyspark.sql.functions import regexp_replace
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ipywidgets as widgets
from IPython.display import display

# Loading the dataset 

gameR = spark.read.options(inferSchema = True).csv('s3://msbx5420-spr24/teams/fastCurious/gamesReviews.csv', header=True)
gameR.limit(5).toPandas().head()

# Creating a Dataframe for the Avg Review to Sales Multiplier Data

# Define the data as a list of tuples
data = [
    (2017, 81.25),
    (2018, 78.125),
    (2019, 71.875),
    (2020, 40.625),
    (2021, 42.1875),
    (2022, 37.5),
    (2023, 34.0)
]

# Define the schema for the DataFrame
schema = StructType([
    StructField("Year", IntegerType(), True),
    StructField("Review Coeff", FloatType(), True)
])

# Create the DataFrame
reviewCoeff = spark.createDataFrame(data, schema)

# Show the DataFrame to verify its creation
reviewCoeff.show()

# Creating a Temp View
reviewCoeff.createOrReplaceTempView("review_coeff")

# Cleaning the 'Title' column
gameR_cleaned = gameR.withColumn("Title", regexp_replace("Title", "[^a-zA-Z0-9 ]", ""))
gameR_cleaned.limit(5).toPandas().head()

# Checking rows where the Review Score is 0%

# Filter for rows where the Review Score is '0%'
zero_percent_reviews = gameR_cleaned.filter(gameR_cleaned["Reviews Score Fancy"] == '0%')

# Count the Rows
zero_percent_reviews.count()


# Checking rows where the Title is NA

# Filter for rows where the Review Score is '0%'
na_titles = gameR_cleaned.filter(gameR["Title"] == '')

# Count the Rows
na_titles.count()

# Removing rows where the Review Score is 0%
gameR_filtered = gameR_cleaned.filter(gameR_cleaned["Reviews Score Fancy"] != '0%')

# Show the result
gameR_filtered.limit(5).toPandas().head()

# Count the rows
print(gameR_filtered.count())

# Battle Royale Games

# Define the tags you're interested in
include_tags = ["Battle Royale", "Online", "Online Co Op"]
exclude_tags = ["RPG", "Racing", "Sports", "RTS", "Builder"]

# Filter the DataFrame
# The game must have at least one of the include_tags
# And must not have any of the exclude_tags
battle_games_df = gameR_filtered.filter(
    # Check if any of the include_tags are present in the 'Tags' column
    (col("Tags").rlike("|".join(include_tags)))
    # And none of the exclude_tags are present in the 'Tags' column
    # Uncomment and adjust the following line if you have specific tags to exclude
    & ~(col("Tags").rlike("|".join(exclude_tags)))
)

# Show some of the results
battle_games_df.select("Title", "Tags").distinct().count()

# Massively Multiplayer Games

# Define the tags you're interested in
include_tags = ["Massively Multiplayer", "Online", "Online Co-op", "MOBA"]
exclude_tags = ["RPG", "Racing", "Strategy", "Sports", "Battle Royale", "Builder", "RTS", "Builder", "RTS", "Simulation", "Singleplayer"]

# Filter the DataFrame
# The game must have at least one of the include_tags
# And must not have any of the exclude_tags
multiplayer_games_df = gameR_filtered.filter(
    # Check if any of the include_tags are present in the 'Tags' column
    (col("Tags").rlike("|".join(include_tags)))
    # And none of the exclude_tags are present in the 'Tags' column
    # Uncomment and adjust the following line if you have specific tags to exclude
    & ~(col("Tags").rlike("|".join(exclude_tags)))
)

# Show some of the results
multiplayer_games_df.select("Title", "Tags").distinct().count()

# Role Playing Games (Offline Only)

# Define the tags you're interested in
include_tags = ["RPG", "Local Co Op", "Action RPG", "Open World", "Singleplayer"]
exclude_tags = ["MOBA", "Racing", "Strategy", "Massively Multiplayer", "Online", "Online Co-Op", "Sports", "Battle Royale", "Multiplayer", "Builder", "RTS"]

# Filter the DataFrame
# The game must have at least one of the include_tags
# And must not have any of the exclude_tags
rpg_games_df = gameR_filtered.filter(
    # Check if any of the include_tags are present in the 'Tags' column
    (col("Tags").rlike("|".join(include_tags)))
    # And none of the exclude_tags are present in the 'Tags' column
    # Uncomment and adjust the following line if you have specific tags to exclude
    & ~(col("Tags").rlike("|".join(exclude_tags)))
)

# Show some of the results
rpg_games_df.select("Title", "Tags").distinct().count()

# Racing Games

# Define the tags you're interested in
include_tags = ["Racing", "Driving", "Simulation"]
exclude_tags = ["MOBA", "Strategy", "Massively Multiplayer", "Battle Royale", "Survival", "Shooter", "RPG", "Builder", "RTS", "Puzzle", "Indie"]

# Filter the DataFrame
# The game must have at least one of the include_tags
# And must not have any of the exclude_tags
racing_games_df = gameR_filtered.filter(
    # Check if any of the include_tags are present in the 'Tags' column
    (col("Tags").rlike("|".join(include_tags)))
    # And none of the exclude_tags are present in the 'Tags' column
    # Uncomment and adjust the following line if you have specific tags to exclude
    & ~(col("Tags").rlike("|".join(exclude_tags)))
)

# Show some of the results
racing_games_df.select("Title", "Tags").distinct().count()

# Strategy Games

# Define the tags you're interested in
include_tags = ["RTS", "Strategy", "Builder", "Puzzle"]
exclude_tags = ["MOBA", "Massively Multiplayer", "Sports", "Battle Royale", "Shooter", "RPG", "Shooter", "Open World", "Racing", "Driving"]

# Filter the DataFrame
# The game must have at least one of the include_tags
# And must not have any of the exclude_tags
strategy_games_df = gameR_filtered.filter(
    # Check if any of the include_tags are present in the 'Tags' column
    (col("Tags").rlike("|".join(include_tags)))
    # And none of the exclude_tags are present in the 'Tags' column
    # Uncomment and adjust the following line if you have specific tags to exclude
    & ~(col("Tags").rlike("|".join(exclude_tags)))
)

# Show some of the results
strategy_games_df.select("Title", "Tags").distinct().count()

# Sports Games

# Define the tags you're interested in
include_tags = ["Sports", "Arcade", "Simulation"]
exclude_tags = ["MOBA", "Massively Multiplayer", "Battle Royale", "Shooter", "RPG", "RTS", "Strategy", "Builder", "Action", "Indie"]

# Filter the DataFrame
# The game must have at least one of the include_tags
# And must not have any of the exclude_tags
sports_games_df = gameR_filtered.filter(
    # Check if any of the include_tags are present in the 'Tags' column
    (col("Tags").rlike("|".join(include_tags)))
    # And none of the exclude_tags are present in the 'Tags' column
    # Uncomment and adjust the following line if you have specific tags to exclude
    & ~(col("Tags").rlike("|".join(exclude_tags)))
)

# Show some of the results
sports_games_df.select("Title", "Tags").distinct().count()

# Free to Play Games

# Define the tags you're interested in
include_tags = ["Free to Play"]

# Filter the DataFrame
# The game must have at least one of the include_tags
# And must not have any of the exclude_tags
freeToPlay_games_df = gameR_filtered.filter(
    # Check if any of the include_tags are present in the 'Tags' column
    (col("Tags").rlike("|".join(include_tags)))
    # And none of the exclude_tags are present in the 'Tags' column
    # Uncomment and adjust the following line if you have specific tags to exclude
    # & ~(col("Tags").rlike("|".join(exclude_tags)))
)

# Show some of the results
freeToPlay_games_df.select("Title", "Tags").count()

# Paid Games

# Define the tags you're interested in
exclude_tags = ["Free to Play"]

# Filter the DataFrame
# The game must have at least one of the include_tags
# And must not have any of the exclude_tags
paid_games_df = gameR_filtered.filter(
    # Check if any of the include_tags are present in the 'Tags' column
    (~col("Tags").rlike("|".join(exclude_tags)))
    # And none of the exclude_tags are present in the 'Tags' column
    # Uncomment and adjust the following line if you have specific tags to exclude
    # & ~(col("Tags").rlike("|".join(exclude_tags)))
)

# Show some of the results
paid_games_df.select("Title", "Tags").count()

# Function to clean and convert percentage to integer
def clean_percentage(df, column_name):
    # Ensure column names with spaces are enclosed in backticks for SQL expressions
    return df.withColumn(column_name, 
                         when(col(column_name).isNull(), None)
                         .otherwise(expr(f"regexp_replace(`{column_name}`, '%', '')").cast('float')))

# Clean review scores for all genre DataFrames
battle_games_df = clean_percentage(battle_games_df, "Reviews Score Fancy")
multiplayer_games_df = clean_percentage(multiplayer_games_df, "Reviews Score Fancy")
rpg_games_df = clean_percentage(rpg_games_df, "Reviews Score Fancy")
racing_games_df = clean_percentage(racing_games_df, "Reviews Score Fancy")
strategy_games_df = clean_percentage(strategy_games_df, "Reviews Score Fancy")
sports_games_df = clean_percentage(sports_games_df, "Reviews Score Fancy")

freeToPlay_games_df = clean_percentage(freeToPlay_games_df, "Reviews Score Fancy")
paid_games_df = clean_percentage(paid_games_df, "Reviews Score Fancy")

# Preparing data for plotting the most popular game in each genre based on review score, revenue generated, and total owners (Plot 1 & 2)

# Add a 'Genre' column to each DataFrame and union them all into one DataFrame
def add_genre_and_union(df, genre_name):
    return df.withColumn("Genre", lit(genre_name))

# Prepare the list of genre DataFrames along with their names
genre_dfs = [
    (multiplayer_games_df, "Multiplayer"),
    (battle_games_df, "Battle Royale"),
    (rpg_games_df, "RPG"),
    (sports_games_df, "Sports"),
    (racing_games_df, "Racing"),
    (strategy_games_df, "Strategy"),
]

# Union all the genre DataFrames with the new 'Genre' column added
final_df1 = reduce(lambda x, y: x.union(add_genre_and_union(y[0], y[1])),
                  genre_dfs,
                  spark.createDataFrame([], schema=multiplayer_games_df.schema.add("Genre", "string")))

# Ensure the Revenue Estimated is in a numeric format
final_df1 = final_df1.withColumn("Revenue Estimated", F.regexp_replace(col("Revenue Estimated"), "[\$,]", "").cast("double"))

# Register the DataFrame as a temp view for SQL processing
final_df1.createOrReplaceTempView("games_data")

final_df1.show()

# Revenue Generated vs Game w/ the Highest Revenue in Each Genre
top_revenue_df = spark.sql("""
SELECT Genre, Title, (`Revenue Estimated` / 1000000) AS Revenue_Millions
FROM (
    SELECT *, ROW_NUMBER() OVER (PARTITION BY Genre ORDER BY `Revenue Estimated` DESC) as rank
    FROM games_data
) WHERE rank = 1
""")
top_revenue_df.createOrReplaceTempView("top_revenue")

# Owners vs Top Game Having the Highest Number of Owners in Each Genre

review_coeff_df = spark.createDataFrame([(2017, 81.25), (2018, 78.125), (2019, 71.875), (2020, 40.625), (2021, 42.1875), (2022, 37.5), (2023, 34.0)], ["Year", "Review Coeff"])
review_coeff_df.createOrReplaceTempView("review_coeff")

top_owners_df = spark.sql("""
SELECT Genre, Title, Owners_Thousand
FROM (
    SELECT 
        g.Genre, 
        g.Title, 
        (`Reviews Total` * rc.`Review Coeff` / 10000) AS Owners_Thousand,
        ROW_NUMBER() OVER (PARTITION BY g.Genre ORDER BY `Reviews Total` * rc.`Review Coeff` / 10000 DESC) as rank
    FROM games_data g
    JOIN review_coeff rc ON YEAR(g.`Release Date`) = rc.Year
    WHERE rc.`Year` BETWEEN 2017 AND 2023
) ranked_games
WHERE rank = 1
""")
top_owners_df.createOrReplaceTempView("top_owners")

top_revenue_df.show()

def plot_top_games(df, x_col, y_col, title_col, title, y_label):
    # Convert Spark DataFrame to Pandas for plotting
    pd_df = df.toPandas()

    plt.figure(figsize=(14, 10))  # Increase figure size for better visibility
    # Set 'hue' to the x_col to have different colors and turn off the legend
    ax = sns.barplot(x=x_col, y=y_col, hue=x_col, data=pd_df, palette="viridis", errorbar=None)
    # ax.get_legend().remove()  # Remove the legend
    plt.title(title)
    plt.xlabel('Genre')
    plt.ylabel(y_label)
    plt.xticks(rotation=45, ha='right')  # Ensure genre names are readable

    # Annotate each bar with the game's title
    for p, label in zip(ax.patches, pd_df[title_col]):
        text_x = p.get_x() + p.get_width() / 2
        text_y = p.get_height()
        # Adjust text placement based on bar height
        if text_y < 20:  # Threshold for text placement inside or outside the bar
            vertical_alignment = 'bottom'
            ytext = 10  # Offset to place text above the bar when bar is too short
        else:
            vertical_alignment = 'center'
            ytext = 0  # Offset to place text inside the bar

        ax.annotate(label,
                    xy=(text_x, text_y),
                    xytext=(0, ytext),  # Nudge text up slightly for visibility
                    textcoords="offset points",
                    ha='center', va=vertical_alignment,
                    color='black' if vertical_alignment == 'center' else 'blue',  # color inside bar differently for visibility
                    fontsize=9, rotation=45)  # Smaller font size

    plt.tight_layout()
    plt.show()
    
plot_top_games(top_revenue_df, 'Genre', 'Revenue_Millions', 'Title', 'Top Game by Revenue in Each Genre', 'Revenue (in Million $)')
%matplot plt

top_owners_df.show()

plot_top_games(top_owners_df, 'Genre', 'Owners_Thousand', 'Title', 'Top Game by Estimated Owners in Each Genre', 'Owners (in Thousand)')
%matplot plt

# Preparing data for plotting the graph for analysing Free-To-Play and paid games (Plot 3, 4 & 5)

# Add a 'Genre' column to each DataFrame and union them all into one DataFrame
def add_genre_and_union(df, genre_name):
    return df.withColumn("Genre", lit(genre_name))

# Prepare the list of genre DataFrames along with their names
dfs = [
    (freeToPlay_games_df, "Free To Play"),
    (paid_games_df, "Paid"),
]

# Union all the genre DataFrames with the new 'Genre' column added
freePaid_df = reduce(lambda x, y: x.union(add_genre_and_union(y[0], y[1])),
                  dfs,
                  spark.createDataFrame([], schema=freeToPlay_games_df.schema.add("Genre", "string")))

# Ensure the Revenue Estimated is in a numeric format
freePaid_df = freePaid_df.withColumn("Revenue Estimated", F.regexp_replace(col("Revenue Estimated"), "[\$,]", "").cast("double"))

# Register the DataFrame as a temp view for SQL processing
freePaid_df.createOrReplaceTempView("free_paid_data")

# Avg Review Score by Genre each Year (Free to Play vs Paid)
avg_review_score_df = spark.sql("""
SELECT Genre, YEAR(f.`Release Date`) AS Year, AVG(`Reviews Score Fancy`) AS Avg_Review_Score
FROM free_paid_data f
WHERE 
    YEAR(f.`Release Date`) BETWEEN 2017 AND 2023
GROUP BY Genre, Year
""")

# Avg Revenue Estimated by Genre each Year (Free to Play vs Paid)
avg_revenue_df = spark.sql("""
SELECT Genre, YEAR(f.`Release Date`) AS Year, (AVG(`Revenue Estimated`)/10000) AS Avg_Revenue_Estimated
FROM free_paid_data f
WHERE 
    YEAR(f.`Release Date`) BETWEEN 2017 AND 2023
GROUP BY Genre, Year
""")

# Avg Owners by Genre each Year (Free to Play vs Paid)
avg_owners_df = spark.sql("""
SELECT 
    f.Genre,
    YEAR(f.`Release Date`) as Year,
    (AVG(f.`Reviews Total`) * AVG(rc.`Review Coeff`)) / 1000 AS Avg_Owners
FROM 
    free_paid_data f
JOIN 
    review_coeff rc 
ON 
    YEAR(f.`Release Date`) = rc.Year
WHERE 
    rc.Year BETWEEN 2017 AND 2023
GROUP BY 
    f.Genre, YEAR(f.`Release Date`)
ORDER BY
    f.Genre
""")

def plot_genre_data(df, x_col, y_col, title, y_label):
    # Convert the Spark DataFrame to a Pandas DataFrame for plotting
    plot_pd = df.toPandas()

    # Set plot size and style
    plt.figure(figsize=(15, 8))
    sns.set(style="whitegrid")

    # Plotting
    ax = sns.barplot(x=x_col, y=y_col, hue='Year', data=plot_pd, palette='viridis')

    # Set a title and adjust the axis labels
    plt.title(title, fontsize=16)
    plt.xlabel('Genre', fontsize=14)
    plt.ylabel(y_label, fontsize=14)

    # Rotate x-labels for better readability
    plt.xticks(rotation=45, ha='right', fontsize=12)

    # Add annotations
    for p in ax.patches:
        height = p.get_height()
        # Skip annotations with height 0 or NaN
        if height > 0 and not pd.isna(height):
            plt.text(p.get_x() + p.get_width() / 2., height, f'{height:.1f}',
                     ha='center', va='bottom', fontsize=10, rotation=0, color='black')

    # Apply tight layout to adjust the plot parameters for the new figure size
    plt.tight_layout()

    # Move the legend outside the plot to the right
    plt.legend(title='Year', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)

    # Show plot
    plt.show()

# Now call the function for each of the dataframes
plot_genre_data(avg_review_score_df, 'Genre', 'Avg_Review_Score', 'Average Review Score by Genre', 'Average Review Score')

plot_genre_data(avg_revenue_df, 'Genre', 'Avg_Revenue_Estimated', 'Average Revenue Estimated by Genre', 'Average Revenue (10K)')
%matplot plt

plot_genre_data(avg_owners_df, 'Genre', 'Avg_Owners', 'Average Owners by Genre', 'Average Owners (in Thousands)')
%matplot plt

# Preparing the data for Plotting Game Genre Trends based on Review Scores (Plot 2)

# Filter rows based on the last 5 years
current_year = 2024
years = [current_year - i for i in range(5)]  # [2024, 2023, 2022, 2021, 2020]

def filter_and_avg(df, genre_name):
    return df.filter(year(col("Release Date")).isin(years)) \
             .groupBy(year(col("Release Date")).alias("Year")) \
             .agg(avg(col("Reviews Score Fancy")).alias("Avg_Score")) \
             .withColumn("Genre", lit(genre_name))

# Apply the function to each genre
genres = ["Battle Royale", "Multiplayer", "RPG", "Racing", "Strategy", "Sports"]
dfs = [battle_games_df, multiplayer_games_df, rpg_games_df, racing_games_df, strategy_games_df, sports_games_df]

genre_avg_scores = [filter_and_avg(df, genre) for df, genre in zip(dfs, genres)]

# Combine all the DataFrames
from functools import reduce
final_df2 = reduce(lambda x, y: x.union(y), genre_avg_scores)

# Collect data for plotting
plot_data = final_df2.toPandas()
plot_data.head()

# Creating the final_df for plotting the graph (Plot 6)

final_df2.createOrReplaceTempView("games_data_avg_review")

# Adjust the years in the BETWEEN clause if necessary
plot_data = spark.sql("""
    SELECT Year, Avg_Score, Genre
    FROM games_data_avg_review
    WHERE Year BETWEEN 2020 AND 2024
    ORDER BY Genre, Year
""")

# Convert the DataFrame to Pandas for plotting
plot_pd = plot_data.toPandas()

# Set the aesthetic style of seaborn
sns.set(style="whitegrid")

# Create the figure and axes
fig, ax = plt.subplots(figsize=(14, 8))

# Sort the dataframe based on Genre to ensure consistent bar ordering
plot_pd.sort_values(by=['Genre', 'Year'], inplace=True)

# Define x-axis and bar width
genres = plot_pd['Genre'].unique()
num_genres = len(genres)
num_years = plot_pd['Year'].nunique()
bar_width = 0.15
x = np.arange(num_genres)

# Plot bars for each year
for i, year in enumerate(sorted(plot_pd['Year'].unique())):
    # Filter by year
    df_year = plot_pd[plot_pd['Year'] == year]
    
    # Plot
    ax.bar(x - bar_width*num_years/2 + i*bar_width, df_year['Avg_Score'], width=bar_width, label=str(year))

# Draw a trend line for each genre
for i, genre in enumerate(genres):
    # Filter by genre
    df_genre = plot_pd[plot_pd['Genre'] == genre]
    
    # Draw a line plot with the same x-coordinate offset for each genre group
    ax.plot(x[i] + np.arange(-bar_width*num_years/2, bar_width*num_years/2, bar_width)[:len(df_genre)], df_genre['Avg_Score'], 
            marker='o', linestyle='-', label=f"{genre} Trend" if i == 0 else "")

# Add labels and title
ax.set_xlabel('Genre')
ax.set_ylabel('Average Review Score')
ax.set_title('Average Review Score by Genre and Year')
ax.set_xticks(x)
ax.set_xticklabels(genres)

# Add a legend
ax.legend(title='Year', loc='center right')

# Rotate the genre labels for better readability
plt.xticks(rotation=45)

# Show the plot
plt.tight_layout()
plt.show()
%matplot plt

# Preparing the data for Plotting Game Genre Trends based on avg revenue generated (Plot 7)

# Similar to first plot
# Register the DataFrame as a temp view for SQL processing
final_df1.createOrReplaceTempView("games_data")

avg_revenue_df = spark.sql("""
SELECT 
    Genre, 
    YEAR(`Release Date`) as Year,
    AVG(`Revenue Estimated`) / 100000 AS AVG_Revenue
FROM 
    games_data
WHERE 
    YEAR(`Release Date`) BETWEEN 2017 AND 2023
GROUP BY 
    Genre, YEAR(`Release Date`)
ORDER BY 
    Genre, YEAR(`Release Date`)
""")

# View results
avg_revenue_df.show()

def plot_data(df, y_col, title, y_label):
    # Convert to Pandas DataFrame for plotting
    pd_df = df.toPandas()

    # Set the figure size to be larger
    plt.figure(figsize=(18, 10))

    # Plot with a larger bar width using the 'dodge' parameter
    barplot = sns.barplot(x='Genre', y=y_col, hue='Year', data=pd_df, palette='viridis', dodge=True)

    # Set the title and labels with a larger font size
    plt.title(title, fontsize=16)
    plt.xlabel('Genre', fontsize=14)
    plt.ylabel(y_label, fontsize=14)

    # Rotate the x-axis labels to prevent overlap
    plt.xticks(rotation=45, ha='right', fontsize=12)

    # Annotate each bar with its height
    for p in barplot.patches:
        height = p.get_height()
        # If height is nan or zero (which can happen with missing data), we don't want to annotate it
        if not pd.isna(height) and not height == 0:
            barplot.annotate(format(height, '.2f'),
                             (p.get_x() + p.get_width() / 2., height),
                             ha='center', va='bottom',
                             xytext=(0, 10),  # 10 points vertical offset
                             textcoords='offset points', fontsize=10)

    # Move the legend outside the plot to the right
    plt.legend(title='Year', bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=12)

    # Apply tight layout to adjust the plot parameters for the new figure size
    plt.tight_layout()

    # Show the plot
    plt.show()

# Plot Average Revenue by Genre
plot_data(avg_revenue_df, 'AVG_Revenue', 'Average Revenue by Genre (2017-2023)', 'Average Revenue (Million $)')
%matplot plt

# Preparing the data for Plotting Game Genre Trends based on avg game owners (Plot 8)

# Using the same data as Plot 1 and Plot 3

reviewCoeff.createOrReplaceTempView("reviewCoeff")

avg_owners_df = spark.sql("""
SELECT
    g.Genre,
    g.Year,
    (g.Avg_Reviews * rc.`Review Coeff`) / 1000 as Avg_Owners
FROM
    (SELECT 
        Genre, 
        YEAR(`Release Date`) as Year,
        AVG(`Reviews Total`) as Avg_Reviews
     FROM 
        games_data
     GROUP BY 
        Genre, YEAR(`Release Date`)
    ) g
JOIN
    reviewCoeff rc
ON
    g.Year = rc.Year
WHERE
    g.Year BETWEEN 2017 AND 2023
ORDER BY
    g.Genre, g.Year
""")

avg_owners_df.show()

def plot_data(df, y_col, title, y_label):
    # Convert to Pandas DataFrame for plotting
    pd_df = df.toPandas()

    # Set the figure size to be larger
    plt.figure(figsize=(14, 7))  # Adjusted figure size

    # Plotting with side-by-side bars
    barplot = sns.barplot(
        x='Genre', 
        y=y_col, 
        hue='Year', 
        data=pd_df, 
        palette='viridis', 
        dodge=True  # Ensure bars are side-by-side
    )

    # Set the title and labels with a larger font size
    plt.title(title, fontsize=16)
    plt.xlabel('Genre', fontsize=14)
    plt.ylabel(y_label, fontsize=14)

    # Rotate the x-axis labels to prevent overlap
    plt.xticks(rotation=45, ha='right', fontsize=12)

    # Adding the values on top of the bars
    for p in barplot.patches:
        height = p.get_height()
        if not pd.isna(height) and height != 0:  # Skip NaN values and zero height
            plt.text(p.get_x() + p.get_width() / 2., height, f'{height:.1f}',
                     ha='center', va='bottom', fontsize=10, rotation=90, color='black')

    # Move the legend outside the plot to the right
    plt.legend(title='Year', bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=12)

    plt.tight_layout()
    plt.show()

# Plot Average Game Owners by Genre and Year
plot_data(avg_owners_df, 'Avg_Owners', 'Average Game Owners by Genre and Year', 'Average Owners (in Thousand)')
%matplot plt

# Using the cleaned dataset
game_rec = gameR_cleaned

# Removing the % from the Review Scores and converting it to int
game_rec = game_rec.withColumn('Reviews Score Fancy', regexp_replace(col('Reviews Score Fancy'), '%', '').cast('integer'))

# Removing the $ from the Revenue Estimated and converting to double
game_rec = game_rec.withColumn('Revenue Estimated', regexp_replace(col('Revenue Estimated'), '[$,]', '').cast('float'))

game_rec.limit(4).toPandas().head()

# Getting count of rows having Review Score of 0
game_rec.filter(game_rec["Reviews Score Fancy"] == 0).count()

# Getting count of rows having Revenue Estimated as 0
game_rec.filter(game_rec["Revenue Estimated"] == 0).count()

# Checking for NaN values
game_rec.filter(isnan(col('Revenue Estimated'))).count()

# Removing rows with Review Score of 0
game_rec_cleaned = game_rec.filter(game_rec["Reviews Score Fancy"] != 0)
game_rec_cleaned.count()

# Sorting the data by Review Score
game_rec_cleaned = game_rec_cleaned[game_rec_cleaned["Reviews Score Fancy"] > 70]

# Filtering Rows where Revenue Estimated is greater than 1 Million
game_rec_cleaned = game_rec.filter(game_rec["Revenue Estimated"] > 1000000)
game_rec_cleaned.count()

print(type(game_rec_cleaned))

%%spark -o pandas_df
pandas_df = game_rec_cleaned

%%local
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import ipywidgets as widgets
from IPython.display import display

# Assuming new_data["Tags"] is already preprocessed
vectorizer = TfidfVectorizer(ngram_range=(1,2))
tfidf_matrix = vectorizer.fit_transform(pandas_df["Tags"])

# Define a search function

def search(genre):
    query_vec = vectorizer.transform([genre])
    similarity = cosine_similarity(query_vec, tfidf_matrix).flatten()
    indices = np.argsort(similarity)[-10:]  # Get indices for a larger pool of results
    similar_items = pandas_df.iloc[indices][::-1]  # Reverse to get the most similar first
    
    # Sort these items first by 'Revenue Estimated' and then by 'Metacritic Score', both in descending order
    sorted_items = similar_items.sort_values(by=['Revenue Estimated', 'Reviews Score Fancy', "Release Date"], ascending=[False, False, False])
    top_items = sorted_items.head(10)  # Get the top 5 results
    return top_items[['Title', "Release Date", 'Reviews Score Fancy', 'Revenue Estimated', 'Tags']]  # Return specific columns



# Create a text widget for input
genre_input = widgets.Text(
    value='',
    placeholder='Type genres (e.g., Action, Adventure)',
    description='Search:',
    disabled=False
)

# Output widget for results
genre_list = widgets.Output()

# Event handler for typing in the search box
def on_type(change):
    with genre_list:
        genre_list.clear_output()
        if change['new']:  # Only search if there is a new input
            results = search(change['new'])
            display(results)

# Bind the event handler
genre_input.observe(on_type, names='value')

# Display the widgets
display(genre_input, genre_list)
