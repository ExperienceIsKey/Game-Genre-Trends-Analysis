# 🎮 Video Game Genre Trend Analysis

📌 Overview

This project explores trends in video game genres using data from 30,000+ video games (2016-2024) sourced from the Steam API and Kaggle. We analyze revenue, review scores, ownership trends, and genre evolution to provide actionable insights for developers, marketers, and industry stakeholders.


🎯 Objectives

* Identify popular and emerging game genres.
* Analyze review scores, revenue, and ownership trends.
* Utilize TF-IDF and Cosine Similarity for content-based recommendations.
* Offer data-driven insights for the gaming industry.


📊 Dataset

* Source: Steam API, Kaggle
* Games Analyzed: 30,000+
* Attributes: Revenue, review scores, ownership data, genres, tags, release dates
* Processing: Data cleaning, standardization, genre classification

📌 Dataset Overview

* Total Number of Games: 🕹️ 65,112
* Total Number of Distinct Games: 🕹️ 38,471

1️⃣ Genre Distribution (Genre):

* Battle Royale: 1,030 games
* Multiplayer: 318 games
* Role-Playing Games (RPG): 15,758 games
* Racing: 1,754 games
* Strategy: 10,281 games
* Sports: 1,606 games

2️⃣ Game Distribution (Pricing Model):

* Free to Play: 605 games
* Paid: 38,399 games

📌 Dataset Columns & Description

1️⃣ Game Identification
* App ID 🏷️ – Unique identifier assigned to each game in the Steam database
* Title 🎮 – Name of the game

2️⃣ Reviews & Ratings
* Reviews Total 📝 – Total number of reviews submitted by users
* Reviews Score Fancy ⭐ – Steam's formatted rating based on user reviews
* Reviews D7 📆 – Number of reviews received in the last 7 days
* Reviews D30 📆 – Number of reviews received in the last 30 days
* Reviews D90 📆 – Number of reviews received in the last 90 days

3️⃣ Game Release & Pricing
* Release Date 🗓️ – Date when the game was launched on Steam
* Launch Price 💰 – Initial price of the game at release

🔍 Methodology

* Data Cleaning & Transformation: Handled missing values, standardized titles, converted data types.
* Ownership Estimation: Applied the Boxleiter method for estimating game ownership.
* Trend Analysis: Visualized genre trends over time.
* Genre Similarity Search: Implemented TF-IDF + Cosine Similarity to recommend similar games.


📈 Key Insights

✔ RPG & Multiplayer games remain dominant in ownership and revenue.

✔ Paid games have higher review scores than Free-to-Play.

✔ Strategy & Sports games show steady, high ratings over time.

✔ Battle Royale & Multiplayer genres are growing rapidly.

✔ Revenue is highly influenced by top-performing titles.


🚀 Technologies Used

* Python, PySpark, AWS, Spark-SQL, Docker
* Jupyter Notebook, Pandas, Matplotlib, Seaborn
* TF-IDF, Cosine Similarity for content-based recommendations


🛠 Future Enhancements

* Cross-platform analysis (PC, console, mobile).
* Impact of emerging technologies (AR/VR, cloud gaming).
* Sentiment analysis of user reviews.
* Influencer impact tracking on game success.
