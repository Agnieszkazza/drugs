import sqlite3
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import cross_val_score
# wczytanie i przygotowanie danych:
drugs = pd.read_csv(r"C:\Users\zales\Documents\projekty\drugs\Medicine_Details.csv")

numeric_data = drugs.select_dtypes(include=['number']).columns
drugs[numeric_data] = drugs[numeric_data].fillna(drugs[numeric_data].mean())

categorical_data = drugs.select_dtypes(include=['object']).columns
for column in categorical_data:
    drugs[column] = drugs[column].fillna(drugs[column].mode()[0])



conn = sqlite3.connect('drugs.db')
cursor = conn.cursor()
drugs.to_sql('drugs', conn, if_exists='replace', index=False)

# Najważniejsze analizy i obliczenia:
#OCENY
print("OCENY")
cursor.execute("DROP TABLE IF EXISTS avg_reviews;")
cursor.execute('''
        CREATE TABLE avg_reviews (
      Medicine_Name TEXT,
      avg_excellent REAL,
      avg_average REAL,
      avg_poor REAL
);''')
conn.commit()

query = ''' SELECT "Medicine Name" AS Medicine_Name,
            COALESCE (AVG("Excellent Review %"),0.0) AS avg_excellent,
            COALESCE (AVG("Average Review %"),0.0) AS avg_average,
            COALESCE (AVG("Poor Review %"),0.0) AS avg_poor
            FROM drugs 
            GROUP BY "Medicine Name";'''
avg_data = cursor.execute(query).fetchall()
cursor.executemany('''
INSERT INTO avg_reviews (Medicine_Name, avg_excellent, avg_average, avg_poor)
VALUES (?, ?, ?, ?);
''', avg_data)
conn.commit()

print("Najlepsze oceny lekow")
for row in cursor.execute('''SELECT *  FROM 
                          avg_reviews ORDER BY avg_excellent DESC LIMIT 5;'''):
    print(row)

print("Najgorsze oceny lekow")
for row in cursor.execute('''SELECT * FROM avg_reviews ORDER BY avg_poor ASC LIMIT 5;'''):
    print(row)

# Zastosowania, skutki uboczne, substancje pomocnicze
#SUMY

print("Najczęstsze zastosowania leku")
for row in cursor.execute('''SELECT "Medicine Name",Uses, COUNT(*) AS count FROM drugs GROUP BY "Medicine Name",Uses  
                          ORDER BY "Medicine Name", count DESC LIMIT 10;'''):
    print(row)

print("Najczęstsze skutki uboczne dla leku")
for row in cursor.execute('''SELECT "Medicine Name", Side_effects, COUNT(*) AS count
                  FROM drugs 
                  GROUP BY "Medicine Name", Side_effects 
                  ORDER BY "Medicine Name", count DESC 
                  LIMIT 10'''):
    
    print("Najczęstsze substancje w lekach")
for row in cursor.execute(''' SELECT "Medicine Name", Composition, COUNT(*) AS count FROM drugs GROUP BY "Medicine Name",
                           Composition ORDER BY "Medicine Name", count DESC LIMIT 10 '''):
    print(row)

uses_query = '''SELECT "Medicine Name", Uses, COUNT(*) AS count
FROM drugs
GROUP BY "Medicine Name", Uses
ORDER BY "Medicine Name", count DESC; '''
uses_data = cursor.execute(uses_query).fetchall()

side_query = '''SELECT "Medicine Name", Side_effects, COUNT(*) AS count
FROM drugs
GROUP BY "Medicine Name", Side_effects
ORDER BY "Medicine Name", count DESC;'''
side_data = cursor.execute(side_query).fetchall()

composition_query = '''SELECT "Medicine Name", Composition, COUNT(*) AS count
FROM drugs
GROUP BY "Medicine Name", Composition
ORDER BY "Medicine Name", count DESC;'''
composition_data = cursor.execute(composition_query).fetchall()


cursor.execute("DROP TABLE IF EXISTS counted_information;")
cursor.execute('''CREATE TABLE counted_information (
    Medicine_Name TEXT,
    Most_Common_Use TEXT,
    Use_Count INTEGER,
    Most_Common_Side_Effect TEXT,
    Side_Effect_Count INTEGER,
    Most_Common_Composition TEXT,
    Composition_Count INTEGER
);''')
conn.commit()

summary_data = []
medicine_names = {row[0] for row in uses_data}  # Pobranie unikalnych nazw leków

for name in medicine_names:
    most_common_use = next(((use, count) for _, use, count in uses_data if _ == name), (None, 0))
    most_common_side_effect = next(((side_effect, count) for _, side_effect, count in side_data if _ == name), (None, 0))
    most_common_composition = next(((composition, count) for _, composition, count in composition_data if _ == name), (None, 0))

    # Dodanie danych do listy
    summary_data.append((
        name,
        most_common_use[0], most_common_use[1],
        most_common_side_effect[0], most_common_side_effect[1],
        most_common_composition[0], most_common_composition[1]
    ))

# Wprowadzenie danych do tabeli
cursor.executemany('''
    INSERT INTO counted_information (
        Medicine_Name,
        Most_Common_Use, Use_Count,
        Most_Common_Side_Effect, Side_Effect_Count,
        Most_Common_Composition, Composition_Count
    )
    VALUES (?, ?, ?, ?, ?, ?, ?);
''', summary_data)

conn.commit()


query = 'SELECT * FROM avg_reviews'
data1 = pd.read_sql_query(query,conn)
query = 'SELECT * FROM counted_information'
data2 = pd.read_sql_query(query,conn)
data2.to_csv(r'C:\Users\zales\Documents\projekty\drugs\counted_information.csv', index=False)
data1.to_csv(r'C:\Users\zales\Documents\projekty\drugs\avg_reviews.csv', index=False)

# wykresy
query = 'SELECT * FROM avg_reviews'
avg_reviews_df = pd.read_sql_query(query,conn)
print(avg_reviews_df.head())

top_med = avg_reviews_df.nlargest(5,'avg_excellent')
sns.set_theme(style='whitegrid')
plt.figure(figsize=(10,6))
sns.barplot(x=top_med['Medicine_Name'],y= top_med['avg_excellent'],palette ='pastel')
plt.title("5 najlepszych leków")
plt.xlabel('Lek')
plt.ylabel('Średnia najlepsza ocena')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

filtered_avg_reviews = avg_reviews_df[avg_reviews_df['avg_poor'] > 0]
worst_med = filtered_avg_reviews.nsmallest(5, 'avg_poor')
sns.set_theme(style='whitegrid')
plt.figure(figsize=(10,6))
sns.barplot(x= worst_med['Medicine_Name'], y=worst_med['avg_poor'], palette = 'PuRd')
plt.title("5 najgorszych leków")
plt.xlabel('Lek')
plt.ylabel('Średnia najgorsza ocena')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

query = 'SELECT * FROM counted_information;'
data = pd.read_sql_query(query,conn) 
print(data.head())

uses_counts = data.groupby("Most_Common_Use").size().sort_values(ascending=False).head(10)
sns.set_theme(style='whitegrid')
plt.figure(figsize=(10,8))
sns.barplot(x=uses_counts.index, y=uses_counts.values, palette='Purples')
plt.title('Najczęstsze zastosowania leków', fontsize=16)
plt.xlabel('Zastosowania',fontsize=9)
plt.ylabel('Liczba wystąpień', fontsize=12)
plt.xticks(rotation=45,ha='right')
plt.tight_layout()
plt.show()

side_counts = data.groupby("Most_Common_Side_Effect").size().sort_values(ascending=False).head(10)
sns.set_theme(style='whitegrid')
plt.figure(figsize=(10,6))
sns.barplot(x=side_counts.index,y=side_counts.values, palette = 'Blues')
plt.title("Najczęstsze skutki uboczne", fontsize=16)
plt.xlabel('Skutki uboczne', fontsize=9)
plt.ylabel('Liczba wystąpień', fontsize=12)
plt.xticks(rotation=45,ha='right')
plt.tight_layout()
plt.show()

composition_counts = data.groupby("Most_Common_Composition").size().sort_values(ascending=False).head(10)
sns.set_theme(style='whitegrid')
plt.figure(figsize=(10,6))
sns.barplot(x=composition_counts.index,y=composition_counts.values,palette='RdPu')
plt.title("Najczęsciej występujące substancje w lekach", fontsize=16)
plt.xlabel('Substancje',fontsize=9)
plt.ylabel('Liczba wystąpień', fontsize=12)
plt.xticks(rotation=45,ha='right')
plt.tight_layout()
plt.show()

print("Sprawdzenie",avg_reviews_df.columns)
# k-means
avg_reviews_df = avg_reviews_df.drop_duplicates(subset='Medicine_Name')
data = data.drop_duplicates(subset='Medicine_Name')
drugs = drugs.drop_duplicates(subset='Medicine Name')
merged_data = pd.concat([avg_reviews_df.set_index('Medicine_Name'),data.set_index('Medicine_Name'),drugs.set_index('Medicine Name')],axis=1)
merged_data.to_csv('medicines.csv', index=False)

features = ['avg_excellent', 'avg_average', 'avg_poor', 'Use_Count', 'Side_Effect_Count']
X = merged_data[features]
print(X.isnull().sum()) 

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3,random_state=42)
kmeans.fit(X_scaled)

merged_data['Cluster']=kmeans.labels_

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 1. Liczba próbek w każdym klastrze
cluster_sizes = merged_data['Cluster'].value_counts()
print("Rozmiar klastra:")
print(cluster_sizes)

# 2. Średnie wartości cech w każdym klastrze
cluster_means = merged_data.groupby('Cluster')[features].mean()
print("Średnie wartości cech w klastrach:")
print(cluster_means)

# 3. Sumy cech w każdym klastrze
cluster_sums = merged_data.groupby('Cluster')[features].sum()
print("Sumy cech w klastrach:")
print(cluster_sums)

# Wizualizacja z liczbowymi informacjami w tytule
plt.figure(figsize=(10,8))
plt.scatter(X_pca[:,0], X_pca[:,1], c=merged_data['Cluster'], cmap='viridis', s=50)
plt.title("Wizualizacja klastrów")

# Dodanie liczby próbek w każdym klastrze
for cluster_id in merged_data['Cluster'].unique():
    cluster_size = cluster_sizes[cluster_id]
    plt.text(X_pca[merged_data['Cluster'] == cluster_id, 0].mean(),
             X_pca[merged_data['Cluster'] == cluster_id, 1].mean(),
             f"Cluster {cluster_id}: {cluster_size} samples",
             fontsize=12, color='black', ha='center')

plt.xlabel('Wymiar 1')
plt.ylabel('Wymiar 2')
plt.colorbar(label='Klaster')
plt.show()

#korelacja między wysoka ocena a liczba skutkow ubocznych
correlation_data = pd.merge(avg_reviews_df, data[['Medicine_Name','Side_Effect_Count']], left_on='Medicine_Name',right_on='Medicine_Name')
correlation = correlation_data[['avg_excellent','Side_Effect_Count']].corr()
print("Korelacja wysoka ocena vs skutki uboczne")
print(correlation)

plt.figure(figsize=(8,6))
plt.scatter(correlation_data['avg_excellent'], correlation_data['Side_Effect_Count'], alpha=0.6,color='pink')
plt.title('Średnia ocena vs liczba skutków ubocznych')
plt.xlabel('Średnia najlepsza ocena (avg_excellent)')
plt.ylabel('Liczba skutków ubocznych (Side_Effect_Count)')
plt.grid(True)
plt.show()

# Rożnice między klastrami wg zastosowan lekow
cluster_usage = merged_data.groupby(['Cluster', 'Most_Common_Use']).size().unstack(fill_value=0)
top_uses = cluster_usage.sum(axis=0).nlargest(10).index
cluster_usage_top = cluster_usage[top_uses]


cluster_usage_percent_top = cluster_usage_top.div(cluster_usage_top.sum(axis=1), axis=0) * 100
cluster_usage_percent_top.T.plot(kind='bar', figsize=(16, 10), colormap='viridis')

plt.title('Procentowy udział najczęstszych zastosowań leków w klastrach', fontsize=16)
plt.xlabel('Zastosowania leków', fontsize=12)
plt.ylabel('Procentowy udział', fontsize=12)
plt.legend(title='Klaster', fontsize=10)
plt.xticks(rotation=60, ha='right', fontsize=10)
plt.tight_layout()
plt.show()


# Przewidywanie oceny leku - regresja liniowa

print(merged_data.head())
np.random.seed(0)
X = np.random.rand(100,2)
epsilon = np.random.randn(100)*2
y = X[:,0] + X[:,1] + epsilon

required_columns = ['avg_excellent','avg_poor','Use_Count','Side_Effect_Count','Composition_Count']
if all(col in merged_data.columns for col in required_columns):
    X = merged_data[['avg_excellent','Use_Count','Composition_Count']]
    y = merged_data['Excellent Review %']


    X_const = sm.add_constant(X)
    model = sm.OLS(y,X_const)
    results = model.fit()

    intercept = results.params[0]
    coefficients = results.params[1:]

    print("intercept",intercept)
    print("coefficients", coefficients)
    
    predictions = results.predict(X_const)
    print("predykcje:",predictions)

mse = mean_squared_error(y,predictions)
r2 = r2_score(y,predictions)
print("MSE:",mse)
print("r2:",r2)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
X_train_const = sm.add_constant(X_train)
model = sm.OLS(y_train,X_train_const)
result = model.fit()

X_test_const = sm.add_constant(X_test)
predictions = result.predict(X_test_const)

mse_test = mean_squared_error(y_test,predictions)
r2_test = r2_score(y_test,predictions)
print("Test MSE:",mse_test)
print("R2 test:",r2_test)

model = LinearRegression()
scores = cross_val_score(model,X,y, cv=5,scoring = 'r2')
print("Cross-validated R^2 scores:", scores)
print("Mean R^2 score:", np.mean(scores))
print(results.summary())
#wizualizacja regresji
conn.close()