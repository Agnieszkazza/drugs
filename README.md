# Analiza danych o lekach

## Opis projektu
Projekt ma na celu przeprowadzenie analizy danych dotyczących leków, ich ocen, zastosowań, skutków ubocznych i składu. Wykorzystano różne techniki analityczne i wizualizacyjne, takie jak grupowanie, analiza korelacji oraz predykcja ocen przy użyciu regresji liniowej.

## Funkcjonalności
- **Czyszczenie i przygotowanie danych**
  - Uzupełnienie brakujących wartości liczbowych za pomocą średniej.
  - Uzupełnienie brakujących wartości kategorii za pomocą wartości modalnej.

- **Integracja z bazą danych SQLite**
  - Wczytanie danych do bazy SQLite.
  - Tworzenie tabel z przetworzonymi danymi (np. średnie oceny leków, najczęstsze zastosowania, skutki uboczne, składniki).

- **Analizy i statystyki**
  - Obliczenie średnich ocen leków (najlepsze, średnie, najgorsze).
  - Identyfikacja najczęstszych zastosowań leków, skutków ubocznych oraz substancji składowych.

- **Wizualizacje**
  - Tworzenie wykresów przedstawiających m.in.:
    - Top 5 leków o najwyższych ocenach.
    - Top 5 leków o najniższych ocenach.
    - Najczęstsze zastosowania leków.
    - Najczęstsze skutki uboczne.
    - Najpopularniejsze składniki leków.
  - Wizualizacja klastrów leków oraz procentowego udziału zastosowań w każdym klastrze.

- **Grupowanie danych**
  - Grupowanie leków za pomocą algorytmu K-Means w oparciu o cechy takie jak oceny, liczba skutków ubocznych czy liczba składników.

- **Modelowanie predykcyjne**
  - Budowa modelu regresji liniowej do przewidywania ocen leków na podstawie ich cech.
  - Walidacja modelu przy użyciu podziału na zbiory treningowe i testowe oraz cross-walidacji.

## Technologie i narzędzia
- **Języki programowania:** Python
- **Baza danych:** SQLite
- **Biblioteki Python:**
  - Analiza danych: `pandas`, `numpy`
  - Wizualizacja: `matplotlib`, `seaborn`
  - Uczenie maszynowe: `scikit-learn`, `statsmodels`
- **Algorytmy:**
  - Klasteryzacja: K-Means
  - Regresja liniowa

## Jak uruchomić projekt
### Wymagania
- Python 3.8 lub nowszy.
- Zainstalowane biblioteki (można użyć pliku `requirements.txt`):
  ```bash
  pip install pandas numpy matplotlib seaborn scikit-learn statsmodels
