import os
import pandas as pd


df = pd.read_csv(r'dataset/results.csv', sep='|', encoding='utf-8')
print(df.columns)
print(df['comment'].values[:20])


# niepotrzebne spacje pred znakami interpunkcyjnymi

print(df[['image_name', 'comment_number']].head())

# on ma tak: przechowuje dataset i tworzy pary na podstawie pierwszej sekwencji czesci a potem na outpucie jest kolejne slowo
