import pandas as pd

print("=== BookA first 5 pages ===")
df = pd.read_csv('BookA_test.csv')
for i in range(5):
    text = str(df['text'].iloc[i])
    print(f"--- Page ID {df['page'].iloc[i]} ---")
    print(' '.join(text.split()[:40]))
    print()

print("=== BookB first 5 pages ===")
df2 = pd.read_csv('BookB_test.csv')
for i in range(5):
    text = str(df2['text'].iloc[i])
    print(f"--- Page ID {df2['page'].iloc[i]} ---")
    print(' '.join(text.split()[:40]))
    print()