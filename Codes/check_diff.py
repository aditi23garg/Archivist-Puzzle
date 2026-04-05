import pandas as pd

# Compare recreation vs what we know sub2 looked like
rec_A = pd.read_csv('BookA_recreation.csv')
rec_B = pd.read_csv('BookB_recreation.csv')

print("BookA recreation - first 15 rows:")
print(rec_A.head(15).to_string(index=False))

print("\nBookB recreation - first 15 rows:")
print(rec_B.head(15).to_string(index=False))

print("\nBookA shuffled_page values (all):")
print(sorted(rec_A['shuffled_page'].tolist()))