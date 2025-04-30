import pandas as pd

def main():
    preferences = pd.read_csv('preferences.csv')
    categories = preferences[['category_1', 'category_2', 'category_3']].values
    for category in categories:
        print('category: ', category)

    category1s = categories[:, 0]
    category2s = categories[:, 1]
    category3s = categories[:, 2]

    for category1, category2, category3 in zip(category1s, category2s, category3s):
        print('category1: ', category1, "category2: ", category2, "category3: ", category3)

if __name__ == "__main__":
    main()