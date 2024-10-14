import pandas as pd
import datasets


def main():
    dataset = datasets.load_dataset("rotten_tomatoes")
    print(dataset)
    train_df = dataset["train"].to_pandas()
    print(train_df.head())


if __name__ == "__main__":
    main()
