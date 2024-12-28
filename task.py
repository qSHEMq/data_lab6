import pandas as pd
import matplotlib.pyplot as plt
import json


def load_data(file_path):
    return pd.read_csv(file_path)


def analyze_data(data):
    memory_usage = data.memory_usage(deep=True).sum() / (1024 * 1024)  # MB
    column_memory = data.memory_usage(deep=True) / (1024 * 1024)  # MB
    column_info = pd.DataFrame(
        {
            "memory_mb": column_memory,
            "percent_of_total": (column_memory / memory_usage) * 100,
            "dtype": data.dtypes,
        }
    )
    return memory_usage, column_info


def save_stats(column_info, file_name="column_memory_stats.json"):
    # Преобразуем типы данных в строку для избежания проблем с сериализацией
    column_info["dtype"] = column_info["dtype"].astype(str)

    # Преобразуем DataFrame в словарь и сохраняем с помощью json
    column_info_dict = column_info.to_dict(orient="index")
    with open(file_name, "w") as f:
        json.dump(column_info_dict, f, indent=4)


def optimize_data(data):
    for col in data.select_dtypes(include=["object"]):
        if data[col].nunique() < 0.5 * len(data):
            data[col] = data[col].astype("category")

    int_cols = data.select_dtypes(include=["int"]).columns
    data[int_cols] = data[int_cols].apply(pd.to_numeric, downcast="integer")

    float_cols = data.select_dtypes(include=["float"]).columns
    data[float_cols] = data[float_cols].apply(pd.to_numeric, downcast="float")

    return data


def load_with_chunks(file_path, selected_columns, chunk_size=10000):
    chunk_list = []
    for chunk in pd.read_csv(file_path, usecols=selected_columns, chunksize=chunk_size):
        chunk_list.append(chunk)
    return pd.concat(chunk_list)


def plot_graphs(data):
    plt.figure(figsize=(12, 8))

    # Линейный график
    if "Vict Age" in data.columns:
        plt.subplot(2, 2, 1)
        data["Vict Age"].plot(kind="line")
        plt.title("Line Chart")

    # Столбчатая диаграмма
    if "Vict Sex" in data.columns:
        plt.subplot(2, 2, 2)
        data["Vict Sex"].value_counts().plot(kind="bar")
        plt.title("Bar Chart")

    # Круговая диаграмма
    if "Vict Descent" in data.columns:
        plt.subplot(2, 2, 3)
        data["Vict Descent"].value_counts().plot(kind="pie")
        plt.title("Pie Chart")

    # Корреляция
    numeric_data = data.select_dtypes(include=["int64", "float64"])
    if not numeric_data.empty:
        plt.subplot(2, 2, 4)
        corr = numeric_data.corr()
        plt.imshow(corr, cmap="coolwarm", interpolation="none")
        plt.colorbar()
        plt.title("Correlation Heatmap")

    plt.tight_layout()
    plt.show()


def main():
    file_path = "D:/data_lab6/data/Crime_Data_from_2020_to_Present.csv"
    data = load_data(file_path)

    # Initial analysis
    initial_memory, initial_column_info = analyze_data(data)
    save_stats(initial_column_info, "initial_column_memory_stats.json")

    # Optimize data
    optimized_data = optimize_data(data)
    optimized_memory, optimized_column_info = analyze_data(optimized_data)
    save_stats(optimized_column_info, "optimized_column_memory_stats.json")

    print(f"Initial memory usage: {initial_memory} MB")
    print(f"Optimized memory usage: {optimized_memory} MB")

    # Убедитесь, что загружаются нужные столбцы
    selected_columns = [
        "DR_NO",
        "Date Rptd",
        "Vict Age",
        "Vict Sex",
        "Vict Descent",
        "LAT",
        "LON",
    ]
    subset_data = load_with_chunks(file_path, selected_columns)
    subset_data.to_csv("subset_data.csv", index=False)

    # Plot graphs
    plot_graphs(subset_data)


if __name__ == "__main__":
    main()
