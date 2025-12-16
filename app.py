import pickle

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

MODEL_DIR_PATH = "./model/"
CT_FILE = "column_transformer.pkl"
FEATURE_NAMES_FILE = "feature_names.pkl"
MODEL_FILE = "model.pkl"
SCALER_FILE = "scaler.pkl"


@st.cache_resource
def load_models() -> tuple[ColumnTransformer, list[str], Ridge, StandardScaler]:
    with open(MODEL_DIR_PATH + CT_FILE, "rb") as ct_file:
        column_transformer = pickle.load(ct_file)
        # print(f"загружено: {column_transformer}")

    with open(MODEL_DIR_PATH + FEATURE_NAMES_FILE, "rb") as fn_file:
        feature_names = pickle.load(fn_file)
        # print(f"загружено: {feature_names}")

    with open(MODEL_DIR_PATH + MODEL_FILE, "rb") as model_file:
        model = pickle.load(model_file)
        # print(f"загружено: {model}")

    with open(MODEL_DIR_PATH + SCALER_FILE, "rb") as scaler_file:
        scaler = pickle.load(scaler_file)
        # print(f"загружено: {scaler}")

    return column_transformer, feature_names, model, scaler


def process_eda(tab, dataset_path) -> None:
    with tab:
        df = pd.read_csv(dataset_path)
        st.subheader("Данные")
        st.dataframe(df)
        st.write(f"Размерность данных: {df.shape[0]} строк и {df.shape[1]} столбцов")
        st.subheader("Pairplot")

        fig = px.scatter_matrix(
            df,
            dimensions=df.select_dtypes(include=["number"]).columns.tolist(),
            title="Общий pairplot по числовым признакам",
        )
        st.plotly_chart(fig, width="stretch", height=800)

        st.subheader("Тепловая карта корреляций")
        fig = px.imshow(
            df.corr(numeric_only=True),
            text_auto=True,
            title="Тепловая карта корреляций между числовыми признаками",
        )
        st.plotly_chart(fig, width="stretch", height=800)

        st.subheader("Распределение целевой переменной")
        fig = px.histogram(
            df,
            x="selling_price",
            nbins=200,
            title="Распределение целевой переменной 'price'",
        )
        st.plotly_chart(fig, width="stretch", height=800)

        st.subheader("Сравнение численности категроий")
        df_cat = df.select_dtypes(include="object").drop("name", axis=1)
        for col in df_cat.columns:
            fig = px.histogram(
                df,
                x=col,
                title=f"Распределение категориального признака '{col}'",
            )
            st.plotly_chart(fig, width="stretch", height=800)


def drop_units(x) -> float | None:
    number = str(x).rstrip("kmpl km/kg bhp CC")
    return float(number) if number else None


def preprocess_dataframe(
    df: pd.DataFrame, column_transformer: ColumnTransformer, scaler: StandardScaler
):
    """Предобработка пользовательских данных

    Args:
        df (pd.DataFrame): Пользовательские данные
        column_transformer (CiolumnTransformer): Преобразователь столбцов
        scaler (StandardScaler): Скейлер числовых признаков

    Returns:
        pd.DataFrame: Предобработанные данные
    """

    try:
        df[["mileage", "engine", "max_power"]] = (
            df[["mileage", "engine", "max_power"]]
            .map(lambda x: drop_units(x))
            .astype(float)
        )
    except ValueError:
        st.error(
            "Не удалось подгтовить данные. Проверьте наличие пропусков значений в таблице"
        )
        st.stop()
    medians = df.median(numeric_only=True)
    df = df.fillna(value=medians)
    if "torque" in df.columns.to_list():
        df = df.drop("torque", axis=1)
    if "selling_price" in df.columns.to_list():
        df = df.drop("selling_price", axis=1)

    df[["seats", "engine"]] = df[["seats", "engine"]].astype(int)
    num_cols = df.select_dtypes(include="number").columns
    df[num_cols] = scaler.transform(df[num_cols])

    df.name = df.name.apply(lambda x: x.split()[0])
    X_val = column_transformer.transform(df)
    return X_val


def main() -> None:
    st.set_page_config(page_title="Car Price Prediction", page_icon="🚗", layout="wide")

    column_transformer, feature_names, model, scaler = load_models()
    st.title("Предсказание стоимости автомобиля")
    st.header("Описание исходного набора данных")

    train_eda, test_eda = st.tabs(["Train EDA", "Test EDA"])

    with train_eda:
        process_eda(train_eda, "./datasets/train_dataset.csv")

    with test_eda:
        process_eda(test_eda, "./datasets/test_dataset.csv")

    st.header("Проверка модели на пользовательских данных")
    st.text(
        "Загрузите Ваши данные удобным способом (csv файл или вручную), а затем ниже отметьте выбранный формат загрузки"
    )

    load_csv, manual_type = st.tabs(["Загрузить из файла", "Ввести вручную"])

    with load_csv:
        uploaded_file = st.file_uploader(
            "Загрузите файл с новыми данными",
            type=["csv"],
        )
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.subheader("Загруженные данные")
            st.dataframe(df)

            st.session_state.csv_df = df

    with manual_type:
        table = pd.DataFrame(columns=feature_names, data=[["0"] * len(feature_names)])
        table[["year", "km_driven"]] = table[["year", "km_driven"]].astype(int)
        config = {
            "name": st.column_config.TextColumn(),
            "fuel": st.column_config.TextColumn(),
            "seller_type": st.column_config.TextColumn(),
            "transmission": st.column_config.TextColumn(),
            "owner": st.column_config.TextColumn(),
            "mileage": st.column_config.TextColumn(),
            "engine": st.column_config.TextColumn(),
            "max_power": st.column_config.TextColumn(),
        }

        edited_df = st.data_editor(
            table,
            num_rows="dynamic",
            column_config=config,
            width="stretch",
            key="manual_editor",
        )

        st.session_state.manual_df = edited_df

    source = st.radio(
        "Источник данных для предсказания",
        options=["CSV файл", "Ручной ввод"],
        horizontal=True,
    )

    if st.button("Предсказать цену"):
        if source == "CSV файл":
            current_df = st.session_state.csv_df
        else:
            current_df = st.session_state.manual_df
        X_val = preprocess_dataframe(current_df, column_transformer, scaler)
        # print(X_val)
        predictions = model.predict(X_val)
        results_df = current_df.copy()
        results_df["predicted_selling_price"] = predictions
        st.subheader("Результаты предсказания")
        st.dataframe(results_df)

        st.subheader("Распределение весов модели")
        fig = px.histogram(model.coef_)
        st.plotly_chart(fig)


if __name__ == "__main__":
    main()
