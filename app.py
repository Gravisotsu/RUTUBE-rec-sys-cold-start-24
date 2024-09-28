import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Инициализация состояния
if "registered" not in st.session_state:
    st.session_state.registered = False


# Функция для обработки регистрации
def register():
    st.session_state.registered = True


# Функция для продолжения без регистрации
def continue_without_registration():
    st.session_state.registered = True  # Или сохраняем состояние без регистрации


# Функция для предсказания похожего описания
def predict_similar_description(new_description, descriptions, tfidf_matrix, vectorizer):
    new_tfidf = vectorizer.transform([new_description])
    similarity_matrix = cosine_similarity(new_tfidf, tfidf_matrix)
    most_similar_idx = similarity_matrix.argmax()
    return descriptions[most_similar_idx], similarity_matrix[0][most_similar_idx], most_similar_idx


# Функция для получения популярных видео
def get_popular_videos(df, num=5):
    return df.nlargest(num, "views")  # Предполагаем, что есть колонка 'views' с количеством просмотров


# Функция для получения случайных видео
def get_random_videos(df, num=5):
    return df.sample(num)


# Главный экран с видео
def main_screen(df):
    st.title("Рекомендательная система для видео")

    # Боковое меню
    with st.sidebar:
        st.header("Меню")
        menu_option = st.radio("Выберите вкладку:", ("Профиль", "Главная", "Персонализированный поиск", "Подписки", "Тренды", "Настройки"))

    # Приведение значений в колонке 'full_text' к строковому типу и удаление NaN
    df["full_text"] = df["full_text"].fillna("").astype(str)
    descriptions = df["full_text"].tolist()

    # Создаем TF-IDF матрицу
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(descriptions)

    # Поле поиска по описанию
    search_query = st.text_input("Введите текст для поиска видео:", "")

    # Фильтрация данных по запросу
    if search_query:
        filtered_df = df[df["full_text"].str.contains(search_query, case=False)]
        if filtered_df.empty:
            st.write("Нет видео, соответствующих вашему запросу.")
        else:
            st.write("Результаты поиска:")
            st.write(filtered_df[["full_text"]])  # Показываем только столбец с полными текстами
            # Показываем видео
            for index, row in filtered_df.iterrows():
                st.video(row["video_url"])  # Предполагаем, что в df есть колонка 'video_url'

    # Пользователь вводит текст для рекомендации
    user_input = st.text_input("Введите описание для поиска рекомендаций:")

    # Если текст введен
    if user_input:
        if st.button("Найти похожее видео"):
            # Выполняем предсказание
            predicted_description, similarity, most_similar_idx = predict_similar_description(
                user_input, descriptions, tfidf_matrix, vectorizer
            )

            # Отображаем результат
            st.subheader("Рекомендация")
            st.write(f"Наиболее похожее описание: {predicted_description}")
            st.write(f"Косинусное сходство: {similarity:.2f}")

    # Если пользователь не ввел текст для рекомендаций
    else:
        st.subheader("Рекомендации для вас")
        # Предлагаем популярные видео
        popular_videos = get_popular_videos(df)
        st.write("Популярные видео:")
        st.write(popular_videos[["full_text"]])  # Предполагаем, что вы хотите показывать полные тексты
        for index, row in popular_videos.iterrows():
            st.video(row["video_url"])  # Показываем видео

        # Или случайные видео
        random_videos = get_random_videos(df)
        st.write("Случайные видео:")
        st.write(random_videos[["full_text"]])  # Предполагаем, что вы хотите показывать полные тексты
        for index, row in random_videos.iterrows():
            st.video(row["video_url"])  # Показываем видео

    # Вывод исходных данных
    st.sidebar.subheader("Имеющиеся данные")
    st.sidebar.write(df)


# Установка цветовой гаммы
st.markdown("""
<style>
    .stApp {
        background-color: #121212;  /* Темный фон */
        color: #e0e0e0;  /* Светло-серый текст */
    }
    .sidebar .sidebar-content {
        background-color: #1e1e1e;  /* Темный фон для бокового меню */
    }
    input, textarea {
        background-color: #333333;  /* Темно-серый для полей ввода */
        color: #ffffff;  /* Белый текст в полях ввода */
    }
    .stButton > button {
        background-color: #bb86fc;  /* Светло-фиолетовый цвет для кнопок */
        color: #ffffff;  /* Белый текст на кнопках */
    }
    .stHeader, .stSubheader, .stText {
        color: #bb86fc;  /* Светло-фиолетовый текст для заголовков */
    }
</style>
""", unsafe_allow_html=True)

# Логика приложения
if not st.session_state.registered:
    st.title("Регистрация")
    st.write("Пожалуйста, зарегистрируйтесь или продолжите без регистрации.")
    if st.button("Зарегистрироваться"):
        register()
    if st.button("Продолжить без регистрации"):
        continue_without_registration()
else:
    # Загрузка данных из CSV файла, который должен находиться в той же директории
    df = pd.read_csv("/Users/dmitry/Хакатон/ML/itog.csv")  # Замените на путь к вашему файлу
    main_screen(df)
