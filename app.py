import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="AI E-Commerce Sales Insight", layout="wide")

st.title("AI E-Commerce Sales Insight System")
st.write("This app helps sellers understand product performance and predict whether a product is likely to be a high-selling or low-selling product.")

# Load data
df = pd.read_csv("sales_data.csv")

st.subheader("Product Sales Data")
st.dataframe(df)

st.subheader("Sales Distribution")
st.bar_chart(df["purchases"])

st.subheader("Views vs Purchases")
st.scatter_chart(df[["views", "purchases"]])

st.write("This model learns patterns from product data such as views, price, rating, and stock to predict sales performance.")

# Create target column
df["high_selling"] = df["purchases"].apply(lambda x: 1 if x >= 70 else 0)

# Features and target
X = df[["price", "views", "rating", "stock"]]
y = df["high_selling"]

# Train model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.subheader("Model Performance")
st.write(f"Prediction Accuracy: {accuracy:.2f}")

# Insights
st.subheader("Business Insights")
top_products = df.sort_values(by="purchases", ascending=False).head(5)
st.write("Top Selling Products")
st.dataframe(top_products[["product_name", "category", "purchases", "rating"]])

low_stock = df[df["stock"] < 20]
st.write("Products with Low Stock")
st.dataframe(low_stock[["product_name", "stock", "purchases"]])

# User input
st.subheader("Predict Product Performance")

price = st.number_input("Price", min_value=1.0, value=20.0)
views = st.number_input("Views", min_value=1, value=300)
rating = st.number_input("Rating", min_value=1.0, max_value=5.0, value=4.0)
stock = st.number_input("Stock", min_value=0, value=30)

if st.button("Predict"):
    input_data = pd.DataFrame([[price, views, rating, stock]], columns=["price", "views", "rating", "stock"])
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.success("This product is likely to be HIGH-SELLING.")
    else:
        st.warning("This product is likely to be LOW-SELLING.")

st.subheader("Project Explanation")
st.write(
    "This beginner AI project uses product data such as price, views, rating, and stock to predict product sales performance. "
    "It can help sellers make better decisions about promotion, stock planning, and product strategy."
)
