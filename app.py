import streamlit as st
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

st.set_page_config(page_title="Diabetes Prediction", layout="centered")
st.title(" Diabetes Progression Prediction")

data = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)
pred = model.predict(X_test)

st.write(f"MSE: {mean_squared_error(y_test, pred):.2f}")
st.write(f"R2 Score: {r2_score(y_test, pred):.2f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.scatter(y_test, pred, color="blue", alpha=0.5)
ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "k--", lw=2)
ax1.set_title("Actual vs. Predicted Values")
ax1.set_xlabel("Actual Values")
ax1.set_ylabel("Predicted Values")
ax1.grid(True)

ax2.scatter(X_test[:, 2], pred, color="green", alpha=0.5)
ax2.set_title("Feature(BMI) vs Predicted Values")
ax2.set_xlabel("BMI (Feature 2)")
ax2.set_ylabel("Predicted Diabetes Progression")
ax2.grid(True)

st.pyplot(fig)
