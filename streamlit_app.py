import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Page config
st.set_page_config(page_title="Walmart Sales Predictor", layout="wide")

# Background image styling
st.markdown(
    '''
    <style>
    .stApp {
        background-image: url("https://images.unsplash.com/photo-1585128792020-96638c29be68");
        background-size: cover;
        background-attachment: fixed;
    }
    .main {
        background-color: rgba(255, 255, 255, 0.85);
        padding: 2rem;
        border-radius: 10px;
    }
    </style>
    ''',
    unsafe_allow_html=True
)

# Load model
with open("rf_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load data
df = pd.read_csv("Walmart_Sales.csv")
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year

# Sidebar inputs
st.sidebar.header("📋 Input Features")
store = st.sidebar.slider("Store (1 to 45)", 1, 45, 20)
holiday = st.sidebar.selectbox("Is it a holiday week?", [0, 1])
temp = st.sidebar.slider("Temperature (°F)", -10.0, 110.0, 60.0)
fuel = st.sidebar.slider("Fuel Price ($)", 2.0, 5.0, 3.3)
cpi = st.sidebar.slider("CPI", 120.0, 230.0, 180.0)
unemployment = st.sidebar.slider("Unemployment Rate", 3.0, 15.0, 8.0)
month = st.sidebar.selectbox("Month", list(range(1, 13)))
year = st.sidebar.selectbox("Year", [2010, 2011, 2012])

# Prediction section
st.markdown('<div class="main">', unsafe_allow_html=True)
st.title("📊 Walmart Weekly Sales Predictor")
input_data = pd.DataFrame({
    'Store': [store],
    'Holiday_Flag': [holiday],
    'Temperature': [temp],
    'Fuel_Price': [fuel],
    'CPI': [cpi],
    'Unemployment': [unemployment],
    'Month': [month],
    'Year': [year]
})

if st.button("🎯 Predict Weekly Sales"):
    prediction = model.predict(input_data)[0]
    st.success(f"📈 Predicted Weekly Sales: ${prediction:,.2f}")

# Charts section
st.subheader("📈 Weekly Sales Trend")
sales_over_time = df.groupby('Date')['Weekly_Sales'].sum()
fig1, ax1 = plt.subplots()
sales_over_time.plot(ax=ax1, title="Total Weekly Sales Over Time")
ax1.set_xlabel("Date")
ax1.set_ylabel("Sales")
st.pyplot(fig1)

st.subheader("🏪 Average Sales Per Store")
avg_sales_store = df.groupby('Store')['Weekly_Sales'].mean()
fig2, ax2 = plt.subplots()
avg_sales_store.plot(kind="bar", ax=ax2, title="Average Weekly Sales Per Store")
ax2.set_xlabel("Store")
ax2.set_ylabel("Average Sales")
st.pyplot(fig2)

st.subheader("📦 Holiday vs Non-Holiday Sales Distribution")
fig3, ax3 = plt.subplots()
sns.boxplot(x='Holiday_Flag', y='Weekly_Sales', data=df, ax=ax3)
ax3.set_xticklabels(['Non-Holiday', 'Holiday'])
ax3.set_title("Sales Distribution by Holiday Flag")
st.pyplot(fig3)

st.markdown('</div>', unsafe_allow_html=True)
