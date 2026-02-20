import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from babel.numbers import format_currency

sns.set(style="dark")
st.set_page_config(layout="wide")
st.title("📊 E-Commerce Performance Dashboard")

# =========================
# LOAD DATA (Google Drive)
# =========================
@st.cache_data
def load_data():
    # Ganti dengan FILE_ID dari Google Drive kamu
    file_id = "1AYLGd9aNBL2dbAF3QRpykzwLGFaJBnpz"
    url = f"https://drive.google.com/uc?id={file_id}"
    df = pd.read_csv(url, parse_dates=["order_purchase_timestamp"])
    return df

all_df = load_data()

# =========================
# SIDEBAR FILTER
# =========================
min_date = all_df["order_purchase_timestamp"].min()
max_date = all_df["order_purchase_timestamp"].max()

with st.sidebar:
    st.image("https://github.com/dicodingacademy/assets/raw/main/logo.png")
    start_date, end_date = st.date_input(
        "Filter Tanggal",
        [min_date, max_date],
        min_value=min_date,
        max_value=max_date
    )

main_df = all_df[
    (all_df["order_purchase_timestamp"] >= pd.to_datetime(start_date)) &
    (all_df["order_purchase_timestamp"] <= pd.to_datetime(end_date))
]

# =========================
# 1️⃣ MONTHLY TREND
# =========================
st.header("📈 Monthly Orders & Revenue Trend")

main_df["total_price"] = main_df["price"] + main_df["freight_value"]

monthly_df = main_df.resample(
    rule="M",
    on="order_purchase_timestamp"
).agg({
    "order_id": "nunique",
    "total_price": "sum"
}).reset_index()

col1, col2 = st.columns(2)

with col1:
    st.metric("Total Orders", monthly_df["order_id"].sum())

with col2:
    total_rev = format_currency(
        monthly_df["total_price"].sum(),
        "BRL",
        locale="pt_BR"
    )
    st.metric("Total Revenue", total_rev)

fig, ax = plt.subplots(figsize=(14,5))
ax.plot(monthly_df["order_purchase_timestamp"],
        monthly_df["order_id"],
        marker="o")
ax.set_title("Monthly Orders")
st.pyplot(fig)

fig, ax = plt.subplots(figsize=(14,5))
ax.plot(monthly_df["order_purchase_timestamp"],
        monthly_df["total_price"],
        marker="o")
ax.set_title("Monthly Revenue")
st.pyplot(fig)

# =========================
# 2️⃣ CATEGORY PERFORMANCE
# =========================
st.header("🏆 Top Product Categories")

category_perf = (
    main_df
    .groupby("product_category_name")
    .agg(
        total_orders=("order_id","nunique"),
        total_revenue=("payment_value","sum")
    )
    .sort_values("total_orders", ascending=False)
)

top_orders = category_perf.head(10)
top_revenue = category_perf.sort_values(
    by="total_revenue",
    ascending=False
).head(10)

col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots(figsize=(10,6))
    sns.barplot(x=top_orders["total_orders"],
                y=top_orders.index,
                ax=ax)
    ax.set_title("Top 10 Categories by Orders")
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots(figsize=(10,6))
    sns.barplot(x=top_revenue["total_revenue"],
                y=top_revenue.index,
                ax=ax)
    ax.set_title("Top 10 Categories by Revenue")
    st.pyplot(fig)

# =========================
# 3️⃣ CITY DISTRIBUTION MAP
# =========================
st.header("🗺️ Order Distribution by City")

city_orders = (
    main_df
    .groupby("customer_city")["order_id"]
    .count()
    .reset_index()
)

city_orders.columns = ["city", "total_orders"]
city_orders = city_orders.sort_values(by="total_orders", ascending=False)

geo_city = (
    main_df
    .groupby("customer_city")[["geolocation_lat", "geolocation_lng"]]
    .mean()
    .reset_index()
)

geo_city.columns = ["city", "lat", "lng"]

city_map_df = city_orders.merge(geo_city, on="city", how="left").dropna()

m = folium.Map(location=[-14.2350, -51.9253], zoom_start=4)

for _, row in city_map_df.head(100).iterrows():
    if row["total_orders"] > 5000:
        color = "red"
    elif row["total_orders"] > 2000:
        color = "orange"
    else:
        color = "blue"

    folium.CircleMarker(
        location=[row["lat"], row["lng"]],
        radius=row["total_orders"] / 500,
        popup=f"{row['city']} - Orders: {row['total_orders']}",
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.6
    ).add_to(m)

st.components.v1.html(m._repr_html_(), width=1200, height=500)

# =========================
# 4️⃣ DELIVERY PERFORMANCE
# =========================
st.header("🚚 Delivery Performance")

col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots()
    ax.hist(main_df["delivery_time"].dropna(), bins=30)
    ax.set_title("Delivery Time Distribution")
    st.pyplot(fig)

with col2:
    late_ratio = main_df["late_delivery"].value_counts(normalize=True)
    fig, ax = plt.subplots()
    late_ratio.plot(kind="bar", ax=ax)
    ax.set_title("Late Delivery Proportion")
    st.pyplot(fig)

# =========================
# 5️⃣ RFM ANALYSIS
# =========================
st.header("👑 Customer Segmentation (RFM)")

rfm_df = main_df.groupby("customer_id", as_index=False).agg({
    "order_purchase_timestamp":"max",
    "order_id":"nunique",
    "total_price":"sum"
})

rfm_df.columns = ["customer_id","last_purchase","frequency","monetary"]

recent_date = main_df["order_purchase_timestamp"].max()
rfm_df["recency"] = (recent_date - rfm_df["last_purchase"]).dt.days

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Avg Recency", round(rfm_df["recency"].mean(),1))

with col2:
    st.metric("Avg Frequency", round(rfm_df["frequency"].mean(),2))

with col3:
    st.metric("Avg Monetary",
              format_currency(rfm_df["monetary"].mean(),"BRL",locale="pt_BR"))

top_customer = rfm_df.sort_values(by="monetary", ascending=False).head(5)

fig, ax = plt.subplots(figsize=(8,6))
sns.barplot(y="customer_id", x="monetary", data=top_customer, ax=ax)
ax.set_title("Top 5 Customers by Monetary")
ax.set_xlabel("Monetary Value")
ax.set_ylabel("Customer ID")
plt.tight_layout()
st.pyplot(fig)
