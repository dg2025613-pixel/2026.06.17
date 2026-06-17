import streamlit as st
import pandas as pd
import numpy as np

import plotly.express as px

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


st.set_page_config(
    page_title="해양 종 다양성 원인 분석",
    layout="wide"
)

st.title("🌊 해양 종 다양성 원인 분석 앱")

# -----------------------------
# 데이터 불러오기
# -----------------------------
df = pd.read_csv("realistic_ocean_climate_dataset.3.csv")

# 문자열 처리
df["Marine Heatwave"] = df["Marine Heatwave"].astype(str)

# 결측치 제거
df = df.dropna()

# -----------------------------
# 사이드바
# -----------------------------
page = st.sidebar.radio(
    "페이지 선택",
    ["대시보드", "원인 분석", "AI 예측"]
)

# ==================================================
# 대시보드
# ==================================================
if page == "대시보드":

    st.header("📊 데이터 개요")

    st.dataframe(df.head())

    col1, col2 = st.columns(2)

    with col1:

        fig = px.histogram(
            df,
            x="Species Observed",
            title="종 다양성 분포"
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:

        location_mean = (
            df.groupby("Location")["Species Observed"]
            .mean()
            .reset_index()
        )

        fig = px.bar(
            location_mean,
            x="Location",
            y="Species Observed",
            title="지역별 평균 종 다양성"
        )

        st.plotly_chart(fig, use_container_width=True)

# ==================================================
# 원인 분석
# ==================================================
elif page == "원인 분석":

    st.header("🔬 종 다양성 감소 원인 분석")

    st.subheader("상관관계")

    corr_df = df[
        [
            "SST (°C)",
            "pH Level",
            "Species Observed"
        ]
    ]

    fig = px.imshow(
        corr_df.corr(),
        text_auto=True,
        title="상관관계 분석"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        fig = px.scatter(
            df,
            x="SST (°C)",
            y="Species Observed",
            trendline="ols",
            title="수온과 종 다양성"
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:

        fig = px.scatter(
            df,
            x="pH Level",
            y="Species Observed",
            trendline="ols",
            title="pH와 종 다양성"
        )

        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.subheader("해양열파 영향")

    heatwave_mean = (
        df.groupby("Marine Heatwave")
        ["Species Observed"]
        .mean()
        .reset_index()
    )

    fig = px.bar(
        heatwave_mean,
        x="Marine Heatwave",
        y="Species Observed",
        title="해양열파 발생 여부와 종 다양성"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("백화현상 영향")

    bleach_mean = (
        df.groupby("Bleaching Severity")
        ["Species Observed"]
        .mean()
        .reset_index()
    )

    fig = px.bar(
        bleach_mean,
        x="Bleaching Severity",
        y="Species Observed",
        title="백화현상 심각도와 종 다양성"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.subheader("가장 큰 원인은?")

    analysis = []

    analysis.append(
        f"평균 수온 : {df['SST (°C)'].mean():.2f}℃"
    )

    analysis.append(
        f"평균 pH : {df['pH Level'].mean():.2f}"
    )

    analysis.append(
        f"평균 종 다양성 : {df['Species Observed'].mean():.1f}"
    )

    for text in analysis:
        st.write(text)

# ==================================================
# AI 예측
# ==================================================
else:

    st.header("🤖 종 다양성 예측")

    model_df = df.copy()

    model_df["Marine Heatwave"] = (
        model_df["Marine Heatwave"]
        .map({"True":1,"False":0})
    )

    model_df["Bleaching Severity"] = (
        model_df["Bleaching Severity"]
        .map({
            "Low":1,
            "Medium":2,
            "High":3
        })
    )

    X = model_df[
        [
            "SST (°C)",
            "pH Level",
            "Bleaching Severity",
            "Marine Heatwave"
        ]
    ]

    y = model_df["Species Observed"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42
    )

    model.fit(X_train, y_train)

    sst = st.slider(
        "해수면 온도",
        20.0,
        35.0,
        28.0
    )

    ph = st.slider(
        "pH",
        7.5,
        8.3,
        8.0
    )

    bleaching = st.selectbox(
        "백화현상",
        ["Low","Medium","High"]
    )

    heatwave = st.selectbox(
        "해양열파",
        ["False","True"]
    )

    bleaching_num = {
        "Low":1,
        "Medium":2,
        "High":3
    }[bleaching]

    heatwave_num = {
        "False":0,
        "True":1
    }[heatwave]

    pred = model.predict(
        [[
            sst,
            ph,
            bleaching_num,
            heatwave_num
        ]]
    )[0]

    st.metric(
        "예상 종 다양성",
        f"{pred:.0f} 종"
    )

    st.subheader("원인 중요도")

    importance = pd.DataFrame({
        "변수": X.columns,
        "중요도": model.feature_importances_
    })

    fig = px.bar(
        importance.sort_values(
            "중요도",
            ascending=False
        ),
        x="중요도",
        y="변수",
        orientation="h"
    )

    st.plotly_chart(fig, use_container_width=True)
