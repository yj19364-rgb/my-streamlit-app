from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# --------------------------------------------------
# 1. 페이지 설정 및 디자인
# --------------------------------------------------
st.set_page_config(
    page_title="기온 기반 의류 판매량 분석",
    page_icon="🧥",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
    div[data-testid="stMetric"] {
        background: #f7f9fc;
        border: 1px solid #e5eaf2;
        padding: 14px 16px;
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🧥 기온 기반 의류 판매량 분석 대시보드")
st.caption("기온·날짜·의류 종류·판매 채널에 따른 판매 현황을 분석하고 판매량을 예측합니다.")


# --------------------------------------------------
# 2. 데이터 불러오기 및 정리
# --------------------------------------------------
DATA_PATHS = [
    Path("data/clothing_sales_weather.csv"),
    Path("clothing_sales_weather.csv"),
    Path(__file__).resolve().parent / "data" / "clothing_sales_weather.csv",
]


@st.cache_data
def load_data(path_string: str) -> pd.DataFrame:
    df = pd.read_csv(path_string)
    df.columns = df.columns.str.strip()

    required_columns = {
        "date", "channel", "clothing_category", "units_sold", "avg_temperature"
    }
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(
            "CSV 파일에 필요한 열이 없습니다: " + ", ".join(sorted(missing))
        )

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["units_sold"] = pd.to_numeric(df["units_sold"], errors="coerce")
    df["avg_temperature"] = pd.to_numeric(df["avg_temperature"], errors="coerce")
    df["channel"] = df["channel"].astype("string").str.strip()
    df["clothing_category"] = df["clothing_category"].astype("string").str.strip()

    df = df.dropna(
        subset=[
            "date", "channel", "clothing_category",
            "units_sold", "avg_temperature"
        ]
    ).copy()

    df = df[
        (df["units_sold"] >= 0)
        & df["channel"].ne("")
        & df["clothing_category"].ne("")
    ].copy()

    df["units_sold"] = df["units_sold"].astype(float)
    df["avg_temperature"] = df["avg_temperature"].astype(float)
    return df.sort_values("date").reset_index(drop=True)


data_path = next((p for p in DATA_PATHS if p.exists()), None)

if data_path is None:
    st.error(
        "데이터 파일을 찾을 수 없습니다. "
        "`app.py`와 같은 프로젝트 폴더 안에 `data/clothing_sales_weather.csv`가 있는지 확인해 주세요."
    )
    st.stop()

try:
    df = load_data(str(data_path))
except Exception as exc:
    st.error(f"데이터를 읽는 중 문제가 발생했습니다: {exc}")
    st.stop()

if df.empty:
    st.error("분석할 수 있는 데이터가 없습니다. CSV 파일 내용을 확인해 주세요.")
    st.stop()

df["date_only"] = df["date"].dt.date
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day_of_week"] = df["date"].dt.dayofweek
df["day_of_month"] = df["date"].dt.day
df["is_weekend"] = (df["date"].dt.dayofweek >= 5).astype(int)


# --------------------------------------------------
# 3. 필터
# --------------------------------------------------
st.sidebar.header("🔎 분석 조건")

min_date = df["date"].min().date()
max_date = df["date"].max().date()

date_selection = st.sidebar.date_input(
    "분석 기간",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

if isinstance(date_selection, (tuple, list)) and len(date_selection) == 2:
    start_date, end_date = date_selection
else:
    start_date, end_date = min_date, max_date

if start_date > end_date:
    st.sidebar.warning("시작 날짜가 종료 날짜보다 늦습니다.")
    start_date, end_date = end_date, start_date

selected_categories = st.sidebar.multiselect(
    "의류 종류",
    options=sorted(df["clothing_category"].dropna().unique().tolist()),
    default=sorted(df["clothing_category"].dropna().unique().tolist()),
)

selected_channels = st.sidebar.multiselect(
    "판매 채널",
    options=sorted(df["channel"].dropna().unique().tolist()),
    default=sorted(df["channel"].dropna().unique().tolist()),
)

filtered = df[
    (df["date"].dt.date >= start_date)
    & (df["date"].dt.date <= end_date)
    & (df["clothing_category"].isin(selected_categories))
    & (df["channel"].isin(selected_channels))
].copy()

if filtered.empty:
    st.warning("선택한 조건에 해당하는 데이터가 없습니다. 왼쪽 필터를 변경해 주세요.")
    st.stop()


# --------------------------------------------------
# 4. 핵심 지표
# --------------------------------------------------
total_sales = filtered["units_sold"].sum()
average_daily_sales = (
    filtered.groupby("date")["units_sold"].sum().mean()
    if not filtered.empty else 0
)
average_temperature = filtered["avg_temperature"].mean()
number_of_days = filtered["date"].dt.date.nunique()

c1, c2, c3, c4 = st.columns(4)
c1.metric("총 판매량", f"{total_sales:,.0f}개")
c2.metric("일평균 판매량", f"{average_daily_sales:,.1f}개")
c3.metric("평균 기온", f"{average_temperature:.1f}°C")
c4.metric("분석 날짜 수", f"{number_of_days:,}일")


# --------------------------------------------------
# 5. 판매량 추세 그래프 - 일별 변동 완화
# --------------------------------------------------
st.subheader("📈 날짜별 판매량 추세")
st.caption(
    "얇은 선은 실제 일별 판매량이고, 굵은 선은 최근 7일 이동평균입니다. "
    "이동평균은 하루 단위의 급격한 출렁임을 줄여 전체 흐름을 보기 쉽게 합니다."
)

daily = (
    filtered.groupby("date", as_index=False)
    .agg(
        daily_sales=("units_sold", "sum"),
        avg_temperature=("avg_temperature", "mean"),
    )
    .sort_values("date")
)
daily["sales_7day_avg"] = daily["daily_sales"].rolling(
    window=7, min_periods=1
).mean()

fig_trend = go.Figure()
fig_trend.add_trace(
    go.Scatter(
        x=daily["date"],
        y=daily["daily_sales"],
        mode="lines",
        name="실제 일별 판매량",
        line=dict(width=1.2, color="#A8B3C5"),
        opacity=0.65,
        hovertemplate="%{x|%Y-%m-%d}<br>실제 판매량: %{y:,.0f}개<extra></extra>",
    )
)
fig_trend.add_trace(
    go.Scatter(
        x=daily["date"],
        y=daily["sales_7day_avg"],
        mode="lines",
        name="7일 이동평균",
        line=dict(width=3, color="#2563EB"),
        hovertemplate="%{x|%Y-%m-%d}<br>7일 평균: %{y:,.1f}개<extra></extra>",
    )
)
fig_trend.update_layout(
    template="plotly_white",
    height=390,
    margin=dict(l=10, r=10, t=25, b=10),
    xaxis_title="날짜",
    yaxis_title="판매량 (개)",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
fig_trend.update_xaxes(showgrid=False)
fig_trend.update_yaxes(rangemode="tozero", gridcolor="#E9EDF3")
st.plotly_chart(fig_trend, use_container_width=True)


# --------------------------------------------------
# 6. 기온과 판매량 관계
# --------------------------------------------------
left, right = st.columns(2)

with left:
    st.subheader("🌡️ 기온과 판매량")
    st.caption("각 점은 날짜별 기록입니다. 추세선을 함께 표시해 전체적인 관계를 확인합니다.")

    if len(daily) >= 2:
        corr = daily["avg_temperature"].corr(daily["daily_sales"])
    else:
        corr = np.nan

    fig_scatter = px.scatter(
        daily,
        x="avg_temperature",
        y="daily_sales",
        trendline="ols" if len(daily) >= 3 else None,
        opacity=0.55,
        labels={
            "avg_temperature": "평균 기온 (°C)",
            "daily_sales": "일일 판매량 (개)",
        },
        hover_data={"date": True, "avg_temperature": ":.1f", "daily_sales": ":,.0f"},
    )
    fig_scatter.update_traces(marker=dict(size=7), selector=dict(mode="markers"))
    fig_scatter.update_layout(
        template="plotly_white",
        height=350,
        margin=dict(l=10, r=10, t=20, b=10),
    )
    fig_scatter.update_xaxes(gridcolor="#E9EDF3")
    fig_scatter.update_yaxes(rangemode="tozero", gridcolor="#E9EDF3")
    st.plotly_chart(fig_scatter, use_container_width=True)

    if pd.notna(corr):
        st.metric("기온과 일일 판매량의 상관계수", f"{corr:.3f}")
        st.caption(
            "상관계수는 두 값이 함께 움직이는 정도를 나타냅니다. "
            "상관관계만으로 기온이 판매량의 원인이라고 단정할 수는 없습니다."
        )

with right:
    st.subheader("🧊 기온 구간별 평균 판매량")
    st.caption("기온 구간별로 하루 판매량의 평균을 비교합니다.")

    bins = [-np.inf, 0, 10, 20, 30, np.inf]
    labels = ["0°C 미만", "0~10°C", "10~20°C", "20~30°C", "30°C 이상"]
    daily["temperature_range"] = pd.cut(
        daily["avg_temperature"],
        bins=bins,
        labels=labels,
        right=False,
    )
    temp_summary = (
        daily.groupby("temperature_range", observed=False)
        .agg(
            average_sales=("daily_sales", "mean"),
            days=("date", "count"),
        )
        .reset_index()
    )
    temp_summary = temp_summary[temp_summary["days"] > 0]

    fig_temp = px.bar(
        temp_summary,
        x="temperature_range",
        y="average_sales",
        text=temp_summary["average_sales"].map(lambda v: f"{v:,.0f}개"),
        labels={
            "temperature_range": "기온 구간",
            "average_sales": "평균 일일 판매량 (개)",
        },
        hover_data={"days": True, "average_sales": ":,.1f"},
    )
    fig_temp.update_traces(
        textposition="outside",
        marker_color="#60A5FA",
        cliponaxis=False,
    )
    fig_temp.update_layout(
        template="plotly_white",
        height=350,
        margin=dict(l=10, r=10, t=20, b=10),
        showlegend=False,
    )
    fig_temp.update_yaxes(rangemode="tozero", gridcolor="#E9EDF3")
    st.plotly_chart(fig_temp, use_container_width=True)


# --------------------------------------------------
# 7. 의류 종류별 판매량
# --------------------------------------------------
st.subheader("👕 의류 종류별 판매량")

category_summary = (
    filtered.groupby("clothing_category", as_index=False)
    .agg(total_sales=("units_sold", "sum"))
    .sort_values("total_sales", ascending=True)
)

fig_category = px.bar(
    category_summary,
    x="total_sales",
    y="clothing_category",
    orientation="h",
    text=category_summary["total_sales"].map(lambda v: f"{v:,.0f}개"),
    labels={
        "total_sales": "총 판매량 (개)",
        "clothing_category": "의류 종류",
    },
)
fig_category.update_traces(
    textposition="outside",
    marker_color="#34D399",
    cliponaxis=False,
)
fig_category.update_layout(
    template="plotly_white",
    height=max(260, 75 * len(category_summary)),
    margin=dict(l=10, r=60, t=15, b=10),
    showlegend=False,
)
fig_category.update_xaxes(rangemode="tozero", gridcolor="#E9EDF3")
st.plotly_chart(fig_category, use_container_width=True)


# --------------------------------------------------
# 8. 동적 판매량 예측 모델
# --------------------------------------------------
st.divider()
st.header("🔮 의류 판매량 예측")
st.write(
    "예측할 날짜, 기온, 의류 종류, 판매 채널을 직접 선택하세요. "
    "예측 버튼을 누를 때마다 선택한 조건을 모델에 넣어 판매량을 계산합니다."
)
st.info(
    "예측 결과는 선택한 날짜·의류 종류·판매 채널 조합의 판매량입니다. "
    "학습 데이터가 합성/모의 데이터라면 결과도 실제 시장 판매량을 보장하지 않습니다."
)

FEATURE_COLUMNS = [
    "year", "month", "day_of_week", "day_of_month", "is_weekend",
    "avg_temperature", "channel", "clothing_category",
]
NUMERIC_FEATURES = [
    "year", "month", "day_of_week", "day_of_month",
    "is_weekend", "avg_temperature",
]
CATEGORICAL_FEATURES = ["channel", "clothing_category"]


@st.cache_resource
def train_prediction_model(model_data: pd.DataFrame):
    training = model_data.copy()
    training["date"] = pd.to_datetime(training["date"], errors="coerce")
    training = training.dropna(subset=["date", "units_sold", "avg_temperature", "channel", "clothing_category"]).copy()

    # CSV에는 날짜와 기본 데이터만 있으므로, 모델에 필요한 날짜 파생 변수를 여기서 생성합니다.
    training["year"] = training["date"].dt.year
    training["month"] = training["date"].dt.month
    training["day_of_week"] = training["date"].dt.dayofweek
    training["day_of_month"] = training["date"].dt.day
    training["is_weekend"] = (training["date"].dt.dayofweek >= 5).astype(int)

    training = training.sort_values("date").reset_index(drop=True)

    unique_dates = np.sort(training["date"].dropna().unique())
    if len(unique_dates) < 5:
        raise ValueError("모델 학습을 위해서는 최소 5일 이상의 데이터가 필요합니다.")

    split_index = int(len(unique_dates) * 0.8)
    split_index = min(max(split_index, 1), len(unique_dates) - 1)
    split_date = pd.Timestamp(unique_dates[split_index])

    train_rows = training[training["date"] < split_date].copy()
    test_rows = training[training["date"] >= split_date].copy()

    if train_rows.empty or test_rows.empty:
        raise ValueError("학습/평가 데이터 분리에 실패했습니다.")

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("numeric", "passthrough", NUMERIC_FEATURES),
        ],
        remainder="drop",
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=160,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    pipeline.fit(train_rows[FEATURE_COLUMNS], train_rows["units_sold"])
    test_predictions = pipeline.predict(test_rows[FEATURE_COLUMNS])

    metrics = {
        "mae": float(mean_absolute_error(test_rows["units_sold"], test_predictions)),
        "rmse": float(np.sqrt(mean_squared_error(test_rows["units_sold"], test_predictions))),
        "r2": float(r2_score(test_rows["units_sold"], test_predictions))
        if len(test_rows) >= 2 else np.nan,
        "train_rows": int(len(train_rows)),
        "test_rows": int(len(test_rows)),
        "split_date": split_date.strftime("%Y-%m-%d"),
    }

    # 평가를 마친 후 전체 과거 데이터를 이용해 최종 예측용 모델을 다시 학습합니다.
    final_model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=160,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    final_model.fit(training[FEATURE_COLUMNS], training["units_sold"])

    return final_model, metrics


model_data = df[
    [
        "date", "channel", "clothing_category",
        "units_sold", "avg_temperature",
    ]
].copy()

try:
    prediction_model, model_metrics = train_prediction_model(model_data)
except Exception as exc:
    st.error(f"예측 모델을 학습하지 못했습니다: {exc}")
    st.stop()

categories = sorted(df["clothing_category"].dropna().unique().tolist())
channels = sorted(df["channel"].dropna().unique().tolist())

default_date = max_date
default_temperature = float(round(df["avg_temperature"].median(), 1))

with st.form("prediction_form"):
    st.subheader("예측 조건 입력")

    p1, p2 = st.columns(2)
    with p1:
        prediction_date = st.date_input(
            "예측 날짜",
            value=default_date,
            min_value=pd.Timestamp("2000-01-01").date(),
            max_value=pd.Timestamp("2035-12-31").date(),
            key="prediction_date",
        )
        prediction_category = st.selectbox(
            "의류 종류",
            options=categories,
            key="prediction_category",
        )
    with p2:
        prediction_temperature = st.number_input(
            "예상 평균 기온 (°C)",
            min_value=-30.0,
            max_value=45.0,
            value=default_temperature,
            step=0.5,
            key="prediction_temperature",
            help="예측하려는 날짜의 예상 평균 기온을 입력하세요.",
        )
        prediction_channel = st.selectbox(
            "판매 채널",
            options=channels,
            key="prediction_channel",
        )

    submitted = st.form_submit_button(
        "선택한 조건으로 판매량 예측하기",
        type="primary",
        use_container_width=True,
    )

if submitted:
    chosen_date = pd.Timestamp(prediction_date)

    prediction_input = pd.DataFrame(
        [
            {
                "year": chosen_date.year,
                "month": chosen_date.month,
                "day_of_week": chosen_date.dayofweek,
                "day_of_month": chosen_date.day,
                "is_weekend": int(chosen_date.dayofweek >= 5),
                "avg_temperature": float(prediction_temperature),
                "channel": prediction_channel,
                "clothing_category": prediction_category,
            }
        ]
    )

    predicted_sales = float(prediction_model.predict(prediction_input[FEATURE_COLUMNS])[0])
    predicted_sales = max(0.0, predicted_sales)

    st.success("예측이 완료되었습니다.")
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("예측 판매량", f"{predicted_sales:,.0f}개")
    r2.metric("예측 날짜", chosen_date.strftime("%Y-%m-%d"))
    r3.metric("입력 기온", f"{prediction_temperature:.1f}°C")
    r4.metric("선택한 조건", f"{prediction_category} · {prediction_channel}")

    st.caption(
        "예측값은 학습 데이터에서 모델이 학습한 패턴에 기반한 추정치입니다. "
        "학습 데이터에 없는 요인이나 미래의 급격한 변화는 반영되지 않을 수 있습니다."
    )

with st.expander("📊 예측 모델 성능 확인"):
    m1, m2, m3 = st.columns(3)
    m1.metric("MAE", f"{model_metrics['mae']:.2f}")
    m2.metric("RMSE", f"{model_metrics['rmse']:.2f}")
    m3.metric(
        "R² 점수",
        f"{model_metrics['r2']:.3f}" if pd.notna(model_metrics["r2"]) else "계산 불가",
    )
    st.write(
        f"- 학습 데이터 행 수: {model_metrics['train_rows']:,}\n"
        f"- 평가 데이터 행 수: {model_metrics['test_rows']:,}\n"
        f"- 평가 데이터 시작일: {model_metrics['split_date']}\n"
        "- 평가 방법: 날짜순으로 과거 약 80%를 학습에 사용하고 이후 약 20%를 평가에 사용\n"
        "- MAE/RMSE는 예측 오차 크기이며, 낮을수록 일반적으로 좋습니다. R²는 1에 가까울수록 데이터 설명력이 높습니다."
    )


# --------------------------------------------------
# 9. 데이터 확인 및 다운로드
# --------------------------------------------------
st.divider()
st.subheader("📋 선택 조건의 데이터 미리보기")

display_columns = [
    "date", "channel", "clothing_category", "units_sold", "avg_temperature"
]
preview = filtered[display_columns].sort_values("date", ascending=False)

st.dataframe(
    preview,
    use_container_width=True,
    hide_index=True,
    column_config={
        "date": st.column_config.DateColumn("날짜", format="YYYY-MM-DD"),
        "channel": st.column_config.TextColumn("판매 채널"),
        "clothing_category": st.column_config.TextColumn("의류 종류"),
        "units_sold": st.column_config.NumberColumn("판매량", format="%.0f개"),
        "avg_temperature": st.column_config.NumberColumn("평균 기온", format="%.1f°C"),
    },
)

csv_bytes = preview.to_csv(index=False).encode("utf-8-sig")
st.download_button(
    label="선택한 데이터 CSV 다운로드",
    data=csv_bytes,
    file_name="clothing_sales_filtered.csv",
    mime="text/csv",
)

st.caption(
    "참고: 대시보드의 분석 필터는 그래프와 표에 적용됩니다. "
    "예측 기능은 아래 예측 입력란에서 선택한 조건을 별도로 사용합니다."
)
