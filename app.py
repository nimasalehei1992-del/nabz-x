import streamlit as st
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="NABZ X", page_icon="⚡", layout="wide")

st.markdown("""
<style>
.block-container{direction:rtl}
[data-testid="stSidebar"]{direction:rtl}
.card{padding:18px;border:1px solid #ddd;border-radius:14px;margin-bottom:10px}
.big{font-size:30px;font-weight:800}
.demo{padding:10px;border-radius:10px;background:#fff4e5}
</style>
""", unsafe_allow_html=True)

st.title("نبض ایکس | NABZ X")
st.caption("موتور هوشمند پیش‌بینی و مداخله پیشگیرانه ریسک اعتباری")
st.markdown('<div class="demo">نسخه Prototype — داده‌ها مصنوعی و فقط برای نمایش منطق محصول هستند.</div>', unsafe_allow_html=True)

@st.cache_data
def load_data():
    return pd.read_csv("nabz_x_synthetic_credit_risk_dataset.csv")

@st.cache_resource
def train(df):
    features = [
        "income_m_toman","account_turnover_m_toman","loan_amount_m_toman",
        "number_of_loans","late_payment_count","max_delay_days",
        "monthly_cashflow_m_toman","existing_obligations_m_toman",
        "credit_history_years","recent_balance_change_pct"
    ]
    X, y = df[features], df["default_12m"]
    model = Pipeline([
        ("scale", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000, class_weight="balanced"))
    ])
    model.fit(X, y)
    return model, features

df = load_data()
model, features = train(df)

st.sidebar.header("تحلیل مشتری")
cid = st.sidebar.selectbox("شناسه مشتری", df["customer_id"].astype(str).tolist())
row = df[df["customer_id"].astype(str)==cid].iloc[0].copy()

st.sidebar.subheader("شبیه‌سازی تغییر رفتار")
late = st.sidebar.slider("تأخیر اضافه", 0, 5, 0)
cash = st.sidebar.slider("تغییر جریان نقدی (%)", -50, 30, 0)
oblig = st.sidebar.slider("افزایش تعهدات (میلیون تومان)", 0, 500, 0)

row["late_payment_count"] += late
row["monthly_cashflow_m_toman"] *= (1 + cash/100)
row["existing_obligations_m_toman"] += oblig

X = pd.DataFrame([[row[f] for f in features]], columns=features)
prob = float(model.predict_proba(X)[0,1])
score = round(prob*100,1)

if score >= 70:
    level, cls = "پرریسک", "🔴"
elif score >= 40:
    level, cls = "متوسط", "🟠"
else:
    level, cls = "کم‌ریسک", "🟢"

a,b,c,d = st.columns(4)
a.metric("شناسه مشتری", cid)
b.metric("Risk Score", score)
c.metric("احتمال نکول ۱۲ ماهه", f"{prob:.1%}")
d.metric("سطح ریسک", f"{cls} {level}")

st.subheader("چرا ریسک تغییر کرده؟")
coef = model.named_steps["model"].coef_[0]
scaler = model.named_steps["scale"]
items=[]
labels={
"income_m_toman":"درآمد","account_turnover_m_toman":"گردش حساب",
"loan_amount_m_toman":"مبلغ تسهیلات","number_of_loans":"تعداد تسهیلات",
"late_payment_count":"تعداد تأخیر پرداخت","max_delay_days":"بیشترین روز تأخیر",
"monthly_cashflow_m_toman":"جریان نقدی ماهانه",
"existing_obligations_m_toman":"تعهدات موجود",
"credit_history_years":"سابقه اعتباری",
"recent_balance_change_pct":"تغییر اخیر مانده حساب"}
for i,f in enumerate(features):
    z=(X.iloc[0,i]-scaler.mean_[i])/scaler.scale_[i]
    items.append((f,z*coef[i]))
for f,v in sorted(items,key=lambda x:abs(x[1]),reverse=True)[:4]:
    st.write(f"• **{labels[f]}** → {'افزایش ریسک' if v>0 else 'کاهش ریسک'}")

st.subheader("Early Warning و اقدام پیشنهادی")
if score >= 70:
    st.error("هشدار: افزایش ریسک نیازمند بررسی اعتباری است.")
    st.info("پیشنهاد: بررسی پرونده، وضعیت تعهدات و تماس پیشگیرانه با مشتری.")
elif score >= 40:
    st.warning("هشدار: نشانه‌های افزایش ریسک مشاهده شده است.")
    st.info("پیشنهاد: افزایش دفعات پایش و بررسی جریان نقدی و تعهدات.")
else:
    st.success("ریسک در محدوده پایین قرار دارد.")
    st.info("پیشنهاد: ادامه پایش دوره‌ای.")

with st.expander("جزئیات داده مشتری"):
    st.dataframe(row.to_frame("مقدار"), use_container_width=True)

st.caption("نبض ایکس تصمیم نهایی اعتباری را جایگزین نمی‌کند؛ یک Decision Support برای شناسایی زودهنگام ریسک است.")
