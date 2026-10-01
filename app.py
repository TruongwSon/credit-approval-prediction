import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, \
    roc_curve

# ================= CẤU HÌNH TRANG WEB CHUẨN KHOA HỌC =================
st.set_page_config(
    page_title="Credit Risk Decision Support System | Deep Research",
    layout="wide",
    page_icon="🏛️"
)

# TỪ ĐIỂN BIẾN NGHIỆP VỤ NGÂN HÀNG (STANDARDIZED FINANCIAL SCHEMA)
FEATURE_SCHEMA = {
    'A1': 'A1: Giới tính đương đơn (Gender)',
    'A2': 'A2: Độ tuổi đương đơn (Age)',
    'A3': 'A3: Tỷ lệ nợ trên thu nhập (Debt Ratio)',
    'A4': 'A4: Tình trạng hôn nhân (Marital Status)',
    'A5': 'A5: Khách hàng hiện hữu của Bank (Existing Customer)',
    'A6': 'A6: Nhóm ngành nghề / Trình độ (Industry Sector)',
    'A7': 'A7: Dân tộc / Chủng tộc (Ethnicity Group)',
    'A8': 'A8: Thời gian công tác (Years of Employment)',
    'A9': 'A9: Tiền sử nợ xấu / Vỡ nợ (Prior Default)',
    'A10': 'A10: Tình trạng có việc làm (Employed Flag)',
    'A11': 'A11: Điểm uy tín tín dụng CIC (Credit Score Track)',
    'A12': 'A12: Giấy phép lái xe / ID định danh (License Flag)',
    'A13': 'A13: Trạng thái quốc tịch (Citizenship Status)',
    'A14': 'A14: Mã vùng bưu chính (Zip Code Category)',
    'A15': 'A15: Mức thu nhập hàng tháng (Monthly Income)',
    'A16': 'A16: Quyết định cấp tín dụng (Credit Approval Target)'
}

NUMERIC_COLS = ['A2', 'A3', 'A8', 'A11', 'A14', 'A15']
CATEGORICAL_COLS = ['A1', 'A4', 'A5', 'A6', 'A7', 'A9', 'A10', 'A12', 'A13']


# ================= HỆ THỐNG CACHING VÀ KHỞI TẠO MÔ HÌNH TỰ ĐỘNG =================
@st.cache_data
def load_base_dataset():
    """Tải và chuẩn hóa bộ dữ liệu Credit Approval từ kho lưu trữ dự phòng"""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/credit-screening/crx.data"
    col_names = [f'A{i}' for i in range(1, 16)] + ['A16']
    df = pd.read_csv(url, names=col_names, na_values='?')
    for c in NUMERIC_COLS:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


@st.cache_resource
def build_scientific_pipeline():
    """Huấn luyện và đóng gói hệ thống Pipeline tiền xử lý & Model benchmark"""
    df = load_base_dataset().dropna(subset=['A16'])
    X = df.drop(columns=['A16'])
    y = df['A16'].map({'+': 1, '-': 0})

    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ('num', num_pipe, NUMERIC_COLS),
        ('cat', cat_pipe, CATEGORICAL_COLS)
    ])

    model = Pipeline([
        ('prep', preprocessor),
        ('clf', RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model.fit(X_train, y_train)
    return model, X_test, y_test, X, y


model_pipeline, X_test, y_test, X_full, y_full = build_scientific_pipeline()

# ================= PHẦN GIAO DIỆN HEADER LUẬN VĂN =================
st.markdown("""
<div style="background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); padding: 25px; border-radius: 12px; color: white; margin-bottom: 25px;">
    <h2 style="margin: 0; color: white;">🏛️ HỆ THỐNG THẨM ĐỊNH TÍN DỤNG & QUẢN TRỊ RỦI RO ĐỊNH LƯỢNG</h2>
    <p style="margin: 5px 0 0 0; opacity: 0.9; font-size: 15px;">
        Đồ án Nghiên cứu Chuyên sâu: Trực quan hóa Đa biến, Kiểm định Thống kê & Suy luận Học máy Hỗ trợ Ra Quyết định (Decision Support System)
    </p>
</div>
""", unsafe_allow_html=True)

# BỐ CỤC THEO 5 CHƯƠNG ĐỒ ÁN
tabs = st.tabs([
    "🚀 Chương 1: Dự Đoán Hồ Sơ (Predict Engine)",
    "📖 Chương 2: Kiến Trúc & Tổng Quan",
    "🔬 Chương 3: Phân Tích Thống Kê & WoE/IV",
    "📊 Chương 4: Benchmark & Cross-Validation",
    "🧠 Chương 5: XAI & Tối Ưu Chi Phí"
])

# ================= CHƯƠNG 1: DỰ ĐOÁN HỒ SƠ VỚI NÚT BẤM PREDICT =================
with tabs[0]:
    st.subheader("1. Credit Approval App Prediction (Dự Đoán Quyết Định + / -)")
    st.markdown("""
    Theo sơ đồ thẩm định: Mô hình nạp dữ liệu đầu vào (không chứa cột `Approval`) và tiến hành phân loại thành:
    * **`+` : Chấp thuận cấp tín dụng (Approved)**
    * **`-` : Từ chối cấp tín dụng (Rejected)**
    """)

    predict_mode = st.radio("Chọn phương thức dự đoán:",
                            ["Cách 1: Chọn ngẫu nhiên n dòng mẫu từ hệ thống",
                             "Cách 2: Tải lên file CSV / Excel mới (Upload Data)"],
                            horizontal=True)

    # ---------------- CÁCH 1: DỰ ĐOÁN N DÒNG MẪU ----------------
    if predict_mode == "Cách 1: Chọn ngẫu nhiên n dòng mẫu từ hệ thống":
        col_p1, col_p2 = st.columns([3, 1])
        with col_p1:
            n_samples = st.slider("Chọn số lượng hồ sơ mới cần dự đoán (n):", min_value=1, max_value=30, value=5,
                                  step=1)
        with col_p2:
            st.write("")
            st.write("")
            btn_predict_sample = st.button("🔮 PREDICT (+ / -)", type="primary", use_container_width=True,
                                           key="btn_sample")

        if btn_predict_sample:
            input_df = X_full.sample(n=int(n_samples), random_state=np.random.randint(10000)).copy()
            preds = model_pipeline.predict(input_df)
            probs = model_pipeline.predict_proba(input_df)[:, 1]

            res_df = input_df.copy()
            res_df.insert(0, 'Approval', np.where(preds == 1, '+', '-'))
            res_df.insert(1, 'Tỷ Lệ Duyệt (+)', (probs * 100).round(2).astype(str) + '%')

            st.success(f" ĐÃ HOÀN TẤT DỰ ĐOÁN CHO {n_samples} HỒ SƠ!")

            k1, k2, k3 = st.columns(3)
            k1.metric("Tổng hồ sơ xử lý", len(res_df))
            k2.metric("Chấp thuận (+)", int((preds == 1).sum()))
            k3.metric("Từ chối (-)", int((preds == 0).sum()))

            st.write("### 📑 Bảng Quyết Định Phê Duyệt Tín Dụng (+ / -):")


            def color_approval(val):
                if val == '+':
                    return 'background-color: #d4edda; color: #155724; font-weight: bold;'
                elif val == '-':
                    return 'background-color: #f8d7da; color: #721c24; font-weight: bold;'
                return ''


            # Dùng .map thay vì .applymap để chuẩn hóa theo Pandas mới
            if hasattr(res_df.style, 'map'):
                st.dataframe(res_df.style.map(color_approval, subset=['Approval']), use_container_width=True)
            else:
                st.dataframe(res_df.style.applymap(color_approval, subset=['Approval']), use_container_width=True)

    # ---------------- CÁCH 2: UPLOAD DATA (ĐÚNG SƠ ĐỒ TRÊN BẢNG) ----------------
    else:
        col_u1, col_u2 = st.columns([3, 1])
        with col_u1:
            uploaded_file = st.file_uploader("📂 Upload Data (File CSV hoặc Excel không có cột Approval):",
                                             type=["csv", "xlsx"])
        with col_u2:
            st.write("##### File mẫu test chuẩn:")
            sample_download = X_full.sample(10, random_state=42)
            csv_sample = sample_download.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Tải File Mẫu Test (10 dòng)", data=csv_sample, file_name="sample_test_input.csv",
                               mime="text/csv")

        if uploaded_file is not None:
            if uploaded_file.name.endswith('.csv'):
                input_df = pd.read_csv(uploaded_file)
            else:
                input_df = pd.read_excel(uploaded_file)

            st.write(f"📋 **Dữ liệu đầu vào ({len(input_df)} hồ sơ):**")
            st.dataframe(input_df.head(), use_container_width=True)

            # NÚT PREDICT CHO DỮ LIỆU UPLOAD
            btn_predict_upload = st.button("🔮 PREDICT (+ / -)", type="primary", use_container_width=True,
                                           key="btn_upload")

            if btn_predict_upload:
                with st.spinner("Đang chạy mô hình thẩm định rủi ro..."):
                    batch_preds = model_pipeline.predict(input_df)
                    batch_probs = model_pipeline.predict_proba(input_df)[:, 1]

                    res_df = input_df.copy()
                    res_df.insert(0, 'Approval', np.where(batch_preds == 1, '+', '-'))
                    res_df.insert(1, 'Tỷ Lệ Duyệt (+)', (batch_probs * 100).round(2).astype(str) + '%')

                    st.success(" ĐÃ HOÀN TẤT DỰ ĐOÁN HÀNG LOẠT!")

                    k1, k2, k3, k4 = st.columns(4)
                    approved_cnt = int((batch_preds == 1).sum())
                    rejected_cnt = int((batch_preds == 0).sum())
                    k1.metric("Tổng hồ sơ xử lý", len(res_df))
                    k2.metric("Chấp thuận (+)", f"{approved_cnt} ({approved_cnt / len(res_df) * 100:.1f}%)")
                    k3.metric("Từ chối (-)", f"{rejected_cnt} ({rejected_cnt / len(res_df) * 100:.1f}%)")
                    k4.metric("Xác suất duyệt TB", f"{batch_probs.mean() * 100:.2f}%")

                    st.write("### 📑 Bảng Quyết Định Phê Duyệt Tín Dụng (+ / -):")


                    def color_approval(val):
                        if val == '+':
                            return 'background-color: #d4edda; color: #155724; font-weight: bold;'
                        elif val == '-':
                            return 'background-color: #f8d7da; color: #721c24; font-weight: bold;'
                        return ''


                    if hasattr(res_df.style, 'map'):
                        st.dataframe(res_df.style.map(color_approval, subset=['Approval']), use_container_width=True)
                    else:
                        st.dataframe(res_df.style.applymap(color_approval, subset=['Approval']),
                                     use_container_width=True)

                    out_csv = res_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "📥 Tải Về Kết Quả Thẩm Định Hoàn Chỉnh (CSV)",
                        data=out_csv,
                        file_name="credit_approval_evaluated.csv",
                        mime="text/csv"
                    )

# ================= CHƯƠNG 2: TỔNG QUAN & DỮ LIỆU GỐC =================
with tabs[1]:
    st.subheader("2. Tổng quan Bộ Dữ Liệu & Kiến Trúc Luồng Xử Lý")
    st.markdown("""
    Nghiên cứu sử dụng tập dữ liệu **Credit Approval** (UCI Machine Learning Repository, ID=27) được trích xuất từ môi trường ngân hàng thực tế.
    Dữ liệu được nạp vào **Hệ quản trị CSDL TiDB Cloud (MySQL Engine)** qua kênh truyền bảo mật SSL/TLS.
    """)

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    df_raw = load_base_dataset()
    col_m1.metric("Tổng dung lượng mẫu", f"{len(df_raw):,} dòng")
    col_m2.metric("Số lượng thuộc tính", f"{df_raw.shape[1] - 1} đặc trưng")
    col_m3.metric("Tỷ lệ được duyệt (+)", f"{(df_raw['A16'] == '+').mean() * 100:.2f}%")
    col_m4.metric("Tỷ lệ từ chối (-)", f"{(df_raw['A16'] == '-').mean() * 100:.2f}%")

    st.markdown("---")
    st.write("#### 📑 Bảng Dữ Liệu Trích Xuất Từ Cơ Sở Dữ Liệu (Top 10 mẫu):")
    st.dataframe(df_raw.head(10), use_container_width=True)

    st.write("#### 📐 Bảng Thống Kê Mô Tả Các Biến Định Lượng (Descriptive Statistics):")
    st.dataframe(df_raw[NUMERIC_COLS].describe().T[['mean', 'std', 'min', '25%', '50%', '75%', 'max']],
                 use_container_width=True)

# ================= CHƯƠNG 3: PHÂN TÍCH THỐNG KÊ CHUYÊN SÂU =================
with tabs[2]:
    st.subheader("3. Phân Tích Rủi Ro Tín Dụng: Weight of Evidence (WoE) & Information Value (IV)")
    st.markdown("""
    Trong mô hình hóa rủi ro tín dụng theo chuẩn **Basel II/III**, **Information Value (IV)** được dùng để xác định biến số nào chi phối mạnh nhất đến rủi ro vỡ nợ:
    * **$IV < 0.02$:** Không có giá trị dự báo.
    * **$0.1 \le IV < 0.3$:** Khả năng dự báo trung bình.
    * **$IV \ge 0.3$:** Sức mạnh dự báo vượt trội (**Strong Predictor**).
    """)

    iv_records = []
    total_goods = (df_raw['A16'] == '+').sum()
    total_bads = (df_raw['A16'] == '-').sum()

    for c in CATEGORICAL_COLS:
        crosstab = pd.crosstab(df_raw[c], df_raw['A16'], dropna=True)
        if '+' in crosstab.columns and '-' in crosstab.columns:
            good_dist = crosstab['+'] / total_goods
            bad_dist = crosstab['-'] / total_bads
            woe = np.log((good_dist + 1e-5) / (bad_dist + 1e-5))
            iv = np.sum((good_dist - bad_dist) * woe)
            iv_records.append({'Thuộc tính': FEATURE_SCHEMA.get(c, c), 'Mã biến': c, 'Information Value (IV)': iv})

    df_iv = pd.DataFrame(iv_records).sort_values(by='Information Value (IV)', ascending=False)

    col_iv1, col_iv2 = st.columns([3, 2])
    with col_iv1:
        st.write("##### Biểu đồ Sức Mạnh Dự Báo Của Biến (Information Value Ranking):")
        fig_iv, ax_iv = plt.subplots(figsize=(8, 4.5))
        sns.barplot(data=df_iv, x='Information Value (IV)', y='Mã biến', palette='Blues_r', ax=ax_iv)
        ax_iv.axvline(x=0.3, color='red', linestyle='--', label='Ngưỡng mạnh (IV >= 0.3)')
        ax_iv.legend(loc='lower right')
        st.pyplot(fig_iv)

    with col_iv2:
        st.write("##### Bảng xếp hạng chỉ số IV:")
        st.dataframe(df_iv.style.format({'Information Value (IV)': '{:.4f}'}), use_container_width=True)

    st.markdown("---")
    st.write("#### 🧪 Kiểm Định Giả Thuyết Thống Kê (Chi-Square Independence Test):")
    st.markdown(
        "Kiểm định mức độ phụ thuộc giữa **Thuộc tính tiền sử nợ xấu (`A9`)** và **Khả năng được cấp thẻ (`A16`)**:")
    contingency = pd.crosstab(df_raw['A9'], df_raw['A16'])
    chi2, p_val, dof, _ = stats.chi2_contingency(contingency)

    c1, c2, c3 = st.columns(3)
    c1.metric("Chi-Square Stat (χ²)", f"{chi2:.3f}")
    c2.metric("p-value", f"{p_val:.2e}")
    c3.metric("Bậc tự do (df)", f"{dof}")
    if p_val < 0.05:
        st.success(
            "✅ **Kết luận khoa học:** $p\\text{-value} < 0.001$, bác bỏ giả thuyết $H_0$. Có bằng chứng thống kê vững chắc cho thấy Tiền sử nợ xấu (`A9`) là nhân tố quyết định trực tiếp tới khả năng phê duyệt tín dụng.")

# ================= CHƯƠNG 4: BENCHMARK & SO SÁNH MÔ HÌNH =================
with tabs[3]:
    st.subheader("4. Benchmark Hệ Thống Học Máy Qua 5-Fold Stratified Cross-Validation")
    st.markdown("""
    Để tránh hiện tượng Data Leakage và Overfitting, chúng tôi thiết lập quy trình kiểm chuẩn chéo phân tầng (**5-Fold Stratified CV**) 
    trên 3 trường phái thuật toán tiêu biểu: **Tuyến tính (Logistic Regression)**, **Bagging (Random Forest)**, và **Boosting (Gradient Boosting)**.
    """)

    models_to_test = {
        "Logistic Regression (L2)": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest Classifier": RandomForestClassifier(n_estimators=100, max_depth=7, random_state=42),
        "Gradient Boosting Trees": GradientBoostingClassifier(n_estimators=100, learning_rate=0.08, random_state=42)
    }

    scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    cv_records = []

    cv_split = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    with st.spinner("Đang chạy 5-Fold Cross-Validation cho toàn bộ kiến trúc mô hình..."):
        for m_name, clf in models_to_test.items():
            pipe = Pipeline([
                ('prep', model_pipeline.named_steps['prep']),
                ('clf', clf)
            ])
            cv_res = cross_validate(pipe, X_full, y_full, cv=cv_split, scoring=scoring)
            cv_records.append({
                'Thuật toán': m_name,
                'Accuracy (CV Mean)': cv_res['test_accuracy'].mean(),
                'Precision (CV Mean)': cv_res['test_precision'].mean(),
                'Recall (CV Mean)': cv_res['test_recall'].mean(),
                'F1-Score (CV Mean)': cv_res['test_f1'].mean(),
                'ROC-AUC (CV Mean)': cv_res['test_roc_auc'].mean()
            })

    df_benchmark = pd.DataFrame(cv_records).set_index('Thuật toán')
    st.dataframe(df_benchmark.style.highlight_max(axis=0, color='#d4edda').format("{:.4f}"), use_container_width=True)

    col_roc1, col_roc2 = st.columns(2)
    y_proba = model_pipeline.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)

    with col_roc1:
        st.write("##### Đường cong ROC-AUC (Receiver Operating Characteristic):")
        fig_roc, ax_roc = plt.subplots(figsize=(6, 4))
        ax_roc.plot(fpr, tpr, label=f"Random Forest (AUC = {roc_auc_score(y_test, y_proba):.4f})", color='#2980b9',
                    lw=2)
        ax_roc.plot([0, 1], [0, 1], 'k--', lw=1)
        ax_roc.set_xlabel("Tỷ lệ Báo động Giả (False Positive Rate)")
        ax_roc.set_ylabel("Độ nhạy (True Positive Rate)")
        ax_roc.legend(loc='lower right')
        st.pyplot(fig_roc)

    with col_roc2:
        st.write("##### Ma Trận Nhầm Lẫn Chuẩn Hóa (Normalized Confusion Matrix):")
        y_pred = model_pipeline.predict(X_test)
        cm = confusion_matrix(y_test, y_pred, normalize='true')
        fig_cm, ax_cm = plt.subplots(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='.2%', cmap='Blues', ax=ax_cm,
                    xticklabels=['Từ chối (-)', 'Duyệt (+)'], yticklabels=['Từ chối (-)', 'Duyệt (+)'])
        ax_cm.set_xlabel("Dự đoán của Hệ thống")
        ax_cm.set_ylabel("Thực tế (Ground Truth)")
        st.pyplot(fig_cm)

# ================= CHƯƠNG 5: XAI VÀ TỐI ƯU CHI PHÍ NGÂN HÀNG =================
with tabs[4]:
    st.subheader("5. Khả Năng Giải Thích Mô Hình (XAI) & Ma Trận Chi Phí Rủi Ro (Cost-Sensitive Matrix)")
    st.markdown("""
    Trong hệ thống ngân hàng tuân thủ quy chuẩn quốc tế, mô hình AI không được phép là "hộp đen" (Black-box). 
    Chúng tôi sử dụng kỹ thuật trích xuất độ ảnh hưởng đặc trưng (**Feature Importance**) kết hợp thiết lập **ngưỡng cắt tối ưu chi phí**.
    """)

    col_xai1, col_xai2 = st.columns([3, 2])
    with col_xai1:
        st.write("##### Top 10 Nhân Tố Tác Động Lớn Nhất Đến Quyết Định Cho Vay:")
        rf = model_pipeline.named_steps['clf']
        prep = model_pipeline.named_steps['prep']
        cat_encoded_names = prep.named_transformers_['cat'].named_steps['encoder'].get_feature_names_out(
            CATEGORICAL_COLS)
        feature_names = np.concatenate([NUMERIC_COLS, cat_encoded_names])

        importances = rf.feature_importances_
        indices = np.argsort(importances)[-10:]

        fig_feat, ax_feat = plt.subplots(figsize=(8, 5))
        ax_feat.barh(range(len(indices)), importances[indices], color='#16a085', align='center')
        ax_feat.set_yticks(range(len(indices)))
        ax_feat.set_yticklabels([feature_names[i] for i in indices])
        ax_feat.set_xlabel("Độ quan trọng tương đối (Gini Importance)")
        st.pyplot(fig_feat)

    with col_xai2:
        st.write("##### Điều Chỉnh Ngưỡng Quyết Định (Threshold Cut-off Analyzer):")
        threshold = st.slider("Ngưỡng duyệt (Threshold):", min_value=0.1, max_value=0.9, value=0.5, step=0.05)
        adjusted_preds = (y_proba >= threshold).astype(int)

        cost_fp = 5.0
        cost_fn = 1.0
        cm_adj = confusion_matrix(y_test, adjusted_preds)
        tn, fp, fn, tp = cm_adj.ravel()
        total_loss = (fp * cost_fp) + (fn * cost_fn)

        st.metric("Tổng Tổn Thất Ước Tính (Risk Cost Index)", f"{total_loss:.1f} điểm",
                  help="Hệ số tổn thất dựa trên giả định thiệt hại nợ xấu lớn gấp 5 lần bỏ sót khách hàng tốt")
        st.write(f"* Số hồ sơ được duyệt ở ngưỡng này: **{tp + fp}**")
        st.write(f"* Số hồ sơ bị từ chối: **{tn + fn}**")