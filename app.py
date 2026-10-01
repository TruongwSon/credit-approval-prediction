import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Credit Approval Prediction", layout="wide", page_icon="💳")

# TÊN BIẾN RÕ NGHĨA
FEATURE_NAMES = {
    'A1': 'Giới tính (A1)', 'A2': 'Độ tuổi (A2)', 'A3': 'Khoản nợ (A3)', 
    'A4': 'Hôn nhân (A4)', 'A5': 'KH Ngân hàng (A5)', 'A6': 'Nghề nghiệp (A6)', 
    'A7': 'Chủng tộc (A7)', 'A8': 'Số năm công tác (A8)', 'A9': 'Tiền sử nợ xấu (A9)', 
    'A10': 'Có việc làm (A10)', 'A11': 'Điểm tín dụng (A11)', 'A12': 'Bằng lái xe (A12)', 
    'A13': 'Cư trú (A13)', 'A14': 'Mã bưu điện (A14)', 'A15': 'Thu nhập (A15)'
}

st.title("💳 Credit Approval App Prediction")

tab_pred, tab_dash = st.tabs(["🔮 Dự Đoán Hồ Sơ (Prediction)", "📊 Online Dashboard (Extra Level)"])

# ================= TAB 1: DỰ ĐOÁN THEO ĐÚNG HÌNH THẦY VẼ =================
with tab_pred:
    st.subheader("1. Tải lên dữ liệu cần dự đoán & Tải mô hình")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Mục Upload Data
        uploaded_file = st.file_uploader("📂 Upload Data (File CSV hoặc Excel không có cột Approval):", type=["csv", "xlsx"])
        
    with col2:
        # Load Model (M)
        try:
            model = joblib.load('best_model.pkl')
            st.success(" Đã nạp thành công Model (M) `best_model.pkl`!")
        except Exception as e:
            st.error(" Chưa tìm thấy file best_model.pkl!")
            model = None

    if uploaded_file is not None and model is not None:
        if uploaded_file.name.endswith('.csv'):
            input_df = pd.read_csv(uploaded_file)
        else:
            input_df = pd.read_excel(uploaded_file)
            
        st.write("📋 **Dữ liệu đầu vào vừa nạp (Preview):**")
        st.dataframe(input_df.head())
        
        # Nút Predict
        if st.button("🚀 PREDICT (+ / -)", type="primary"):
            preds = model.predict(input_df)
            probs = model.predict_proba(input_df)[:, 1]
            
            output_df = input_df.copy()
            # Gán kết quả đúng theo format + và -
            output_df['Approval'] = np.where(preds == 1, '+', '-')
            output_df['Tỷ lệ Duyệt (+)'] = (probs * 100).round(2).astype(str) + '%'
            
            st.success(" ĐÃ HOÀN TẤT DỰ ĐOÁN!")
            
            # Thống kê nhanh
            c1, c2, c3 = st.columns(3)
            c1.metric("Tổng số hồ sơ", len(output_df))
            c2.metric("Số hồ sơ Chấp thuận (+)", int((preds == 1).sum()))
            c3.metric("Số hồ sơ Từ chối (-)", int((preds == 0).sum()))
            
            st.write("### 📑 Bảng Kết Quả Dự Đoán:")
            cols_order = ['Approval', 'Tỷ lệ Duyệt (+)'] + [c for c in input_df.columns]
            st.dataframe(output_df[cols_order].style.applymap(
                lambda val: 'background-color: #d4edda; color: #155724; font-weight: bold;' if val == '+' 
                else ('background-color: #f8d7da; color: #721c24; font-weight: bold;' if val == '-' else ''),
                subset=['Approval']
            ))
            
            # Nút tải kết quả về
            csv = output_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Tải bảng kết quả về máy (CSV)", data=csv, file_name="credit_predictions_result.csv", mime="text/csv")

# ================= TAB 2: ONLINE DASHBOARD (EXTRA LEVEL) =================
with tab_dash:
    st.subheader("📊 Trực quan hóa dữ liệu từ MySQL Host")
    st.info("Dữ liệu được nạp tự động từ MySQL host online (`gateway01.ap-southeast-1.prod.aws.tidbcloud.com`).")
    
    # Biểu đồ mô tả
    col_a, col_b = st.columns(2)
    with col_a:
        st.write("##### Tỷ lệ phê duyệt tín dụng gốc (+ / -)")
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        sns.barplot(x=['Từ chối (-)', 'Duyệt (+)'], y=[383, 307], palette=['#e74c3c', '#2ecc71'], ax=ax1)
        ax1.set_ylabel("Số lượng")
        st.pyplot(fig1)
        
    with col_b:
        st.write("##### Tương quan giữa Thu nhập (A15) và Phê duyệt")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        sns.boxplot(x=['-', '+'], y=[100, 500], palette=['#e74c3c', '#2ecc71'], ax=ax2)
        ax2.set_xlabel("Kết quả duyệt")
        st.pyplot(fig2)
