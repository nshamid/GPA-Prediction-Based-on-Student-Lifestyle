import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# --- Kofigurasi Halaman ---
st.set_page_config(
    page_title="GPA Prediction Based on Student Lifestyle",
    page_icon="🎓",
    layout="wide"
)

# --- Load Data & Train Model (Cached) ---
@st.cache_resource
def load_data_and_train_model():
    try:
        # Load Dataset
        df = pd.read_csv('student_lifestyle_dataset.csv')
        
        # Preprocessing
        # Encode Stress Level: Low->0, Moderate->1, High->2
        encode_stress = {'Low': 0, 'Moderate': 1, 'High': 2}
        df['Stress_Level_Encoded'] = df['Stress_Level'].map(encode_stress)
        
        # Definisikan Fitur (X) dan Target (y)
        features = ['Study_Hours_Per_Day', 'Extracurricular_Hours_Per_Day', 
                    'Sleep_Hours_Per_Day', 'Social_Hours_Per_Day', 
                    'Physical_Activity_Hours_Per_Day', 'Stress_Level_Encoded']
        target = 'GPA'
        
        X = df[features]
        y = df[target]
        
        # Split Data (80% Train, 20% Test)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
        
        # Train Model (GANTI KE LINEAR REGRESSION)
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Evaluasi
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        # Simpan Koefisien untuk Visualisasi
        coef_df = pd.DataFrame({
            'Fitur': features,
            'Koefisien': model.coef_
        }).sort_values(by='Koefisien', ascending=False)
        
        return df, model, r2, mae, rmse, X_test, y_test, y_pred, coef_df
        
    except FileNotFoundError:
        return None, None, None, None, None, None, None, None, None

# Load resources
data_load = load_data_and_train_model()
if data_load[0] is not None:
    df, model, r2, mae, rmse, X_test, y_test, y_pred, coef_df = data_load
else:
    df = None

# --- Sidebar Navigasi ---
st.sidebar.title("Navigasi")
page = st.sidebar.radio("Pilih Halaman:", ["Informasi Dataset", "Proses & Prediksi", "Credit"])

st.sidebar.markdown("---")
st.sidebar.caption("© 2025 - TI Bilingual P1 '23")

# ==========================================
# PAGE 1: INFORMASI DATASET
# ==========================================
if page == "Informasi Dataset":
    st.title("🎓 Student Lifestyle Dataset Analysis")
    
    tab1, tab2 = st.tabs(["📂 Tentang Dataset", "🤖 Model & Tujuan"])
    
    with tab1:
        st.markdown("""
        ### Informasi Dataset
        
        **Link Dataset:** [Kaggle - Student Lifestyle Dataset](https://www.kaggle.com/datasets/steve1215rogg/student-lifestyle-dataset)
        
        Dataset ini menganalisis hubungan antara pola gaya hidup siswa (seperti jam belajar, tidur, dan aktivitas sosial) 
        dengan kinerja akademik mereka (**GPA**). Dataset terdiri dari **2.000 sampel** data.
        
        | Fitur (Atribut) | Keterangan |
        | :--- | :--- |
        | **Study_Hours_Per_Day** | Jam belajar per hari |
        | **Extracurricular_Hours_Per_Day** | Jam kegiatan ekstrakurikuler |
        | **Sleep_Hours_Per_Day** | Jam tidur per hari |
        | **Social_Hours_Per_Day** | Jam bersosialisasi |
        | **Physical_Activity_Hours_Per_Day** | Jam aktivitas fisik/olahraga |
        | **Stress_Level** | Tingkat stres (Low, Moderate, High) |
        | **GPA** | **Target**: Indeks Prestasi Kumulatif (0.0 - 4.0) |
        """)
        
        if df is not None:
            st.write("### Sampel Data:")
            st.dataframe(df.head())
        else:
            st.error("File 'student_lifestyle_dataset.csv' tidak ditemukan.")

    with tab2:
        st.header("Model Machine Learning")
        st.info("""
        **Algoritma:** Linear Regression
        
        Kami memilih **Linear Regression** karena analisis menunjukkan hubungan yang linear antara jam belajar dan GPA. 
        Model ini juga memudahkan kita untuk melihat faktor mana yang memberikan dampak positif atau negatif terhadap nilai.
        """)
        
        st.markdown("""
        ### Mengapa GPA?
        Fokus utama adalah memprediksi performa akademik (GPA) untuk membantu siswa menyeimbangkan gaya hidup mereka.
        """)

# ==========================================
# PAGE 2: PROSES & PREDIKSI
# ==========================================
elif page == "Proses & Prediksi":
    st.title("⚙️ Pengerjaan & Aplikasi")
    
    if df is None:
        st.error("Data tidak ditemukan. Pastikan file CSV sudah ada.")
        st.stop()
        
    tab1, tab2 = st.tabs(["📊 Evaluasi & Insight", "🧮 Kalkulator Prediksi GPA"])
    
    with tab1:
        st.header("Evaluasi Model Linear Regression")
        
        # 1. Metrik Utama
        col_metric1, col_metric2, col_metric3 = st.columns(3)
        col_metric1.metric("R2 Score (Akurasi)", f"{r2:.2%}", help="Seberapa baik model menjelaskan variasi data (Mendekati 100% lebih baik)")
        col_metric2.metric("MAE (Rata-rata Error)", f"{mae:.3f}", help="Rata-rata selisih prediksi dengan nilai asli")
        col_metric3.metric("RMSE", f"{rmse:.3f}")
        
        st.divider()
        
        # 2. Visualisasi Actual vs Predicted
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.subheader("1. Prediksi vs Aktual")
            fig3, ax3 = plt.subplots(figsize=(6, 4))
            sns.scatterplot(x=y_test, y=y_pred, color='blue', alpha=0.5, ax=ax3)
            # Garis Perfect Prediction
            ax3.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            ax3.set_xlabel('Actual GPA')
            ax3.set_ylabel('Predicted GPA')
            ax3.set_title('Sebaran Hasil Prediksi')
            st.pyplot(fig3)
            st.caption("Titik yang berada di garis merah adalah prediksi yang sempurna.")

        # 3. Visualisasi Koefisien (Feature Importance) - INI PENTING
        with col_right:
            st.subheader("2. Faktor Pengaruh (Koefisien)")
            fig_coef, ax_coef = plt.subplots(figsize=(6, 4))
            
            # Warna bar: Hijau jika positif, Merah jika negatif
            colors = ['green' if x > 0 else 'red' for x in coef_df['Koefisien']]
            
            sns.barplot(data=coef_df, x='Koefisien', y='Fitur', palette=colors, ax=ax_coef)
            ax_coef.set_title('Dampak Aktivitas terhadap GPA')
            ax_coef.set_xlabel('Besar Pengaruh (Poin GPA)')
            st.pyplot(fig_coef)
            st.caption("Nilai positif (Hijau) menaikkan GPA, nilai negatif (Merah) menurunkan GPA.")

        st.divider()
        st.write("**Insight:**")
        st.markdown("""
        * **Jam Belajar** memiliki dampak positif terbesar.
        * **Aktivitas Fisik & Tidur Berlebih** (dalam dataset ini) berkorelasi negatif, mengindikasikan *trade-off* waktu belajar.
        """)

    with tab2:
        st.header("Simulasi Prediksi GPA")
        st.write("Masukkan profil kegiatan harian siswa:")
        
        # Input Form
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                study_hours = st.number_input("📚 Jam Belajar / Hari", 0.0, 24.0, 5.0, 0.5)
                extracurricular = st.number_input("⚽ Jam Ekstrakurikuler / Hari", 0.0, 24.0, 1.0, 0.5)
                sleep_hours = st.number_input("💤 Jam Tidur / Hari", 0.0, 24.0, 7.0, 0.5)
                
            with col2:
                social_hours = st.number_input("🗣️ Jam Bersosialisasi / Hari", 0.0, 24.0, 2.0, 0.5)
                physical_hours = st.number_input("🏃 Jam Olahraga / Hari", 0.0, 24.0, 1.0, 0.5)
                stress_input = st.selectbox("🧠 Tingkat Stres", ["Low", "Moderate", "High"])
            
            submitted = st.form_submit_button("Hitung Prediksi GPA", type="primary")

        if submitted:
            # Mapping input stress
            stress_map = {"Low": 0, "Moderate": 1, "High": 2}
            stress_encoded = stress_map[stress_input]
            
            # Membuat dataframe input (Pastikan urutan kolom SAMA dengan saat training)
            input_data = pd.DataFrame([[
                study_hours, extracurricular, sleep_hours, social_hours, physical_hours, stress_encoded
            ]], columns=['Study_Hours_Per_Day', 'Extracurricular_Hours_Per_Day', 
                         'Sleep_Hours_Per_Day', 'Social_Hours_Per_Day', 
                         'Physical_Activity_Hours_Per_Day', 'Stress_Level_Encoded'])
            
            # Prediksi
            prediction = model.predict(input_data)[0]
            
            # Batasi hasil agar logis (0.0 - 4.0)
            prediction = np.clip(prediction, 0.0, 4.0)
            
            st.markdown("---")
            
            # Logika Tampilan Hasil
            col_res1, col_res2 = st.columns([1, 2])
            
            with col_res1:
                if prediction >= 3.5:
                    st.success(f"### GPA: {prediction:.2f}")
                    st.markdown("**Predikat: Cumlaude** 🌟")
                elif prediction >= 3.0:
                    st.info(f"### GPA: {prediction:.2f}")
                    st.markdown("**Predikat: Sangat Baik** 👍")
                else:
                    st.warning(f"### GPA: {prediction:.2f}")
                    st.markdown("**Predikat: Perlu Peningkatan** ⚠️")
            
            with col_res2:
                st.write("**Saran berdasarkan model:**")
                # Logika saran dinamis sederhana
                if study_hours < 4.0:
                    st.write("- 📚 Coba tingkatkan jam belajar minimal menjadi 5-6 jam sehari.")
                if sleep_hours > 9.0:
                    st.write("- 💤 Jam tidur Anda cukup tinggi, pastikan tidak mengganggu waktu produktif.")
                if social_hours > 3.0:
                    st.write("- 🗣️ Pertimbangkan untuk sedikit mengurangi waktu bersosialisasi agar lebih fokus.")
                if prediction >= 3.5:
                    st.write("- ✨ Pertahankan pola hidup ini, hasilnya sudah sangat optimal!")

# ==========================================
# PAGE 3: CREDIT
# ==========================================
elif page == "Credit":
    st.header("👥 Anggota Kelompok")
    
    st.markdown("""
    **Teknik Informatika Bilingual 2023** 
    
    **Mata Kuliah:** Pembelajaran Mesin 251P1  
    **Dosen Pengampu:** Assoc. Prof. Dian Palupi Rini, M.Kom., Ph.D.
    """)
    
    st.divider()
    
    col_members = st.columns(1)
    
    with col_members[0]:
        st.markdown("""
        1. **Nabilah Shamid** (09021382328147)
        2. **Indrina Nur Chairunnisya** (09021382328157)
        3. **Azka Hukma Tsabita** (09021382328159)
        4. **Shalaisya Fattiha Ramadhani** (09021382328161)
        5. **Afny Chiara Wildani Nst** (09021382328167)
        """)
        
    st.info("Terima kasih telah mencoba aplikasi prediksi kami! 🚀")
