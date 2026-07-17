import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

# ================= KONFIGURASI HALAMAN =================
st.set_page_config(page_title="Decision Support System HR Analytics", layout="wide", initial_sidebar_state="expanded")

# Kustomisasi CSS untuk merapikan margin dan metrik (Meniru Looker)
st.markdown("""
    <style>
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    div[data-testid="stMetricValue"] { font-size: 28px; }
    </style>
""", unsafe_allow_html=True)

# ================= FUNGSI MEMUAT & PROSES DATA (CACHE) =================
@st.cache_data
def load_and_process_data():
    df = pd.read_csv("HR_Analytics.csv")
    
    # --- 1. PROSES KLASIFIKASI (Random Forest Classifier) ---
    le_attr = LabelEncoder()
    le_over = LabelEncoder()
    df['Attrition_Num'] = le_attr.fit_transform(df['Attrition']) # Yes=1, No=0
    df['OverTime_Num'] = le_over.fit_transform(df['OverTime'])
    
    features_clf = ['OverTime_Num', 'MonthlyIncome', 'JobSatisfaction', 'WorkLifeBalance']
    clf = RandomForestClassifier(random_state=42, max_depth=5)
    clf.fit(df[features_clf], df['Attrition_Num'])
    # Menghasilkan persentase risiko
    df['Risk_Score_Percentage'] = clf.predict_proba(df[features_clf])[:, 1] * 100 
    
    # --- 2. PROSES REGRESI (Random Forest Regressor) ---
    features_reg = ['JobLevel', 'TotalWorkingYears', 'PerformanceRating', 'YearsAtCompany']
    reg = RandomForestRegressor(random_state=42, max_depth=5)
    reg.fit(df[features_reg], df['MonthlyIncome'])
    df['Expected_MonthlyIncome'] = reg.predict(df[features_reg])
    df['Selisih_Gaji'] = df['MonthlyIncome'] - df['Expected_MonthlyIncome']
    
    # --- 3. PROSES CLUSTERING (K-Means) ---
    features_clu = ['MonthlyIncome', 'PercentSalaryHike', 'PerformanceRating', 'TotalWorkingYears']
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['Cluster_ID'] = kmeans.fit_predict(df[features_clu])
    
    # Mapping nama cluster sesuai desain gambar
    def map_segment(cid):
        if cid == 0: return "Standar (Gaji & Performa Standar)"
        elif cid == 1: return "Veteran (Gaji Tinggi, Performa Stabil)"
        else: return "Bintang (Performa Maksimal)"
    df['Value_Segment'] = df['Cluster_ID'].apply(map_segment)
    
    return df

df = load_and_process_data()

# ================= SIDEBAR NAVIGASI =================
st.sidebar.title("Sistem Pendukung Keputusan HR Analytics")
menu = st.sidebar.radio(
    "",
    ("👥 Dashboard Prediksi Risiko Resign", 
     "💰 Dashboard Regresi: Analisis Gaji", 
     "📈 Dashboard Clustering: Kinerja", 
     "📊 Dashboard Analisis SDM (EDA)")
)

# Palet warna seragam meniru Looker (Biru/Teal)
color_palette = ['#008fd5', '#00c6d9', '#004a7c', '#4cb5f5']

# ================= HALAMAN 1: KLASIFIKASI =================
if menu == "👥 Dashboard Prediksi Risiko Resign":
    st.markdown("<h3 style='background-color:#1E56A0; color:white; padding:10px; border-radius:5px;'>Klasifikasi Prediksi Karyawan Resign</h3>", unsafe_allow_html=True)
    
    # Info Panel
    st.info("**Algoritma:** Random Forest Classifier | **Fitur:** Attrition, OverTime, MonthlyIncome, JobSatisfaction, WorkLifeBalance. | **Fungsi:** Membantu HR melakukan intervensi dini sebelum karyawan berpotensi keluar.")
    
    # Filter
    st.markdown("##### Filter Pencarian Data")
    col_f1, col_f2, col_f3 = st.columns(3)
    dept_filter = col_f1.selectbox("Department", ["Semua"] + list(df['Department'].unique()))
    role_filter = col_f2.selectbox("JobRole", ["Semua"] + list(df['JobRole'].unique()))
    over_filter = col_f3.selectbox("OverTime", ["Semua"] + list(df['OverTime'].unique()))
    
    # Apply Filter
    df_filtered = df.copy()
    if dept_filter != "Semua": df_filtered = df_filtered[df_filtered['Department'] == dept_filter]
    if role_filter != "Semua": df_filtered = df_filtered[df_filtered['JobRole'] == role_filter]
    if over_filter != "Semua": df_filtered = df_filtered[df_filtered['OverTime'] == over_filter]
    
    # Scorecards
    c1, c2, c3 = st.columns(3)
    karyawan_aktif = len(df_filtered[df_filtered['Attrition'] == 'No'])
    karyawan_keluar = len(df_filtered[df_filtered['Attrition'] == 'Yes'])
    avg_risk = df_filtered['Risk_Score_Percentage'].mean()
    
    c1.metric("Total Karyawan Aktif", f"{karyawan_aktif:,}".replace(',', '.'))
    c2.metric("Rata-rata Risiko Resign (%)", f"{avg_risk:.2f}%".replace('.', ','))
    c3.metric("Total Karyawan Keluar (Masa Lalu)", f"{karyawan_keluar:,}".replace(',', '.'))
    
    # Tabel dengan Data Bar (Meniru Looker)
    st.markdown("##### Daftar Karyawan dengan Risiko Resign Tertinggi")
    top_risk = df_filtered[['EmpID', 'Department', 'JobRole', 'Risk_Score_Percentage', 'MonthlyIncome', 'OverTime']].sort_values(by='Risk_Score_Percentage', ascending=False).head(100)
    
    st.dataframe(
        top_risk,
        column_config={
            "Risk_Score_Percentage": st.column_config.ProgressColumn("Risk_Score (%)", format="%.2f", min_value=0, max_value=100),
            "MonthlyIncome": st.column_config.NumberColumn("MonthlyIncome", format="%d")
        },
        use_container_width=True, hide_index=True
    )
    
    # Grafik Bawah
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        fig1 = px.histogram(df_filtered, x='OverTime', color='Attrition', barmode='group',
                            title="Tingkat Risiko Resign: Karyawan Lembur vs Tidak",
                            color_discrete_sequence=['#008fd5', '#00c6d9'], text_auto=True)
        st.plotly_chart(fig1, use_container_width=True)
    with col_g2:
        fig2 = px.scatter(df_filtered, x='MonthlyIncome', y='Risk_Score_Percentage', hover_data=['EmpID'],
                          title="Hubungan Antara Besaran Gaji dan Risiko Resign",
                          color_discrete_sequence=['#00c6d9'])
        st.plotly_chart(fig2, use_container_width=True)

# ================= HALAMAN 2: REGRESI =================
elif menu == "💰 Dashboard Regresi: Analisis Gaji":
    st.markdown("<h3 style='background-color:#1E56A0; color:white; padding:10px; border-radius:5px;'>Estimasi Standar Gaji (Model Regresi)</h3>", unsafe_allow_html=True)
    
    st.info("**Algoritma:** Random Forest Regressor | **Fitur Utama:** JobLevel, TotalWorkingYears, PerformanceRating, YearsAtCompany. | **Fungsi:** Menyusun anggaran gaji yang objektif dan menemukan kesenjangan (gap) kompensasi.")
    
    # Filters
    col_f1, col_f2, col_f3 = st.columns(3)
    dept_filter = col_f1.selectbox("Department", ["Semua"] + list(df['Department'].unique()))
    role_filter = col_f2.selectbox("JobRole", ["Semua"] + list(df['JobRole'].unique()))
    lvl_filter = col_f3.selectbox("JobLevel", ["Semua"] + list(df['JobLevel'].sort_values().unique()))
    
    df_filtered = df.copy()
    if dept_filter != "Semua": df_filtered = df_filtered[df_filtered['Department'] == dept_filter]
    if role_filter != "Semua": df_filtered = df_filtered[df_filtered['JobRole'] == role_filter]
    if lvl_filter != "Semua": df_filtered = df_filtered[df_filtered['JobLevel'] == lvl_filter]
    
    # Scorecards
    c1, c2, c3 = st.columns(3)
    c1.metric("Rata-rata Gaji Saat Ini", f"Rp {df_filtered['MonthlyIncome'].mean():,.2f}")
    c2.metric("Rata-rata Gaji Ideal (Sistem)", f"Rp {df_filtered['Expected_MonthlyIncome'].mean():,.2f}")
    c3.metric("Total Karyawan Diperiksa", f"{len(df_filtered):,}".replace(',', '.'))
    
    # Tabel
    st.markdown("##### Daftar Rincian dan Selisih Gaji Karyawan")
    tabel_regresi = df_filtered[['EmpID', 'Department', 'JobRole', 'MonthlyIncome', 'Expected_MonthlyIncome', 'Selisih_Gaji']]
    st.dataframe(
        tabel_regresi,
        column_config={
            "MonthlyIncome": st.column_config.NumberColumn("MonthlyIncome", format="%d"),
            "Expected_MonthlyIncome": st.column_config.NumberColumn("Expected_MonthlyIncome", format="%.2f"),
            "Selisih_Gaji": st.column_config.NumberColumn("Selisih Gaji", format="%.2f")
        },
        use_container_width=True, hide_index=True
    )
    
    # Grafik Bawah
    fig = px.scatter(df_filtered, x='Expected_MonthlyIncome', y='MonthlyIncome', hover_data=['EmpID'],
                     title="Grafik Sebaran: Gaji Saat Ini vs Gaji Ideal",
                     color_discrete_sequence=['#00c6d9'])
    st.plotly_chart(fig, use_container_width=True)

# ================= HALAMAN 3: CLUSTERING =================
elif menu == "📈 Dashboard Clustering: Kinerja":
    st.markdown("<h3 style='background-color:#1E56A0; color:white; padding:10px; border-radius:5px;'>Dashboard Evaluasi Gaji & Kinerja Karyawan</h3>", unsafe_allow_html=True)
    
    st.info("**Algoritma:** K-Means Clustering | **Variabel:** Gaji Bulanan, Persentase Kenaikan Gaji, Kinerja, Pengalaman Kerja. | **Fungsi:** Melakukan audit efisiensi anggaran untuk memastikan apresiasi yang sepadan.")
    
    # Filters
    col_f1, col_f2, col_f3 = st.columns(3)
    dept_filter = col_f1.selectbox("Department", ["Semua"] + list(df['Department'].unique()))
    role_filter = col_f2.selectbox("JobRole", ["Semua"] + list(df['JobRole'].unique()))
    seg_filter = col_f3.selectbox("Value_Segment", ["Semua"] + list(df['Value_Segment'].unique()))
    
    df_filtered = df.copy()
    if dept_filter != "Semua": df_filtered = df_filtered[df_filtered['Department'] == dept_filter]
    if role_filter != "Semua": df_filtered = df_filtered[df_filtered['JobRole'] == role_filter]
    if seg_filter != "Semua": df_filtered = df_filtered[df_filtered['Value_Segment'] == seg_filter]
    
    # Scorecards
    c1, c2, c3 = st.columns(3)
    c1.metric("Rata-rata Kinerja (Skala 1-4)", f"{df_filtered['PerformanceRating'].mean():.2f}".replace('.', ','))
    c2.metric("Rata-rata Kenaikan Gaji (%)", f"{df_filtered['PercentSalaryHike'].mean():.2f}%".replace('.', ','))
    c3.metric("Total Pengeluaran Gaji Bulan Ini", f"Rp {df_filtered['MonthlyIncome'].sum():,.2f}")
    
    # Grafik Atas
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        pie_data = df_filtered['Value_Segment'].value_counts().reset_index()
        pie_data.columns = ['Segment', 'Count']
        fig1 = px.pie(pie_data, names='Segment', values='Count', title='Jumlah Karyawan di Setiap Kelompok',
                      color_discrete_sequence=['#00c6d9', '#008fd5', '#004a7c'])
        fig1.update_traces(hole=0.4)
        st.plotly_chart(fig1, use_container_width=True)
    with col_g2:
        fig2 = px.scatter(df_filtered, x='MonthlyIncome', y='PercentSalaryHike', color='Value_Segment',
                          title="Grafik Perbandingan: Gaji Saat Ini vs Persentase Kenaikan",
                          color_discrete_sequence=['#004a7c', '#00c6d9', '#008fd5'])
        st.plotly_chart(fig2, use_container_width=True)
        
    # Tabel
    st.markdown("##### Daftar Rincian Kinerja dan Gaji Karyawan")
    tabel_cluster = df_filtered[['EmpID', 'Department', 'JobRole', 'Value_Segment', 'MonthlyIncome', 'PercentSalaryHike', 'PerformanceRating']]
    st.dataframe(
        tabel_cluster,
        column_config={
            "PercentSalaryHike": st.column_config.ProgressColumn("PercentSalaryHike", format="%d", min_value=0, max_value=30),
            "PerformanceRating": st.column_config.ProgressColumn("PerformanceRating", format="%d", min_value=1, max_value=4)
        },
        use_container_width=True, hide_index=True
    )

# ================= HALAMAN 4: ANALISIS SDM (EDA) =================
elif menu == "📊 Dashboard Analisis SDM (EDA)":
    st.markdown("<h3 style='background-color:#1E56A0; color:white; padding:10px; border-radius:5px;'>Dashboard Analisis SDM (HR Analytics)</h3>", unsafe_allow_html=True)
    
    # Top Scorecards
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Total Karyawan", f"{len(df):,}".replace(',', '.'))
    m2.metric("Total Attrition", f"{(len(df[df['Attrition']=='Yes'])/len(df))*100:.2f}%".replace('.', ','))
    m3.metric("AVG Pendapatan", f"{df['MonthlyIncome'].mean():,.2f}")
    m4.metric("AVG Usia", f"{int(df['Age'].mean())}")
    m5.metric("AVG Masa Kerja", f"{df['YearsAtCompany'].mean():.2f}".replace('.', ','))
    m6.metric("AVG Kepuasan Kerja", f"{df['JobSatisfaction'].mean():.2f}".replace('.', ','))
    
    st.write("---")
    
    # Baris 1: Demografi
    r1c1, r1c2, r1c3, r1c4 = st.columns(4)
    with r1c1:
        dept_counts = df['Department'].value_counts().reset_index()
        fig_dept = px.bar(dept_counts, x='count', y='Department', orientation='h', title='Karyawan Berdasar Departemen', color_discrete_sequence=['#00c6d9'], text_auto=True)
        st.plotly_chart(fig_dept, use_container_width=True)
    with r1c2:
        role_counts = df['JobRole'].value_counts().reset_index()
        fig_role = px.bar(role_counts, x='JobRole', y='count', title='Karyawan Berdasar Jabatan', color_discrete_sequence=['#00c6d9'], text_auto=True)
        st.plotly_chart(fig_role, use_container_width=True)
    with r1c3:
        fig_gen = px.pie(df, names='Gender', title='Jenis Kelamin', hole=0.5, color_discrete_sequence=['#008fd5', '#00c6d9'])
        st.plotly_chart(fig_gen, use_container_width=True)
    with r1c4:
        fig_edu = px.pie(df, names='EducationField', title='Bidang Pendidikan', hole=0.5, color_discrete_sequence=px.colors.sequential.Blues_r)
        st.plotly_chart(fig_edu, use_container_width=True)

    # Baris 2 & 3: Attrition & Kompensasi
    r2c1, r2c2 = st.columns(2)
    with r2c1:
        fig_age = px.histogram(df, x='AgeGroup', color='Attrition', barmode='group', title='Attrition Berdasar Kelompok Usia', color_discrete_sequence=['#00c6d9', '#008fd5'], text_auto=True)
        st.plotly_chart(fig_age, use_container_width=True)
        
        fig_ot = px.histogram(df, x='OverTime', color='Attrition', barmode='group', title='Attrition vs OverTime', color_discrete_sequence=['#00c6d9', '#008fd5'], orientation='h', text_auto=True)
        st.plotly_chart(fig_ot, use_container_width=True)
        
    with r2c2:
        fig_att_dept = px.histogram(df, y='Department', color='Attrition', barmode='group', title='Tingkat Attrition Berdasarkan Departemen', color_discrete_sequence=['#00c6d9', '#008fd5'], orientation='h', text_auto=True)
        st.plotly_chart(fig_att_dept, use_container_width=True)
        
        avg_inc_lvl = df.groupby('JobLevel')['MonthlyIncome'].mean().reset_index()
        fig_inc_lvl = px.bar(avg_inc_lvl, y='JobLevel', x='MonthlyIncome', orientation='h', title='Rata-rata Pendapatan Bulanan Berdasarkan Level', color_discrete_sequence=['#00c6d9'], text_auto=True)
        st.plotly_chart(fig_inc_lvl, use_container_width=True)