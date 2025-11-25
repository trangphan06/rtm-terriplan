import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from k_means_constrained import KMeansConstrained
import numpy as np
import io
from itertools import combinations
import folium
from streamlit_folium import st_folium
import math

# --- KHỞI TẠO STATE ---
if 'sidebar_state' not in st.session_state: st.session_state.sidebar_state = 'expanded'
if 'df' not in st.session_state: st.session_state.df = None
if 'df_edited' not in st.session_state: st.session_state.df_edited = None
if 'df_map' not in st.session_state: st.session_state.df_map = None
if 'original_report_df' not in st.session_state: st.session_state.original_report_df = None
if 'col_mapping' not in st.session_state: st.session_state.col_mapping = {}
if 'mapping_confirmed' not in st.session_state: st.session_state.mapping_confirmed = False
if 'time_matrix' not in st.session_state: 
    st.session_state.time_matrix = {
        'MT': 19.5, 'Cooler': 18.0, 'Gold': 9.0, 'Silver': 7.8, 
        'Bronze': 6.8, 'default': 10.0
    }

# State riêng cho từng Version
# V1 State
if 'v1_df_edited' not in st.session_state: st.session_state.v1_df_edited = None
if 'v1_df_map' not in st.session_state: st.session_state.v1_df_map = None
if 'v1_report' not in st.session_state: st.session_state.v1_report = None
if 'v1_df_original' not in st.session_state: st.session_state.v1_df_original = None
if 'v1_map_snapshot' not in st.session_state: st.session_state.v1_map_snapshot = None # [NEW] Cache Map Gốc

# V2 State
if 'v2_df_edited' not in st.session_state: st.session_state.v2_df_edited = None
if 'v2_df_map' not in st.session_state: st.session_state.v2_df_map = None
if 'v2_report' not in st.session_state: st.session_state.v2_report = None
if 'v2_df_original' not in st.session_state: st.session_state.v2_df_original = None
if 'v2_map_snapshot' not in st.session_state: st.session_state.v2_map_snapshot = None # [NEW] Cache Map Gốc

# State quản lý hiển thị
if 'map_version' not in st.session_state: st.session_state.map_version = 0
if 'col_widths' not in st.session_state: st.session_state.col_widths = {}

# --- CẤU HÌNH TRANG & CSS ---
st.set_page_config(
    layout="wide",
    page_title="Territory Planner Pro",
    initial_sidebar_state=st.session_state.sidebar_state 
)

st.markdown("""
    <style>
        .block-container { padding-top: 1rem; padding-bottom: 1rem; padding-left: 1rem; padding-right: 1rem; }
        iframe { width: 100% !important; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 1. CÁC HÀM HỖ TRỢ
# ==========================================

@st.cache_data
def load_excel_file(file):
    return pd.read_excel(file, dtype=str)

@st.cache_data
def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(float, [lat1, lon1, lat2, lon2])
    R = 6371
    dLat = np.radians(lat2 - lat1)
    dLon = np.radians(lon2 - lon1)
    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)
    a = np.sin(dLat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dLon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

# --- LOGIC VERSION 1 ---
def run_territory_planning_v1(df, lat_col, lon_col, n_clusters, min_size, max_size, n_init=50):
    df_run = df.copy()
    df_run[lat_col] = pd.to_numeric(df_run[lat_col], errors='coerce')
    df_run[lon_col] = pd.to_numeric(df_run[lon_col], errors='coerce')
    df_run = df_run.dropna(subset=[lat_col, lon_col])
    
    coords = df_run[[lat_col, lon_col]]
    scaler = StandardScaler()
    coords_scaled = scaler.fit_transform(coords)
    
    if min_size * n_clusters > len(df): return None, "Lỗi: Số lượng tối thiểu quá lớn."
    if max_size * n_clusters < len(df): return None, "Lỗi: Số lượng tối đa quá nhỏ."

    progress_text = "Đang xử lý..."
    my_bar = st.progress(0, text=progress_text)
    best_clf = None
    best_inertia = float('inf')

    try:
        for i in range(n_init):
            clf = KMeansConstrained(
                n_clusters=n_clusters, size_min=min_size, size_max=max_size, 
                random_state=42 + i, n_init=1
            )
            clf.fit(coords_scaled)
            if clf.inertia_ < best_inertia:
                best_inertia = clf.inertia_
                best_clf = clf
            percent = int((i + 1) / n_init * 100)
            my_bar.progress((i + 1) / n_init, text=f"Đang xử lý... {percent}%")
            
        my_bar.empty()
        df_run['territory_id'] = best_clf.labels_ + 1
        
        stats = df_run['territory_id'].value_counts().sort_index().reset_index()
        stats.columns = ['Tuyến (RouteID)', 'Số lượng KH']
        
        return df_run, stats
    except Exception as e:
        my_bar.empty()
        return None, str(e)

# --- LOGIC VERSION 2 ---
def run_territory_planning_v2(df, lat_col, lon_col, freq_col, type_col, time_matrix, n_clusters, min_hours, max_hours):
    df_run = df.copy()
    df_run[lat_col] = pd.to_numeric(df_run[lat_col], errors='coerce')
    df_run[lon_col] = pd.to_numeric(df_run[lon_col], errors='coerce')
    df_run = df_run.dropna(subset=[lat_col, lon_col])

    def calc_load(row):
        try: freq = float(row[freq_col])
        except: freq = 1.0
        c_type = str(row[type_col]).strip()
        time_val = time_matrix.get(c_type, time_matrix.get('default', 10.0))
        return freq * time_val

    df_run['workload_min'] = df_run.apply(calc_load, axis=1)
    total_minutes = df_run['workload_min'].sum()
    
    TARGET_POINTS = 50000 
    raw_quantum = total_minutes / TARGET_POINTS
    QUANTUM = max(1, math.ceil(raw_quantum)) 
    
    df_run['weight_points'] = np.ceil(df_run['workload_min'] / QUANTUM).astype(int)
    
    df_exploded = df_run.loc[df_run.index.repeat(df_run['weight_points'])].copy()
    df_exploded['original_index'] = df_exploded.index
    df_exploded = df_exploded.reset_index(drop=True)
    
    size_min = int((min_hours * 60) / QUANTUM)
    size_max = int((max_hours * 60) / QUANTUM)
    
    scaler = StandardScaler()
    coords = df_exploded[[lat_col, lon_col]]
    coords_scaled = scaler.fit_transform(coords)
    
    n_init = 5
    progress_text = f"Đang xử lý..."
    my_bar = st.progress(0, text=progress_text)
    
    best_clf = None
    best_inertia = float('inf')

    try:
        for i in range(n_init):
            clf = KMeansConstrained(
                n_clusters=n_clusters, size_min=size_min, size_max=size_max,
                random_state=42 + i, n_init=1
            )
            clf.fit(coords_scaled)
            if clf.inertia_ < best_inertia:
                best_inertia = clf.inertia_
                best_clf = clf
            percent = int((i + 1) / n_init * 100)
            my_bar.progress((i + 1) / n_init, text=f"Đang xử lý...{percent}%")
        
        my_bar.empty()
        
        df_exploded['territory_id'] = best_clf.labels_ + 1
        final_labels = df_exploded.groupby('original_index')['territory_id'].agg(lambda x: x.mode()[0])
        df_run['territory_id'] = final_labels
        
        stats = df_run.groupby('territory_id').agg(
            count_kh=('territory_id', 'count'),
            sum_min=('workload_min', 'sum')
        ).reset_index()
        stats.columns = ['Tuyến (RouteID)', 'Số lượng KH', 'Workload_Total_Min']
        stats['Workload_Day'] = (stats['Workload_Total_Min'] / 60 / 22).round(2)
        
        return df_run, stats
        
    except Exception as e:
        my_bar.empty()
        return None, str(e)

# [OPTIMIZE]: Không dùng Cache cho hàm này để dễ dàng quản lý Object trong Session State
def generate_folium_map(_df, _mapping, _time_matrix, mode="Chế độ 1"):
    if _df.empty: return None, None
    
    df_plot = _df.copy()
    lat_col = _mapping['lat']
    lon_col = _mapping['lon']
    df_plot[lat_col] = pd.to_numeric(df_plot[lat_col], errors='coerce')
    df_plot[lon_col] = pd.to_numeric(df_plot[lon_col], errors='coerce')
    df_plot = df_plot.dropna(subset=[lat_col, lon_col])
    
    if df_plot.empty: return None, None

    map_center = [df_plot[lat_col].mean(), df_plot[lon_col].mean()]
    m = folium.Map(location=map_center, zoom_start=11)
    
    col_code = _mapping['customer_code']
    col_name = _mapping.get('customer_name')
    col_addr = _mapping.get('address')
    col_vol = _mapping.get('vol_ec')
    col_freq = _mapping.get('freq')
    col_type = _mapping.get('type')
    
    # Tạo màu
    colors = ["#FF0000", "#0000FF", "#00FF00", "#FFFF00", "#FF00FF", "#00FFFF", "#800000", "#008000", "#000080", "#FFA500"]
    unique_ids = sorted(df_plot['territory_id'].unique())
    color_map = {int(id): colors[(int(id) - 1) % len(colors)] for id in unique_ids}
    
    for _, row in df_plot.iterrows():
        tooltip_parts = [f"<b>KH: {row[col_code]}</b>", f"Tuyến: {row['territory_id']}"]
        
        if col_name and pd.notna(row.get(col_name)): tooltip_parts.append(f"Tên: {row[col_name]}")
        if col_addr and pd.notna(row.get(col_addr)): tooltip_parts.append(f"Đ/c: {row[col_addr]}")
        if col_vol and pd.notna(row.get(col_vol)): tooltip_parts.append(f"Vol: {row[col_vol]}")
        
        if mode == "Chế độ 2":
            if col_freq and pd.notna(row.get(col_freq)): tooltip_parts.append(f"Tần suất: {row[col_freq]}")
            if col_type and pd.notna(row.get(col_type)):
                seg_val = str(row[col_type]).strip()
                tooltip_parts.append(f"Phân loại: {seg_val}")
                time_val = _time_matrix.get(seg_val, _time_matrix.get('default', 10.0))
                tooltip_parts.append(f"Thời gian: {time_val}p")

        tooltip_txt = "<br>".join(tooltip_parts)
        
        c = color_map.get(int(row['territory_id']), 'gray')

        folium.CircleMarker(
            location=[row[lat_col], row[lon_col]],
            radius=4,
            color=c, fill=True, fill_color=c, fill_opacity=0.7,
            tooltip=tooltip_txt
        ).add_to(m)
    return m, map_center

# ==========================================
# 2. GIAO DIỆN (UI)
# ==========================================

st.title("Công cụ Phân chia Địa bàn (Territory Plan)")

current_mode = "Chế độ 1"
current_df_edited = None

with st.sidebar:
    st.header("1. Chế độ Phân tuyến")
    plan_mode = st.radio(
        "Chọn chế độ:",
        ["Chế độ 1: Cân bằng Số lượng KH", "Chế độ 2: Cân bằng Số giờ viếng thăm"],
        index=0
    )
    mode_key = "Chế độ 1" if "Chế độ 1" in plan_mode else "Chế độ 2"
    current_mode = mode_key

    if mode_key == "Chế độ 1":
        current_df_edited = st.session_state.v1_df_edited
    else:
        current_df_edited = st.session_state.v2_df_edited
    
    if st.session_state.get('last_mode') != mode_key:
        st.session_state.last_mode = mode_key
        st.rerun()

    st.divider()
    st.header("2. Tải dữ liệu")

    @st.cache_data
    def create_template_v1():
        df = pd.DataFrame({
            "Customer Code (*)": ["KH01", "KH02"], "Latitude (*)": [10.7, 10.8], "Longitude (*)": [106.6, 106.7],
            "Customer Name": ["A", "B"], "Address": ["HCM", "HCM"], "VolEC": [100, 200],
            "Frequency": [4, 8], "Segment": ["MT", "Cooler"]
        })
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer: df.to_excel(writer, index=False)
        return output.getvalue()

    @st.cache_data
    def create_template_v2():
        df = pd.DataFrame({
            "Customer Code (*)": ["KH01", "KH02"], "Latitude (*)": [10.7, 10.8], "Longitude (*)": [106.6, 106.7],
            "Customer Name": ["A", "B"], "Address": ["HCM", "HCM"], "VolEC": [100, 200],
            "Frequency (*)": [4, 8], "Segment (*)": ["MT", "Cooler"]
        })
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer: df.to_excel(writer, index=False)
        return output.getvalue()

    template_data = create_template_v1() if mode_key == "Chế độ 1" else create_template_v2()
    st.download_button(f"Tải template mẫu ({mode_key})", template_data, f"Template_{mode_key}.xlsx", use_container_width=True)

    uploaded_file = st.file_uploader("Upload", type=['xlsx', 'xls'], label_visibility="collapsed")

    if uploaded_file:
        try:
            df = load_excel_file(uploaded_file)
            st.session_state.df = df
            all_cols = df.columns.tolist()
            options_cols = ["[Bỏ qua]"] + all_cols

            st.subheader("3. Chọn cột dữ liệu")

            with st.form("column_mapping_form"):
                c1, c2 = st.columns(2)
                def get_idx(name, lst, is_opt=False):
                    target = options_cols if is_opt else lst
                    idx = 0
                    if name in target: idx = target.index(name)
                    return idx

                with c1:
                    cc_col = st.selectbox("Code KH (*)", all_cols, index=get_idx("Customer Code", all_cols))
                    lat_col = st.selectbox("Lat (Vĩ độ) (*)", all_cols, index=get_idx("Latitude", all_cols))
                    addr_col = st.selectbox("Địa chỉ", options_cols, index=get_idx("Address", options_cols, True))

                with c2:
                    lon_col = st.selectbox("Long (Kinh độ) (*)", all_cols, index=get_idx("Longitude", all_cols))
                    name_col = st.selectbox("Tên KH", options_cols, index=get_idx("Customer Name", options_cols, True))
                    vol_col = st.selectbox("VolEC", options_cols, index=get_idx("VolEC", options_cols, True))

                c3, c4 = st.columns(2)
                with c3:
                    if mode_key == "Chế độ 1":
                        freq_col = st.selectbox("Tần suất", options_cols, index=get_idx("Frequency", options_cols, True))
                    else:
                        freq_col = st.selectbox("Tần suất (*)", all_cols, index=get_idx("Frequency", all_cols))

                with c4:
                    if mode_key == "Chế độ 1":
                        type_col = st.selectbox("Phân loại Segment", options_cols, index=get_idx("Segment", options_cols, True))
                    else:
                        type_col = st.selectbox("Phân loại Segment (*)", all_cols, index=get_idx("Segment", all_cols))
                
                submitted = st.form_submit_button("Xác nhận")

            if submitted:
                final_freq = None if (mode_key == "Chế độ 1" and freq_col == "[Bỏ qua]") else freq_col
                final_type = None if (mode_key == "Chế độ 1" and type_col == "[Bỏ qua]") else type_col

                mapping_local = {
                    "customer_code": cc_col, "lat": lat_col, "lon": lon_col,
                    "customer_name": None if name_col=="[Bỏ qua]" else name_col,
                    "address": None if addr_col=="[Bỏ qua]" else addr_col,
                    "vol_ec": None if vol_col=="[Bỏ qua]" else vol_col,
                    "freq": final_freq, "type": final_type
                }
                st.session_state.col_mapping = mapping_local
                st.session_state.mapping_confirmed = True
                
                if st.session_state.df is not None:
                    st.session_state.df[lat_col] = pd.to_numeric(st.session_state.df[lat_col], errors='coerce')
                    st.session_state.df[lon_col] = pd.to_numeric(st.session_state.df[lon_col], errors='coerce')
                    st.session_state.df = st.session_state.df.dropna(subset=[lat_col, lon_col])
                    st.success(f"Đã tải {len(st.session_state.df)} dòng. Sẵn sàng phân tuyến!")
                st.rerun()

            if mode_key == "Chế độ 2":
                with st.expander("Cài đặt Thời gian viếng thăm (Phút)", expanded=False):
                    current_time_data = pd.DataFrame(list(st.session_state.time_matrix.items()), columns=['Type', 'Min'])
                    edited_time = st.data_editor(current_time_data, num_rows="dynamic", hide_index=True)
                    st.session_state.time_matrix = dict(zip(edited_time['Type'], edited_time['Min']))

        except Exception as e:
            st.error(f"Lỗi file: {e}")

    # --- 4. THAM SỐ CHẠY ---
    if st.session_state.get('mapping_confirmed') and st.session_state.df is not None:
        st.divider()
        st.header("4. Điều kiện Phân tuyến")

        n_routes = st.number_input("Số tuyến (Routes)", 1, 100, 9)
        current_mapping = st.session_state.col_mapping

        if mode_key == "Chế độ 1":
            avg_qty = len(st.session_state.df) // n_routes
            st.caption(f"Trung bình: ~{avg_qty} KH/tuyến")
            sug_min_v1 = int(avg_qty*0.9)
            sug_max_v1 = int(avg_qty*1.1)
            min_v = st.number_input("Số KH tối thiếu", 0, value=sug_min_v1)
            max_v = st.number_input("Số KH tối đa", 0, value=sug_max_v1)
        else:
            temp_df = st.session_state.df.copy()
            time_matrix = st.session_state.time_matrix
            def quick_load(r):
                try: f = float(r[current_mapping['freq']])
                except: f = 1
                t = time_matrix.get(str(r[current_mapping['type']]).strip(), 10)
                return f * t
            try:
                total_min = temp_df.apply(quick_load, axis=1).sum()
                avg_hours = (total_min / 60) / n_routes
                st.caption(f"Tổng thời gian: {total_min/60:.1f}h. Trung bình: ~{avg_hours:.1f}h/tuyến")
                sug_min_h = round(avg_hours*0.9, 1)
                sug_max_h = round(avg_hours*1.1, 1)
                min_h = st.number_input("Số giờ viếng thăm tối thiểu", 0.0, value=sug_min_h, step=0.5)
                max_h = st.number_input("Số giờ viếng thăm tối đa", 0.0, value=sug_max_h, step=0.5)
            except Exception as e:
                st.error(f"Lỗi tính toán: {e}")
                min_h, max_h = 0.0, 0.0

        if st.button("🚀 Bắt đầu phân tuyến", type="primary", use_container_width=True):
            df_input = st.session_state.df
            res_df, res_stats = None, None
            err = None
            final_mapping = st.session_state.col_mapping

            if mode_key == "Chế độ 1":
                res_df, res_stats = run_territory_planning_v1(
                    df_input, final_mapping['lat'], final_mapping['lon'],
                    n_routes, min_v, max_v
                )
                if res_df is None: err = res_stats
            else:
                if not final_mapping.get('freq') or not final_mapping.get('type'):
                    err = "Lỗi: Chế độ 2 cần cột Tần suất & Phân loại."
                else:
                    res_df, res_stats = run_territory_planning_v2(
                        df_input, final_mapping['lat'], final_mapping['lon'],
                        final_mapping['freq'], final_mapping['type'], st.session_state.time_matrix,
                        n_routes, min_h, max_h
                    )
                    if res_df is None: err = res_stats

            if err:
                st.error(err)
            elif res_df is not None:
                # [OPTIMIZE]: Reset các state cache
                st.session_state.map_version = 0 
                st.session_state.col_widths = {}
                
                # Clear snapshot cũ để ép tạo mới
                st.session_state.v1_map_snapshot = None
                st.session_state.v2_map_snapshot = None
                
                if mode_key == "Chế độ 1":
                    st.session_state.v1_df_edited = res_df.copy()
                    st.session_state.v1_df_map = res_df.copy()
                    st.session_state.v1_report = res_stats.copy()
                    st.session_state.v1_df_original = res_df.copy()
                else:
                    st.session_state.v2_df_edited = res_df.copy()
                    st.session_state.v2_df_map = res_df.copy()
                    st.session_state.v2_report = res_stats.copy()
                    st.session_state.v2_df_original = res_df.copy()

                if st.session_state.sidebar_state == 'expanded':
                    st.session_state.sidebar_state = 'collapsed'
                    st.rerun()

# ==========================================
# 5. KHU VỰC HIỂN THỊ KẾT QUẢ (FINAL ADJUSTED FOR MODE 2)
# ==========================================

if current_df_edited is None:
    st.info(f"Vui lòng chạy phân tuyến cho {mode_key} để xem kết quả.")
else:
    # --- A. KHỞI TẠO ---
    mapping = st.session_state.col_mapping

    if mode_key == "Chế độ 1":
        key_original, key_saved = 'v1_df_original', 'v1_df_edited'
        key_snapshot = 'v1_map_snapshot'
    else:
        key_original, key_saved = 'v2_df_original', 'v2_df_edited'
        key_snapshot = 'v2_map_snapshot'

    if key_original not in st.session_state or st.session_state[key_original] is None:
        st.session_state[key_original] = st.session_state[key_saved].copy()
    
    df_saved = st.session_state[key_saved].copy()
    df_original = st.session_state[key_original].copy()

    # [OPTIMIZE]: TẠO SNAPSHOT BẢN ĐỒ GỐC (Chỉ chạy 1 lần duy nhất)
    if st.session_state.get(key_snapshot) is None:
        m_snapshot, _ = generate_folium_map(df_original, mapping, st.session_state.time_matrix, mode=mode_key)
        st.session_state[key_snapshot] = m_snapshot

    st.header(f"BẢN ĐỒ PHÂN TUYẾN ({mode_key})")

    # Scorecard
    if mode_key == "Chế độ 1":
        curr_stats = df_saved['territory_id'].value_counts().sort_index()
        display_data = [{'id': r, 'val': curr_stats.get(r, 0), 'count': curr_stats.get(r, 0)} for r in sorted(df_saved['territory_id'].unique())]
    else:
        grouped_v2 = df_saved.groupby('territory_id').agg(
            count=('territory_id', 'count'),
            workload_min=('workload_min', 'sum')
        ).sort_index()
        display_data = []
        for route_id in sorted(df_saved['territory_id'].unique()):
            if route_id in grouped_v2.index:
                row_data = grouped_v2.loc[route_id]
                display_data.append({'id': route_id, 'val': row_data['workload_min']/60/22, 'count': row_data['count']})
            else:
                display_data.append({'id': route_id, 'val': 0, 'count': 0})

    html_items = []
    for item in display_data:
        route_id = item['id']
        val = item['val']
        current_kh = item['count']
        orig_kh = df_original[df_original['territory_id'] == route_id].shape[0]
        arrow = " 🔼" if current_kh > orig_kh else (" 🔽" if current_kh < orig_kh else "")
        content = f"<b>Tuyến {route_id}{arrow}</b><br>{int(val)} KH" if mode_key == "Chế độ 1" else f"<b>Tuyến {route_id}{arrow}</b><br>{item['count']} KH<br>{val:.1f} h/ngày"
        html_items.append(f'<span style="display: inline-block; padding: 5px 15px; text-align: center; border-right: 1px solid #eee; line-height: 1.4;">{content}</span>')

    st.markdown(f"""<div style="overflow-x: auto; white-space: nowrap; display: inline-block; max-width: 100%; border: 1px solid #eee; border-radius: 5px; background-color: #f9f9f9;">{''.join(html_items)}</div>""", unsafe_allow_html=True)
    st.divider()

    # --- C. QUẢN LÝ STATE UI ---
    if 'editor_filter_mode' not in st.session_state: st.session_state.editor_filter_mode = 'all'
    if 'editor_filter_key' not in st.session_state: st.session_state.editor_filter_key = None
    if 'confirm_reset' not in st.session_state: st.session_state.confirm_reset = False
    
    # State Map Object (Cache cho bản đồ hiện tại - Editable)
    map_key_obj = f"map_obj_{mode_key}"
    if map_key_obj not in st.session_state: st.session_state[map_key_obj] = None
    if 'map_needs_refresh' not in st.session_state: st.session_state.map_needs_refresh = True

    c_map, c_edit = st.columns([3, 1])

    with c_map:
        # [OPTIMIZE]: Chỉ tạo lại map object khi cần thiết
        if st.session_state[map_key_obj] is None or st.session_state.map_needs_refresh:
            # Nếu vừa reset xong -> Lấy snapshot gắn vào luôn (Không tính toán lại)
            if st.session_state.get('just_reset', False):
                st.session_state[map_key_obj] = st.session_state[key_snapshot]
                st.session_state.just_reset = False
            else:
                m_obj, _ = generate_folium_map(df_saved, mapping, st.session_state.time_matrix, mode=mode_key)
                st.session_state[map_key_obj] = m_obj
            
            st.session_state.map_needs_refresh = False
        
        # [OPTIMIZE]: st_folium
        map_data = None
        if st.session_state[map_key_obj]:
            map_data = st_folium(
                st.session_state[map_key_obj], 
                center=[df_saved[mapping['lat']].mean(), df_saved[mapping['lon']].mean()], 
                zoom=11, 
                height=600, 
                returned_objects=["last_object_clicked"], 
                key=f"map_v{st.session_state.map_version}", 
                use_container_width=True
            )

        # [OPTIMIZE]: Xử lý click mà KHÔNG RERUN ở đây
        if map_data and map_data.get("last_object_clicked"):
            clicked_obj = map_data["last_object_clicked"]
            c_lat, c_lng = clicked_obj['lat'], clicked_obj['lng']
            
            mask = (np.isclose(df_saved[mapping['lat']], c_lat, atol=1e-5)) & \
                   (np.isclose(df_saved[mapping['lon']], c_lng, atol=1e-5))
            found_rows = df_saved[mask]
            
            if not found_rows.empty:
                found_code = found_rows.iloc[0][mapping['customer_code']]
                if st.session_state.editor_filter_key != found_code:
                    st.session_state.editor_filter_mode = 'single'
                    st.session_state.editor_filter_key = found_code

        # Legend
        colors = ["#FF0000", "#0000FF", "#00FF00", "#FFFF00", "#FF00FF", "#00FFFF", "#800000", "#008000", "#000080", "#FFA500"]
        unique_ids = sorted(df_saved['territory_id'].unique())
        color_map = {int(id): colors[(int(id) - 1) % len(colors)] for id in unique_ids}
        legend_html = " ".join([f'<span style="background-color:{c};width:10px;height:10px;display:inline-block;margin-right:3px;"></span>T{i}' for i, c in color_map.items()])
        st.markdown(legend_html, unsafe_allow_html=True)

    with c_edit:
        with st.expander("Chỉnh sửa thủ công", expanded=True):
            
            df_display_source = df_saved.copy()
            orig_map_dict = dict(zip(df_original[mapping['customer_code']], df_original['territory_id']))
            
            def check_change(row):
                code = row[mapping['customer_code']]
                curr_route = int(row['territory_id'])
                orig_route = int(orig_map_dict.get(code, curr_route))
                return "✏️" if curr_route != orig_route else ""

            df_display_source['Trạng thái'] = df_display_source.apply(check_change, axis=1)

            # Filter Logic
            filter_mode = st.session_state.editor_filter_mode
            
            if filter_mode == 'single' and st.session_state.editor_filter_key:
                df_display = df_display_source[df_display_source[mapping['customer_code']] == st.session_state.editor_filter_key]
                st.info(f"KH: {st.session_state.editor_filter_key}")
            
            elif filter_mode == 'changed':
                df_display = df_display_source[df_display_source['Trạng thái'] != ""]
                st.warning(f"Đang lọc {len(df_display)} KH thay đổi.")
            else:
                df_display = df_display_source

            # --- [UPDATED] Config Cột (Mode 2: Thêm Freq & Segment) ---
            cols_cfg = {}
            show_cols = ['Trạng thái', mapping['customer_code'], 'territory_id']
            if mapping.get('customer_name'): show_cols.append(mapping['customer_name'])
            
            # Chỉ thêm Tần suất & Phân loại nếu ở Mode 2
            if mode_key == "Chế độ 2":
                if mapping.get('freq'): show_cols.append(mapping['freq'])
                if mapping.get('type'): show_cols.append(mapping['type'])

            if mapping.get('address'): show_cols.append(mapping['address'])
            if mapping.get('vol_ec'): show_cols.append(mapping['vol_ec'])
            
            # Tính width nếu chưa có
            if not st.session_state.col_widths:
                for c in show_cols:
                    if c == 'territory_id': st.session_state.col_widths[c] = "small"
                    elif c == 'Trạng thái': st.session_state.col_widths[c] = 40
                    else:
                        sample = df_display_source[c].astype(str).head(50)
                        max_len = sample.str.len().max()
                        if pd.isna(max_len): max_len = 10
                        st.session_state.col_widths[c] = int(max(50, min(300, max_len * 8)))
            
            for c in df_display.columns:
                if c not in show_cols: 
                    cols_cfg[c] = None 
                else:
                    if c == 'territory_id':
                        t_options = [int(x) for x in sorted(df_saved['territory_id'].unique())]
                        cols_cfg[c] = st.column_config.SelectboxColumn("Tuyến", options=t_options, required=True, width="small")
                    else:
                        w = st.session_state.col_widths.get(c, 100)
                        cols_cfg[c] = st.column_config.TextColumn(c if c != 'Trạng thái' else "TT", width=w, disabled=True)

            # Render Editor
            editor_key = f"editor_{mode_key}_{filter_mode}_{st.session_state.editor_filter_key}"
            
            edited_data_sub = st.data_editor(
                df_display,
                column_config=cols_cfg,
                column_order=show_cols,
                use_container_width=True,
                hide_index=True,
                height=600 if filter_mode == 'all' else 200,
                key=editor_key
            )

            # Check Unsaved
            has_unsaved_changes = False
            try:
                input_routes = df_display['territory_id'].astype(str).values
                output_routes = edited_data_sub['territory_id'].astype(str).values
                if not np.array_equal(input_routes, output_routes):
                    has_unsaved_changes = True
            except: pass 

            if has_unsaved_changes:
                st.warning("Có thay đổi chưa lưu. Nếu Lọc bây giờ, các sửa đổi này sẽ mất.", icon="⚠️")
            # Buttons (Hiển thị ở cả 2 chế độ)
            c_update, c_filter_change, c_clear = st.columns([1.2, 1, 0.8])
            
            with c_update:
                btn_update = st.button("💾 Cập nhật", use_container_width=True, type="primary")
            with c_filter_change:
                btn_show_changed = st.button("⚠️ Tuyến đã đổi", use_container_width=True, disabled=(filter_mode == 'changed'))
            with c_clear:
                btn_clear = st.button("🔄 Bỏ lọc", use_container_width=True, disabled=(filter_mode == 'all'))

            st.divider()
            
            if not st.session_state.confirm_reset:
                if st.button("Hủy bỏ & Reset", type="secondary", use_container_width=True):
                    st.session_state.confirm_reset = True
                    st.rerun()
            else:
                st.error("Quay về gốc? Mất dữ liệu đã sửa.")
                c_yes, c_no = st.columns(2)
                if c_yes.button("✅ Đồng ý", use_container_width=True, type="primary"):
                    # [OPTIMIZE]: RESET SIÊU TỐC
                    st.session_state[key_saved] = df_original.copy()
                    st.session_state.editor_filter_mode = 'all'
                    st.session_state.editor_filter_key = None
                    st.session_state.confirm_reset = False
                    
                    st.session_state.just_reset = True 
                    st.session_state.map_needs_refresh = True
                    st.session_state.map_version += 1
                    
                    st.success("Đã Reset!")
                    st.rerun()
                if c_no.button("❌ Hủy", use_container_width=True):
                    st.session_state.confirm_reset = False
                    st.rerun()

            # Handle Actions
            if btn_clear:
                st.session_state.editor_filter_mode = 'all'
                st.session_state.editor_filter_key = None
                st.rerun()

            if btn_show_changed:
                st.session_state.editor_filter_mode = 'changed'
                st.session_state.editor_filter_key = None
                st.rerun()

            if btn_update:
                new_map = dict(zip(edited_data_sub[mapping['customer_code']], edited_data_sub['territory_id']))
                def update_route_logic(row):
                    code = row[mapping['customer_code']]
                    return new_map.get(code, row['territory_id'])
                
                df_to_save = df_saved.copy()
                df_to_save['territory_id'] = df_to_save.apply(update_route_logic, axis=1)

                st.session_state[key_saved] = df_to_save
                st.session_state.map_needs_refresh = True
                st.session_state.map_version += 1
                
                st.success("Đã cập nhật!")
                st.rerun()

    _, c_dl, _ = st.columns([1,1,1])
    with c_dl:
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            df_saved.to_excel(writer, index=False)
        st.download_button("Tải về file Excel kết quả", buffer.getvalue(), "Result.xlsx")