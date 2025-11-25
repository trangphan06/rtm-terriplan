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
if 'v1_df_edited' not in st.session_state: st.session_state.v1_df_edited = None
if 'v1_df_map' not in st.session_state: st.session_state.v1_df_map = None
if 'v1_report' not in st.session_state: st.session_state.v1_report = None

if 'v2_df_edited' not in st.session_state: st.session_state.v2_df_edited = None
if 'v2_df_map' not in st.session_state: st.session_state.v2_df_map = None
if 'v2_report' not in st.session_state: st.session_state.v2_report = None

# --- CẤU HÌNH TRANG & CSS ---
st.set_page_config(
    layout="wide",
    page_title="Territory Planner Pro",
    initial_sidebar_state=st.session_state.sidebar_state 
)

st.markdown("""
    <style>
        .block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
            padding-left: 1rem;
            padding-right: 1rem;
        }
        iframe {
            width: 100% !important;
        }
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

@st.cache_data
def get_farthest_distance(_group_df, lat_col, lon_col):
    if len(_group_df) < 2: return 0
    if len(_group_df) > 500: return -1
    max_dist = 0
    lats = pd.to_numeric(_group_df[lat_col], errors='coerce').values
    lons = pd.to_numeric(_group_df[lon_col], errors='coerce').values
    points = list(zip(lats, lons))
    for p1, p2 in combinations(points, 2):
        dist = haversine(p1[0], p1[1], p2[0], p2[1])
        if dist > max_dist: max_dist = dist
    return max_dist

# --- LOGIC VERSION 1 ---
def run_territory_planning_v1(df, lat_col, lon_col, n_clusters, min_size, max_size, n_init=50):
    df_run = df.copy()
    # Ép kiểu số
    df_run[lat_col] = pd.to_numeric(df_run[lat_col], errors='coerce')
    df_run[lon_col] = pd.to_numeric(df_run[lon_col], errors='coerce')
    df_run = df_run.dropna(subset=[lat_col, lon_col])
    
    coords = df_run[[lat_col, lon_col]]
    scaler = StandardScaler()
    coords_scaled = scaler.fit_transform(coords)
    
    if min_size * n_clusters > len(df): return None, "Lỗi: Số lượng tối thiểu quá lớn."
    if max_size * n_clusters < len(df): return None, "Lỗi: Số lượng tối đa quá nhỏ."

    progress_text = "Đang tối ưu hóa phân cụm (V1)..."
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
            my_bar.progress((i + 1) / n_init, text=f"Đang phân tuyến... {percent}%")
            
        my_bar.empty()
        df_run['territory_id'] = best_clf.labels_ + 1
        
        # Stats gốc V1
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
    
    TARGET_POINTS = 50000 # Tối ưu tốc độ
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
    
    n_init = 5 # Tối ưu tốc độ
    progress_text = f"Đang cân bằng khối lượng công việc ({n_init} lần lặp)..."
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
            my_bar.progress((i + 1) / n_init, text=f"Đang xử lý Workload (Lần {i+1}/{n_init})... {percent}%")
        
        my_bar.empty()
        
        df_exploded['territory_id'] = best_clf.labels_ + 1
        final_labels = df_exploded.groupby('original_index')['territory_id'].agg(lambda x: x.mode()[0])
        df_run['territory_id'] = final_labels
        
        # Thống kê gốc V2
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

@st.cache_data(show_spinner="Đang tạo bản đồ...")
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
    
    for _, row in df_plot.iterrows():
        tooltip_txt = f"<b>KH: {row[col_code]}</b><br>Tuyến: {row['territory_id']}"
        
        if col_name and col_name in row and pd.notna(row[col_name]): 
            tooltip_txt += f"<br>Tên: {row[col_name]}"
        if col_addr and col_addr in row and pd.notna(row[col_addr]): 
            tooltip_txt += f"<br>Đ/c: {row[col_addr]}"
        if col_vol and col_vol in row and pd.notna(row[col_vol]): 
            tooltip_txt += f"<br>Vol: {row[col_vol]}"
        
        # Tooltip V2: Tần suất, Segment, Thời gian visit
        if mode == "Chế độ 2":
            if col_freq and col_freq in row:
                tooltip_txt += f"<br>Tần suất: {row[col_freq]}"
            if col_type and col_type in row:
                seg_val = str(row[col_type]).strip()
                tooltip_txt += f"<br>Phân loại: {seg_val}"
                # Tra cứu thời gian từ matrix
                time_val = _time_matrix.get(seg_val, _time_matrix.get('default', 10.0))
                tooltip_txt += f"<br>Thời gian visit: {time_val}p"

        folium.CircleMarker(
            location=[row[lat_col], row[lon_col]],
            radius=4,
            color=row['color'], fill=True, fill_color=row['color'], fill_opacity=0.7,
            tooltip=tooltip_txt
        ).add_to(m)
    return m, map_center

# ==========================================
# 2. GIAO DIỆN (UI)
# ==========================================

st.title("Công cụ Phân chia Địa bàn (Territory Plan)")

# Kiểm tra xem chế độ hiện tại có dữ liệu chưa
current_mode = "Chế độ 1" # Default, sẽ được cập nhật ở Sidebar
# Biến để lưu dữ liệu hiển thị hiện tại (V1 hoặc V2)
current_df_edited = None
current_df_map = None
current_report = None

with st.sidebar:
    st.header("1. Chế độ Phân tuyến")
    plan_mode = st.radio(
        "Chọn chế độ:",
        ["Chế độ 1: Cân bằng Số lượng KH", "Chế độ 2: Cân bằng Số giờ viếng thăm"],
        index=0
    )
    mode_key = "Chế độ 1" if "Chế độ 1" in plan_mode else "Chế độ 2"
    current_mode = mode_key

    # Lấy dữ liệu tương ứng với Mode
    if mode_key == "Chế độ 1":
        current_df_edited = st.session_state.v1_df_edited
        current_df_map = st.session_state.v1_df_map
        current_report = st.session_state.v1_report
    else:
        current_df_edited = st.session_state.v2_df_edited
        current_df_map = st.session_state.v2_df_map
        current_report = st.session_state.v2_report
    
    # [FIX STATE]: Nếu đổi chế độ -> Rerun để UI update theo biến current_df_...
    if st.session_state.get('last_mode') != mode_key:
        st.session_state.last_mode = mode_key
        st.rerun()

    st.divider()

    st.header("2. Tải dữ liệu")

    @st.cache_data
    def create_template_v1():
        df = pd.DataFrame({
            "Customer Code (*)": ["KH01", "KH02"], "Latitude (*)": [10.7, 10.8], "Longitude (*)": [106.6, 106.7],
            "Customer Name": ["A", "B"], "Address": ["HCM", "HCM"], "VolEC/month": [100, 200],
            "Frequency": [4, 8], "Customer Type": ["MT", "Cooler"]
        })
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer: df.to_excel(writer, index=False)
        return output.getvalue()

    @st.cache_data
    def create_template_v2():
        df = pd.DataFrame({
            "Customer Code (*)": ["KH01", "KH02"], "Latitude (*)": [10.7, 10.8], "Longitude (*)": [106.6, 106.7],
            "Customer Name": ["A", "B"], "Address": ["HCM", "HCM"], "VolEC/month": [100, 200],
            "Frequency (*)": [4, 8], "Customer Type (*)": ["MT", "Cooler"]
        })
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer: df.to_excel(writer, index=False)
        return output.getvalue()

    template_data = create_template_v1() if mode_key == "Chế độ 1" else create_template_v2()
    st.download_button(f"Tải template mẫu ({mode_key})", template_data, f"Template_{mode_key}.xlsx", use_container_width=True)

    uploaded_file = st.file_uploader("Upload", type=['xlsx', 'xls'], label_visibility="collapsed")

    if uploaded_file:
        try:
            # Dùng hàm cached để load file
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
                        type_col = st.selectbox("Phân loại Segment", options_cols, index=get_idx("Customer Type", options_cols, True))
                    else:
                        type_col = st.selectbox("Phân loại Segment (*)", all_cols, index=get_idx("Customer Type", all_cols))
                
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
                
                # Validate Numeric Data
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

    # --- 4. THAM SỐ CHẠY (CHỈ HIỆN KHI ĐÃ CONFIRM MAPPING) ---
    if st.session_state.get('mapping_confirmed') and st.session_state.df is not None:
        st.divider()
        st.header("4. Tham số Phân tuyến")

        n_routes = st.number_input("Số tuyến (Routes)", 1, 100, 9)
        current_mapping = st.session_state.col_mapping

        if mode_key == "Chế độ 1":
            avg_qty = len(st.session_state.df) // n_routes
            st.caption(f"Trung bình: ~{avg_qty} KH/tuyến")

            sug_min_v1 = int(avg_qty*0.9)
            sug_max_v1 = int(avg_qty*1.1)

            min_v = st.number_input("Số KH tối thiếu", 0, value=sug_min_v1)
            st.caption(f"Đề xuất: {sug_min_v1} KH/tuyến - thấp hơn 10% so với trung bình")

            max_v = st.number_input("Số KH tối đa", 0, value=sug_max_v1)
            st.caption(f"Đề xuất: {sug_max_v1} KH/tuyến - cao hơn 10% so với trung bình")

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
                st.caption(f"Tổng thời gian viếng thăm: {total_min/60:.1f}h. Trung bình thời gian viếng thăm: ~{avg_hours:.1f}h/tuyến")

                sug_min_h = round(avg_hours*0.9, 1)
                sug_max_h = round(avg_hours*1.1, 1)

                min_h = st.number_input("Số giờ viếng thăm tối thiểu", 0.0, value=sug_min_h, step=0.5)
                st.caption(f"Đề xuất: {sug_min_h} giờ/tuyến - thấp hơn 10% so với trung bình")

                max_h = st.number_input("Số giờ viếng thăm tối đa", 0.0, value=sug_max_h, step=0.5)
                st.caption(f"Đề xuất: {sug_max_h} giờ/tuyến - cao hơn 10% so với trung bình")
            except Exception as e:
                st.error(f"Lỗi tính toán: {e}. Vui lòng kiểm tra cột Tần suất/Phân loại.")
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
                    err = "Lỗi: Với Chế độ 2, bạn BẮT BUỘC phải chọn cột Tần suất và Phân loại."
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
                # LƯU VÀO STATE RIÊNG BIỆT
                if mode_key == "Chế độ 1":
                    st.session_state.v1_df_edited = res_df.copy()
                    st.session_state.v1_df_map = res_df.copy()
                    st.session_state.v1_report = res_stats.copy()
                else:
                    st.session_state.v2_df_edited = res_df.copy()
                    st.session_state.v2_df_map = res_df.copy()
                    st.session_state.v2_report = res_stats.copy()

                if st.session_state.sidebar_state == 'expanded':
                    st.session_state.sidebar_state = 'collapsed'
                    st.rerun()

# ==========================================
# 5. KHU VỰC HIỂN THỊ KẾT QUẢ
# ==========================================

# Hiển thị thông báo nếu chưa có data cho mode hiện tại
if current_df_edited is None:
    st.info(f"Vui lòng chạy phân tuyến cho {mode_key} để xem kết quả.")
else:
    # Gán biến cục bộ để dùng chung logic hiển thị
    df_map = current_df_map
    df_edited = current_df_edited
    report_df = current_report
    mapping = st.session_state.col_mapping

    colors = ["#FF0000", "#0000FF", "#00FF00", "#FFFF00", "#FF00FF", "#00FFFF", "#800000", "#008000", "#000080", "#FFA500"]
    unique_ids = sorted(df_map['territory_id'].unique())
    color_map = {int(id): colors[(int(id) - 1) % len(colors)] for id in unique_ids}
    df_map['color'] = df_map['territory_id'].map(color_map)

    st.header(f"BẢN ĐỒ PHÂN TUYẾN ({mode_key})")

    # --- TÍNH TOÁN SỐ LIỆU HIỆN TẠI (THEO MODE) ---
    if mode_key == "Chế độ 1":
        # V1: Chỉ đếm số lượng
        curr_stats = df_map['territory_id'].value_counts().sort_index()
        
        display_data = []
        for route_id in unique_ids:
            val = curr_stats.get(route_id, 0)
            display_data.append({'id': route_id, 'val': val, 'count': val})
    else:
        # V2: Tính Workload_Day
        grouped_v2 = df_map.groupby('territory_id').agg(
            count=('territory_id', 'count'),
            workload_min=('workload_min', 'sum')
        ).sort_index()
        
        display_data = []
        for route_id in unique_ids:
            if route_id in grouped_v2.index:
                row_data = grouped_v2.loc[route_id]
                kh_count = row_data['count']
                hours_per_day = row_data['workload_min'] / 60 / 22 # Tính giờ/ngày
                display_data.append({'id': route_id, 'val': hours_per_day, 'count': kh_count})
            else:
                display_data.append({'id': route_id, 'val': 0, 'count': 0})

    # --- HIỂN THỊ BẢNG TÓM LƯỢC ---
    html_items = []
    for item in display_data:
        route_id = item['id']
        val = item['val']
        current_kh = item['count']
        
        # [UPDATE]: Mũi tên dựa trên SỐ LƯỢNG KH (Cả V1 và V2)
        # Lấy số KH gốc
        orig_kh = report_df.set_index('Tuyến (RouteID)')['Số lượng KH'].get(route_id, 0)

        arrow = ""
        # So sánh số lượng KH hiện tại và gốc
        if current_kh > orig_kh: arrow = " 🔼"
        elif current_kh < orig_kh: arrow = " 🔽"

        if mode_key == "Chế độ 1":
            content = f"<b>Tuyến {route_id}{arrow}</b><br>{int(val)} KH"
        else:
            kh = item['count']
            # V2: Số KH (dòng 1) | Số giờ/ngày (dòng 2)
            content = f"<b>Tuyến {route_id}{arrow}</b><br>{kh} KH<br>{val:.1f} h/ngày"

        html_item = f'<span style="display: inline-block; padding: 5px 15px; text-align: center; border-right: 1px solid #eee; line-height: 1.4;">{content}</span>'
        html_items.append(html_item)

    st.markdown(
        f"""<div style="overflow-x: auto; white-space: nowrap; display: inline-block; max-width: 100%; border: 1px solid #eee; border-radius: 5px; background-color: #f9f9f9;">{''.join(html_items)}</div>""",
        unsafe_allow_html=True
    )
    st.divider()

    # --- LAYOUT 3:1 ---
    c_map, c_edit = st.columns([3, 1])

    with c_map:
        m, _ = generate_folium_map(df_map, mapping, st.session_state.time_matrix, mode=mode_key)
        if m:
            st_folium(m, center=[df_map[mapping['lat']].mean(), df_map[mapping['lon']].mean()], zoom=11, height=600, returned_objects=[], use_container_width=True)

        col_btn, col_legend = st.columns([1, 3])
        with col_btn:
            update_clicked = st.button("Cập nhật", use_container_width=True)
        with col_legend:
            legend_html = " ".join([f'<span style="background-color:{c};width:10px;height:10px;display:inline-block;margin-right:3px;"></span>T{i}' for i, c in color_map.items()])
            st.markdown(legend_html, unsafe_allow_html=True)

    with c_edit:
        with st.expander("Chỉnh sửa thủ công", expanded=True):
            
            cols_cfg = {}
            show_cols = [mapping['customer_code'], 'territory_id']
            if mapping.get('customer_name'): show_cols.insert(1, mapping['customer_name'])
            if mapping.get('address'): show_cols.insert(2, mapping['address'])
            if mapping.get('vol_ec'): show_cols.insert(3, mapping['vol_ec'])

            df_editor_instance = df_edited 
            for c in df_editor_instance.columns:
                if c not in show_cols: cols_cfg[c] = None
                else:
                    if c == 'territory_id':
                        t_options = [int(x) for x in sorted(df_map['territory_id'].unique())]
                        cols_cfg[c] = st.column_config.SelectboxColumn("Tuyến", options=t_options, required=True)
                    else:
                        cols_cfg[c] = st.column_config.TextColumn(c, disabled=True)

            edited_data = st.data_editor(
                df_editor_instance,
                column_config=cols_cfg,
                use_container_width=True,
                height=600,
                key=f"editor_key_{mode_key}" 
            )

    if update_clicked:
        if mode_key == "Chế độ 1":
            st.session_state.v1_df_edited = edited_data.copy()
            st.session_state.v1_df_map = edited_data.copy()
        else:
            st.session_state.v2_df_edited = edited_data.copy()
            st.session_state.v2_df_map = edited_data.copy()
        
        generate_folium_map.clear()
        st.rerun()

    st.divider()

    _, c_dl, _ = st.columns([1,1,1])
    with c_dl:
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            df_edited.to_excel(writer, index=False)
        st.download_button("Tải về file Excel kết quả", buffer.getvalue(), "Result.xlsx")