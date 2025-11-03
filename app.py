import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from k_means_constrained import KMeansConstrained
import numpy as np
import io
from itertools import combinations
import folium               
from streamlit_folium import st_folium 

# --- CẤU HÌNH TRANG ---
st.set_page_config(
    layout="wide",
    page_title="Công cụ Phân chia Địa bàn (Territory Plan)"
)

# --- CÁC HÀM HỖ TRỢ ---
# (Các hàm haversine, get_farthest_distance, run_territory_planning
#  giữ nguyên như cũ. Không cần thay đổi.)
@st.cache_data
def haversine(lat1, lon1, lat2, lon2):
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
    max_dist = 0
    if len(_group_df) < 2: return 0
    if len(_group_df) > 500: return -1
    for (i, p1), (j, p2) in combinations(_group_df.iterrows(), 2):
        dist = haversine(p1[lat_col], p1[lon_col], p2[lat_col], p2[lon_col])
        if dist > max_dist:
            max_dist = dist
    return max_dist
def run_territory_planning(df, lat_col, lon_col, n_clusters, min_size, max_size, n_init):
    df_original = df.copy()
    coords = df_original[[lat_col, lon_col]]
    scaler = StandardScaler()
    coords_scaled = scaler.fit_transform(coords)
    if min_size is None or min_size == 0: min_size = 1
    if max_size is None or max_size == 0: max_size = len(df)
    if min_size * n_clusters > len(df):
        st.error(f"Lỗi Ràng buộc: Yêu cầu tối thiểu ({min_size} min * {n_clusters} tuyến = {min_size * n_clusters} KH) > Tổng số KH ({len(df)})")
        return None, None
    if max_size * n_clusters < len(df):
        st.error(f"Lỗi Ràng buộc: Khả năng tối đa ({max_size} max * {n_clusters} tuyến = {max_size * n_clusters} KH) < Tổng số KH ({len(df)})")
        return None, None
    best_model = None
    best_inertia = np.inf
    progress_bar = st.progress(0, text="Đang chạy phân cụm...")
    try:
        for i in range(n_init):
            model = KMeansConstrained(
                n_clusters=n_clusters, size_min=min_size, size_max=max_size,
                random_state=42 + i, n_init=1
            )
            model.fit(coords_scaled)
            if model.inertia_ < best_inertia:
                best_inertia = model.inertia_
                best_model = model
            progress_bar.progress((i + 1) / n_init, text=f"Đang chạy lần {i + 1}/{n_init}")
    except Exception as e:
        progress_bar.empty()
        st.error(f"LỖI CHIA TUYẾN: {e}")
        st.error("Ràng buộc (min/max) quá chặt. Hãy thử nới lỏng.")
        return None, None
    progress_bar.empty()
    if best_model is None:
        st.error("Lỗi: Không thể hoàn tất chia tuyến.")
        return None, None
    df_original['territory_id'] = best_model.labels_ + 1
    cluster_counts = df_original['territory_id'].value_counts().sort_index()
    report_df = pd.DataFrame({
        "Tuyến (RouteID)": cluster_counts.index,
        "Số lượng KH": cluster_counts.values
    })
    return df_original, report_df

@st.cache_data(show_spinner="Đang tạo bản đồ...")
def generate_folium_map(_df, _mapping):
    if _df.empty:
        return None, None
    map_center = [_df[_mapping['lat']].mean(), _df[_mapping['lon']].mean()]
    m = folium.Map(location=map_center, zoom_start=11)
    for _, row in _df.iterrows():
        folium.CircleMarker(
            location=[row[_mapping['lat']], row[_mapping['lon']]],
            radius=5,
            color=row['color'], fill=True, fill_color=row['color'], fill_opacity=0.7,
            tooltip=f"<b>KH: {row[_mapping['customer_code']]}</b><br>Tuyến: {row['territory_id']}"
        ).add_to(m)
    return m, map_center

# --- KHỞI TẠO SESSION STATE ---
if 'df' not in st.session_state: st.session_state.df = None
if 'col_mapping' not in st.session_state: st.session_state.col_mapping = {}
if 'report_df' not in st.session_state: st.session_state.report_df = None
# *** THAY ĐỔI: Tách 2 state cho map và editor ***
if 'df_map' not in st.session_state: st.session_state.df_map = None # Dữ liệu cho bản đồ
if 'df_edited' not in st.session_state: st.session_state.df_edited = None # Dữ liệu cho editor/báo cáo

# --- GIAO DIỆN CHÍNH ---
st.title("Công cụ Phân chia Địa bàn (Territory Plan)")
st.info("Tải dữ liệu & điều chỉnh tham số ở thanh bên trái để bắt đầu.")

# --- THANH BÊN (SIDEBAR) ---
# (Phần sidebar giữ nguyên, trừ phần cuối cùng)
with st.sidebar:
    st.header("1. Tải lên dữ liệu")
    uploaded_file = st.file_uploader(
        "1. Tải lên file Excel", type=['xlsx', 'xls'],
        help="File phải chứa cột Customer Code, Vĩ độ (latitude) và Kinh độ (longitude)."
    )
    st.caption("Dạng file: .xlsx, .xls. Giới hạn: 200MB.")
    if uploaded_file:
        try:
            df = pd.read_excel(uploaded_file)
            st.session_state.df = df
            all_cols = df.columns.tolist()
            st.subheader("2. Chọn cột")
            st.info("Chọn các cột tương ứng từ file của bạn.")
            col_customer_code = st.selectbox("Customer Code", all_cols, index=all_cols.index(all_cols[0]) if all_cols else 0)
            col_lat = st.selectbox("Vĩ độ (Latitude)", all_cols, index=all_cols.index('lat') if 'lat' in all_cols else 1)
            col_lon = st.selectbox("Kinh độ (Longitude)", all_cols, index=all_cols.index('long') if 'long' in all_cols else 2)
            st.session_state.col_mapping = {"customer_code": col_customer_code, "lat": col_lat, "lon": col_lon}
            st.subheader("3. Kiểm tra dữ liệu")
            total_customers = len(df)
            st.metric("Tổng số khách hàng (dòng)", total_customers)
            required_cols = [col_customer_code, col_lat, col_lon]
            duplicates = df.duplicated(subset=required_cols).sum()
            if duplicates > 0:
                st.warning(f"Tìm thấy {duplicates} dòng bị trùng (duplicate).")
            else:
                st.success("File không có dữ liệu trùng.")
        except Exception as e:
            st.error(f"Lỗi khi đọc file: {e}")
            st.session_state.df = None
    if st.session_state.df is not None:
        st.divider()
        st.header("⚙️ Điều chỉnh Tham số")
        total_customers = len(st.session_state.df)
        mapping = st.session_state.col_mapping
        n_routes = st.number_input("Số lượng RouteID/Số SR", min_value=1, value=9, step=1)
        avg_customers = 0
        if n_routes > 0:
            avg_customers = total_customers // n_routes
            st.info(f"Ước tính: ~{avg_customers} KH/tuyến")
        suggested_min = int(avg_customers * 0.8)
        suggested_max = int(avg_customers * 1.2)
        min_customers = st.number_input(
            "Số KH tối thiểu trên tuyến", min_value=0, value=suggested_min, step=1,
            help=f"Gợi ý: Số KH tối thiểu nên từ {suggested_min} trở lên."
        )
        st.caption(f"Gợi ý dựa trên mức trung bình: {suggested_min} (dưới 20%)")
        max_customers = st.number_input(
            "Số KH tối đa trên tuyến", min_value=0, value=suggested_max, step=1,
            help=f"Gợi ý: Số KH tối đa nên từ {suggested_max} trở xuống."
        )
        st.caption(f"Gợi ý dựa trên mức trung bình: {suggested_max} (trên 20%)")
        n_init_runs = st.number_input("Số lần chạy (n_init)", min_value=1, value=50, step=10)
        st.caption("Đề xuất: Nhập 50 để có kết quả tốt nhất.")
        run_button = st.button("Bắt đầu phân tuyến", type="primary", use_container_width=True)
        if run_button:
            if not mapping.get("customer_code") or not mapping.get("lat") or not mapping.get("lon"):
                st.error("Lỗi: Vui lòng chọn đủ 3 cột cần thiết để phân tuyến.")
            elif min_customers > max_customers:
                st.error("Lỗi: Số KH tối thiểu không thể lớn hơn số KH tối đa.")
            else:
                with st.spinner("Đang xử lý..."):
                    try:
                        df_result, report_df = run_territory_planning(
                            df=st.session_state.df, lat_col=mapping['lat'], lon_col=mapping['lon'],
                            n_clusters=n_routes, min_size=min_customers, max_size=max_customers,
                            n_init=n_init_runs
                        )
                        if df_result is not None:
                            # *** THAY ĐỔI: Lưu kết quả vào CẢ HAI state ***
                            st.session_state.df_map = df_result.copy()
                            st.session_state.df_edited = df_result.copy()
                            st.session_state.report_df = report_df
                            st.success("Phân tuyến thành công!")
                    except Exception as e:
                        st.error(f"Lỗi không xác định: {e}")
                        st.exception(e)

# --- KHU VỰC HIỂN THỊ KẾT QUẢ ---
if st.session_state.df_edited is not None:
    # Lấy dữ liệu từ state
    df_map = st.session_state.df_map
    df_edited = st.session_state.df_edited
    report_df = st.session_state.report_df
    mapping = st.session_state.col_mapping

    st.header("📊 Kết quả tóm lược")
    st.dataframe(report_df.set_index('Tuyến (RouteID)'))
    
    # Thêm màu (việc này nhanh, không cần cache)
    colors_list = ["#FF0000", "#0000FF", "#00FF00", "#FFFF00", "#FF00FF", "#00FFFF", "#800000", "#008000", "#000080", "#FFA500"]
    color_map = {id: colors_list[(id - 1) % len(df_map['territory_id'].unique())] for id in df_map['territory_id'].unique()}
    df_map['color'] = df_map['territory_id'].map(color_map)

    # --- BẢN ĐỒ ---
    st.header("🗺️ Bản đồ phân tuyến")
    
    # *** THAY ĐỔI: Nút bấm để tải lại bản đồ ***
    if st.button("Tải lại bản đồ (với các thay đổi từ bảng chỉnh sửa)"):
        # Cập nhật state của bản đồ = state của editor
        st.session_state.df_map = st.session_state.df_edited.copy()
        # Xóa cache của hàm generate_folium_map để nó build lại
        generate_folium_map.clear()
        st.rerun() # Chạy lại script để hiển thị map mới

    all_tuyen = [int(x) for x in sorted(df_map['territory_id'].unique())]

    # Tạo chú giải (Legend)
    legend_items = []
    for tuyen_id in all_tuyen:
        color = color_map.get(tuyen_id) # Dùng .get() để an toàn hơn
        if color:
            legend_items.append(
                f'<span style="background-color: {color}; width: 12px; height: 12px; display: inline-block; margin-right: 5px; border: 1px solid #000;"></span> Tuyến {tuyen_id}'
            )
    st.markdown("<b>Chú giải màu:</b>&nbsp;&nbsp;&nbsp;" + "&nbsp;&nbsp;&nbsp;".join(legend_items), unsafe_allow_html=True)
    
    
    # Gọi hàm đã cache
    m, map_center = generate_folium_map(df_map, mapping)
    
    if m:
        # *** THAY ĐỔI: Thêm returned_objects=[] để chặn lag ***
        st_folium(
            m, 
            center=map_center, 
            zoom=11, 
            use_container_width=True, 
            height=500,
            returned_objects=[] # Quan trọng: Chặn zoom/pan gửi tín hiệu về
        )
    
    st.caption("Ghi chú: Di chuột qua các điểm để xem Customer Code.")

    # --- CHỈNH SỬA THỦ CÔNG ---
    with st.expander("✍️ Bảng chỉnh sửa thủ công (Click để mở)"):
        st.warning("Lưu ý: Sau khi sửa, hãy nhấn nút 'Tải lại bản đồ' ở trên để xem thay đổi.")
        
        # *** THAY ĐỔI: Bảng này chỉ đọc/ghi vào 'df_edited' ***
        all_tuyen_options = [int(x) for x in sorted(st.session_state.df_edited['territory_id'].unique())]
        
        edited_df = st.data_editor(
            st.session_state.df_edited, # Đọc từ 'df_edited'
            column_config={
                "territory_id": st.column_config.SelectboxColumn("Tuyến", options=all_tuyen_options, required=True),
                "color": None, 
            },
            use_container_width=True, num_rows="dynamic", key="data_editor"
        )
        # *** THAY ĐỔI: Lưu lại vào 'df_edited' ***
        st.session_state.df_edited = edited_df
    
    # --- BÁO CÁO CHI TIẾT ---
    st.header("📋 Chi tiết từng tuyến (Read-Only)")
    st.caption("Dữ liệu này được cập nhật dựa trên bảng chỉnh sửa thủ công. Click vào từng tuyến để xem chi tiết.")
    
    # *** THAY ĐỔI: Báo cáo luôn đọc từ 'df_edited' ***
    df_for_report = st.session_state.df_edited
    
    for tuyen_id in sorted(df_for_report['territory_id'].unique()):
        with st.expander(f"### Tuyến {tuyen_id} (Click để xem chi tiết)"):
            group_df = df_for_report[df_for_report['territory_id'] == tuyen_id]
            total_kh = len(group_df)
            farthest_dist_km = get_farthest_distance(group_df, mapping['lat'], mapping['lon'])
            col1, col2 = st.columns(2)
            col1.metric("Tổng số khách hàng", total_kh)
            if farthest_dist_km == -1:
                col2.metric("Khoảng cách xa nhất", "Quá lớn (>500 KH) để tính.")
            else:
                col2.metric("Khoảng cách xa nhất", f"{farthest_dist_km:.2f} km")
            st.dataframe(group_df, use_container_width=True, hide_index=True)

    # --- NÚT DOWNLOAD ---
    st.header("📥 Tải về")
    st.caption("File tải về sẽ chứa các dữ liệu MỚI NHẤT từ bảng chỉnh sửa thủ công.")
    output_buffer = io.BytesIO()
    with pd.ExcelWriter(output_buffer, engine='openpyxl') as writer:
        # *** THAY ĐỔI: Luôn tải về 'df_edited' ***
        st.session_state.df_edited.to_excel(writer, index=False, sheet_name='Territory_Output')
    st.download_button(
        label="Tải file Excel kết quả (Đã chỉnh sửa)",
        data=output_buffer.getvalue(),
        file_name="territory_output_edited.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )