import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from k_means_constrained import KMeansConstrained
import numpy as np
import io
from itertools import combinations
import folium               
from streamlit_folium import st_folium 

# --- KHỞI TẠO SIDEBAR STATE ---
if 'sidebar_state' not in st.session_state:
    st.session_state.sidebar_state = 'expanded' 

# --- CẤU HÌNH TRANG ---
st.set_page_config(
    layout="wide",
    page_title="Công cụ Phân chia Địa bàn (Territory Plan)",
    initial_sidebar_state=st.session_state.sidebar_state 
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
            progress_value = (i + 1) / n_init
            progress_percent = int(progress_value * 100)
            progress_bar.progress(progress_value, text=f"Đang phân tuyến {progress_percent}%")
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
    col_code = _mapping['customer_code']
    col_name = _mapping.get('customer_name')
    col_addr = _mapping.get('address')
    col_vol = _mapping.get('vol_ec')
    for _, row in _df.iterrows():
        tooltip_html = f"<b>KH: {row[col_code]}</b><br>Tuyến: {row['territory_id']}"
        if col_name and col_name in row:
            tooltip_html += f"<br>Tên: {row[col_name]}"
        if col_addr and col_addr in row:
            tooltip_html += f"<br>Địa chỉ: {row[col_addr]}"
        if col_vol and col_vol in row:
            tooltip_html += f"<br>VolEC: {row[col_vol]}"
        folium.CircleMarker(
            location=[row[_mapping['lat']], row[_mapping['lon']]],
            radius=5,
            color=row['color'], fill=True, fill_color=row['color'], fill_opacity=0.7,
            tooltip=tooltip_html
        ).add_to(m)
    return m, map_center

# --- KHỞI TẠO SESSION STATE ---
if 'df' not in st.session_state: st.session_state.df = None
if 'col_mapping' not in st.session_state: st.session_state.col_mapping = {}
if 'original_report_df' not in st.session_state: st.session_state.original_report_df = None
if 'df_map' not in st.session_state: st.session_state.df_map = None 
if 'df_edited' not in st.session_state: st.session_state.df_edited = None 

# --- GIAO DIỆN CHÍNH ---
st.title("Công cụ Phân chia Địa bàn (Territory Plan)")
if st.session_state.df_edited is None:
    st.info("Tải dữ liệu & điều chỉnh tham số ở thanh bên trái để bắt đầu.")

# --- SIDEBAR ---
with st.sidebar:
    # (Toàn bộ code sidebar giữ nguyên)
    @st.cache_data 
    def create_template_excel():
        template_df = pd.DataFrame({
            "Customer Code (*)": ["1", "2", "3"],
            "Latitude (*)": [10.7769, 10.7765, 10.8231],
            "Longitude (*)": [106.7009, 106.7012, 106.6297],
            "Customer Name": ["Tap hoa co Hai", "Nha hang Nam Anh", "Quan nhau Ty"],
            "Address": ["123 Duong Nguyen Hue, Quan 1", "456 Duong Le Loi, Quan 1", "789 Duong Cong Hoa, Quan Tan Binh"],
            "VolEC/month": ["20", "150", "100"]
        })
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            template_df.to_excel(writer, index=False, sheet_name='Data')
        return output.getvalue()
    template_excel = create_template_excel()
    st.download_button(
        label="Tải template tại đây",
        data=template_excel,
        file_name="Template_PhanTuyen.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )
    st.divider() 
    
    st.header("1. Tải lên dữ liệu")
    uploaded_file = st.file_uploader(
        label="Tải lên file excel", 
        type=['xlsx', 'xls'],
        help=None,
        label_visibility="collapsed"
    )
    st.caption("File phải chứa cột Customer Code, Vĩ độ (latitude) và Kinh độ (Longitude).")
    
    if uploaded_file:
        try:
            df = pd.read_excel(uploaded_file, dtype={"Customer Code": str, "Customer Code (*)": str})
            st.session_state.df = df
            all_cols = df.columns.tolist()
            options_cols = ["[Bỏ qua]"] + all_cols
            
            st.subheader("2. Chọn cột")
            st.info("Chọn các cột tương ứng từ file của bạn.")
            
            col_map1, col_map2 = st.columns(2)
            
            def find_index(col_name, all_cols_list, is_optional=False):
                default_index = 0
                target_list = all_cols_list
                if is_optional:
                    target_list = options_cols
                    default_index = 0
                
                if col_name in target_list:
                    return target_list.index(col_name)
                variation_name = col_name.replace(" (*)", "")
                if variation_name in target_list:
                    return target_list.index(variation_name)
                
                return default_index

            with col_map1:
                col_customer_code = st.selectbox("Customer Code (*)", all_cols, 
                                                 index=find_index("Customer Code (*)", all_cols) or find_index("Customer Code", all_cols))
                col_lat = st.selectbox("Vĩ độ (Latitude) (*)", all_cols, 
                                       index=find_index("Latitude (*)", all_cols) or find_index("latitude", all_cols) or 1)
                col_name_select = st.selectbox("Customer Name", options_cols, 
                                               index=find_index("Customer Name", options_cols, is_optional=True))

            with col_map2:
                col_lon = st.selectbox("Kinh độ (Longitude) (*)", all_cols, 
                                       index=find_index("Longitude (*)", all_cols) or find_index("longitude", all_cols) or 2)
                col_vol_select = st.selectbox("VolEC/month", options_cols, 
                                              index=find_index("VolEC/month", options_cols, is_optional=True))
                col_addr_select = st.selectbox("Address", options_cols, 
                                               index=find_index("Address", options_cols, is_optional=True))

            st.caption("(*) là các cột bắt buộc.")

            st.session_state.col_mapping = {
                "customer_code": col_customer_code,
                "lat": col_lat,
                "lon": col_lon,
                "customer_name": None if col_name_select == "[Bỏ qua]" else col_name_select,
                "address": None if col_addr_select == "[Bỏ qua]" else col_addr_select,
                "vol_ec": None if col_vol_select == "[Bỏ qua]" else col_vol_select
            }
            
            st.subheader("3. Kiểm tra dữ liệu")
            total_customers = len(df)
            st.metric("Tổng số khách hàng (dòng)", total_customers)
            
            required_cols_check = [col_customer_code, col_lat, col_lon]
            duplicates = df.duplicated(subset=required_cols_check).sum()
            
            if duplicates > 0:
                st.warning(f"Tìm thấy {duplicates} dòng bị trùng (duplicate) theo 3 cột bắt buộc.")
            else:
                st.success("File không có dữ liệu trùng (theo 3 cột bắt buộc).")
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
        
        suggested_min = int(avg_customers * 0.9)
        suggested_max = int(avg_customers * 1.1)
        min_customers = st.number_input(
            "Số KH tối thiểu trên tuyến", min_value=0, value=suggested_min, step=1,
            help=f"Gợi ý: Số KH tối thiểu nên từ {suggested_min} trở lên."
        )
        st.caption(f"Gợi ý dựa trên mức trung bình: {suggested_min} (dưới 10%)")
        max_customers = st.number_input(
            "Số KH tối đa trên tuyến", min_value=0, value=suggested_max, step=1,
            help=f"Gợi ý: Số KH tối đa nên từ {suggested_max} trở xuống."
        )
        st.caption(f"Gợi ý dựa trên mức trung bình: {suggested_max} (trên 10%)")
        
        run_button = st.button("Bắt đầu phân tuyến", type="primary", use_container_width=True)
        
        if run_button:
            if not mapping.get("customer_code") or not mapping.get("lat") or not mapping.get("lon"):
                st.error("Lỗi: Vui lòng chọn đủ 3 cột bắt buộc (*).")
            elif min_customers > max_customers:
                st.error("Lỗi: Số KH tối thiểu không thể lớn hơn số KH tối đa.")
            else:
                with st.spinner("Đang xử lý..."):
                    try:
                        df_result, report_df = run_territory_planning(
                            df=st.session_state.df, lat_col=mapping['lat'], lon_col=mapping['lon'],
                            n_clusters=n_routes, min_size=min_customers, max_size=max_customers,
                            n_init=60
                        )
                        if df_result is not None:
                            st.session_state.df_map = df_result.copy()
                            st.session_state.df_edited = df_result.copy()
                            st.session_state.original_report_df = report_df.copy()
                            st.success("Phân tuyến thành công!")
                            
                            if st.session_state.sidebar_state == 'expanded':
                                st.session_state.sidebar_state = 'collapsed'
                                st.rerun() 
                                
                    except Exception as e:
                        st.error(f"Lỗi không xác định: {e}")
                        st.exception(e)

# --- KHU VỰC HIỂN THỊ KẾT QUẢ ---
if st.session_state.df_edited is not None:
    # --- THAY ĐỔI: Tách df_map (ổn định) và df_edited (nháp) ---
    df_map = st.session_state.df_map
    # df_edited sẽ được gọi bên trong data_editor
    original_report_df = st.session_state.original_report_df
    mapping = st.session_state.col_mapping

    colors_list = ["#FF0000", "#0000FF", "#00FF00", "#FFFF00", "#FF00FF", "#00FFFF", "#800000", "#008000", "#000080", "#FFA500"]
    color_map = {id: colors_list[(id - 1) % len(df_map['territory_id'].unique())] for id in df_map['territory_id'].unique()}
    df_map['color'] = df_map['territory_id'].map(color_map)
    
    all_tuyen_map = [int(x) for x in sorted(df_map['territory_id'].unique())]

    # --- TÓM LƯỢC NGANG (FULL-WIDTH) ---
    st.header("BẢN ĐỒ PHÂN TUYẾN")
    
    # --- THAY ĐỔI: SỬA LỖI LAG & FIX ĐỘ RỘNG CỘT ---
    # Bảng tóm lược này CHỈ ĐỌC TỪ df_map (bản ổn định)
    current_counts_series = df_map['territory_id'].value_counts().sort_index()
    current_report_df = pd.DataFrame({
        "Tuyến (RouteID)": current_counts_series.index,
        "Số lượng KH": current_counts_series.values
    })
    original_counts = original_report_df.set_index('Tuyến (RouteID)')['Số lượng KH']
    
    def get_arrow(row):
        route_id = row['Tuyến (RouteID)']
        current_count = row['Số lượng KH']
        original_count = original_counts.get(route_id, 0) 
        if current_count > original_count: return " 🔼" 
        elif current_count < original_count: return " 🔽"
        else: return ""
    current_report_df['arrow'] = current_report_df.apply(get_arrow, axis=1)

    html_items = []
    for _, row in current_report_df.iterrows():
        route_id = row['Tuyến (RouteID)']
        kh_count = row['Số lượng KH']
        arrow = row['arrow']
        item_html = f'<span style="display: inline-block; padding: 5px 15px; text-align: center; border-right: 1px solid #eee; line-height: 1.4;"><b>Tuyến {route_id}</b><br>{kh_count} KH{arrow}</span>'
        html_items.append(item_html)
    
    # FIX LỖI VISUAL: Thêm 'display: inline-block' để div tự co lại
    st.markdown(
        f"""
        <div style="overflow-x: auto; white-space: nowrap; display: inline-block; max-width: 100%; border: 1px solid #eee; border-radius: 5px; background-color: #f9f9f9;">
            {''.join(html_items)}
        </div>
        """,
        unsafe_allow_html=True
    )
    # --- KẾT THÚC TÓM LƯỢC NGANG ---
    
    st.divider()

    # --- CHIA CỘT 2:1 ---
    col_map_display, col_editor_display = st.columns([2, 1]) 

    with col_map_display:
        update_button_clicked = st.button("Cập nhật") # Đặt nút ở đây

        legend_items = []
        for tuyen_id in all_tuyen_map:
            color = color_map.get(tuyen_id)
            if color:
                legend_items.append(
                    f'<span style="background-color: {color}; width: 12px; height: 12px; display: inline-block; margin-right: 5px; border: 1px solid #000;"></span> Tuyến {tuyen_id}'
                )
        st.markdown("<b></b>&nbsp;&nbsp;&nbsp;" + "&nbsp;&nbsp;&nbsp;".join(legend_items), unsafe_allow_html=True)
        
        m, map_center = generate_folium_map(df_map, mapping) # Vẽ bản đồ từ df_map
        
        if m:
            st_folium(
                m, center=map_center, zoom=11, 
                use_container_width=True, height=600,
                returned_objects=[]
            )

    with col_editor_display:
        with st.expander("Chỉnh sửa thủ công", expanded=True):
            st.warning("Sau khi sửa, hãy nhấn nút 'Cập nhật' ở bên trái.")
            
            # Đọc df_edited từ state CHỈ MỘT LẦN
            df_editor_instance = st.session_state.df_edited
            
            all_cols_in_df = df_editor_instance.columns.tolist()
            col_code_name = mapping['customer_code']
            col_name_name = mapping.get('customer_name')
            col_addr_name = mapping.get('address')
            col_vol_name = mapping.get('vol_ec')
            col_tuyen_name = 'territory_id'
            
            cols_to_show = [col_code_name]
            if col_name_name: cols_to_show.append(col_name_name)
            if col_addr_name: cols_to_show.append(col_addr_name)
            if col_vol_name: cols_to_show.append(col_vol_name)
            cols_to_show.append(col_tuyen_name)
            
            column_config = {}
            for col in all_cols_in_df:
                if col not in cols_to_show:
                    column_config[col] = None
            
            all_tuyen_options = [int(x) for x in sorted(df_editor_instance['territory_id'].unique())]
            
            column_config[col_code_name] = st.column_config.TextColumn(f"Customer Code", disabled=True)
            if col_name_name:
                column_config[col_name_name] = st.column_config.TextColumn(f"Customer Name", disabled=True)
            if col_addr_name:
                column_config[col_addr_name] = st.column_config.TextColumn(f"Address", disabled=True)
            if col_vol_name:
                 column_config[col_vol_name] = st.column_config.TextColumn(f"VolEC/month", disabled=True)
            column_config[col_tuyen_name] = st.column_config.SelectboxColumn("Tuyến", options=all_tuyen_options, required=True)

            # --- THAY ĐỔI: BỎ CẬP NHẬT STATE TRỰC TIẾP ---
            edited_df_output = st.data_editor(
                df_editor_instance, 
                column_config=column_config,
                use_container_width=True, 
                num_rows="dynamic", 
                key="data_editor",
                height=600 
            )
            # Dòng st.session_state.df_edited = edited_df_output ĐÃ BỊ XÓA (để chống lag)

    # --- THAY ĐỔI: XỬ LÝ NÚT BẤM SAU KHI CÁC BIẾN ĐÃ TỒN TẠI ---
    if update_button_clicked:
        st.session_state.df_edited = edited_df_output.copy() # Lưu thay đổi vào state "nháp"
        st.session_state.df_map = edited_df_output.copy() # Cập nhật state "ổn định"
        generate_folium_map.clear() # Xóa cache bản đồ
        st.rerun() # Tải lại toàn bộ trang

    st.divider() 

    # --- NÚT DOWNLOAD (CĂN GIỮA) ---
    col1_dl, col2_dl, col3_dl = st.columns([1, 1, 1]) 

    with col2_dl: 
        output_buffer = io.BytesIO()
        with pd.ExcelWriter(output_buffer, engine='openpyxl') as writer:
            # Luôn tải về bản dữ liệu đã chỉnh sửa mới nhất
            st.session_state.df_edited.to_excel(writer, index=False, sheet_name='Territory_Output')
        
        st.download_button(
            label="Tải về file Excel kết quả",
            data=output_buffer.getvalue(),
            file_name="territory_output_edited.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )