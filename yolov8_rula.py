import streamlit as st
import cv2
import numpy as np
import pandas as pd
import os
import zipfile
import io
import tempfile
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

# ==================== 【正統 RULA 查表矩陣】 ====================
TABLE_A = {
    1: { 1: {1: {1:1, 2:2}, 2: {1:2, 2:2}, 3: {1:2, 2:3}, 4: {1:3, 2:3}},
         2: {1: {1:2, 2:2}, 2: {1:2, 2:3}, 3: {1:3, 2:3}, 4: {1:3, 2:4}},
         3: {1: {1:2, 2:3}, 2: {1:3, 2:3}, 3: {1:3, 2:4}, 4: {1:4, 2:4}} },
    2: { 1: {1: {1:1, 2:2}, 2: {1:2, 2:2}, 3: {1:3, 2:3}, 4: {1:3, 2:4}},
         2: {1: {1:2, 2:3}, 2: {1:3, 2:3}, 3: {1:3, 2:4}, 4: {1:4, 2:4}},
         3: {1: {1:3, 2:4}, 2: {1:3, 2:4}, 3: {1:4, 2:4}, 4: {1:4, 2:5}} },
    3: { 1: {1: {1:2, 2:3}, 2: {1:3, 2:3}, 3: {1:3, 2:4}, 4: {1:4, 2:4}},
         2: {1: {1:3, 2:3}, 2: {1:3, 2:4}, 3: {1:4, 2:4}, 4: {1:4, 2:5}},
         3: {1: {1:3, 2:4}, 2: {1:4, 2:4}, 3: {1:4, 2:5}, 4: {1:5, 2:5}} },
    4: { 1: {1: {1:3, 2:4}, 2: {1:4, 2:4}, 3: {1:4, 2:5}, 4: {1:5, 2:5}},
         2: {1: {1:4, 2:4}, 2: {1:4, 2:5}, 3: {1:4, 2:5}, 4: {1:5, 2:6}},
         3: {1: {1:4, 2:5}, 2: {1:5, 2:5}, 3: {1:5, 2:6}, 4: {1:6, 2:7}} },
    5: { 1: {1: {1:4, 2:4}, 2: {1:4, 2:5}, 3: {1:4, 2:5}, 4: {1:5, 2:6}},
         2: {1: {1:4, 2:5}, 2: {1:5, 2:5}, 3: {1:5, 2:6}, 4: {1:6, 2:7}},
         3: {1: {1:5, 2:6}, 2: {1:6, 2:6}, 3: {1:6, 2:7}, 4: {1:7, 2:7}} },
    6: { 1: {1: {1:5, 2:5}, 2: {1:5, 2:6}, 3: {1:5, 2:6}, 4: {1:6, 2:7}},
         2: {1: {1:5, 2:6}, 2: {1:6, 2:6}, 3: {1:6, 2:7}, 4: {1:7, 2:7}},
         3: {1: {1:6, 2:7}, 2: {1:7, 2:7}, 3: {1:7, 2:8}, 4: {1:8, 2:8}} }
}

TABLE_B = {
    1: {1: {1:1, 2:3}, 2: {1:2, 2:3}, 3: {1:3, 2:4}, 4: {1:5, 2:5}, 5: {1:6, 2:6}, 6: {1:7, 2:7}},
    2: {1: {1:2, 2:3}, 2: {1:3, 2:3}, 3: {1:4, 2:4}, 4: {1:5, 2:5}, 5: {1:6, 2:7}, 6: {1:7, 2:8}},
    3: {1: {1:3, 2:3}, 2: {1:4, 2:4}, 3: {1:4, 2:5}, 4: {1:6, 2:6}, 5: {1:7, 2:7}, 6: {1:7, 2:8}},
    4: {1: {1:5, 2:5}, 2: {1:5, 2:6}, 3: {1:6, 2:7}, 4: {1:7, 2:7}, 5: {1:8, 2:8}, 6: {1:8, 2:8}},
    5: {1: {1:6, 2:6}, 2: {1:6, 2:7}, 3: {1:7, 2:8}, 4: {1:8, 2:8}, 5: {1:8, 2:8}, 6: {1:8, 2:8}},
    6: {1: {1:7, 2:7}, 2: {1:7, 2:8}, 3: {1:8, 2:8}, 4: {1:8, 2:8}, 5: {1:8, 2:8}, 6: {1:8, 2:8}}
}

TABLE_C = [
    [1, 2, 3, 3, 4, 5, 5], [2, 2, 3, 4, 4, 5, 5], [3, 3, 3, 4, 4, 5, 6], [3, 3, 3, 4, 5, 6, 6],
    [4, 4, 4, 5, 6, 7, 7], [4, 4, 5, 6, 6, 7, 7], [5, 5, 6, 6, 7, 7, 7], [5, 5, 6, 7, 7, 7, 7]
]

# ==================== 【純 2D 向量運算與字型模組】 ====================
def calculate_2d_angle(a, b, c):
    """計算 b 為頂點，ba 與 bc 在 2D 矢狀面的夾角"""
    ba = np.array([a[0] - b[0], a[1] - b[1]])
    bc = np.array([c[0] - b[0], c[1] - b[1]])
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba == 0 or norm_bc == 0:
        return 0.0
    cosine = np.clip(np.dot(ba, bc) / (norm_ba * norm_bc), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))

def put_chinese_text(img, text, position, color, size=16):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font = None
    font_candidates = [
        "font.ttf",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "msjh.ttc",
        "simhei.ttf"
    ]
    for p in font_candidates:
        if os.path.exists(p):
            try:
                font = ImageFont.truetype(p, size)
                break
            except Exception:
                continue
    if font is None:
        font = ImageFont.load_default()
    draw.text(position, text, fill=(color[2], color[1], color[0]), font=font)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# ==================== 【RULA 分級評分規則】 ====================
def get_upper_arm_score(angle, is_abducted=False):
    if angle <= 20: base = 1
    elif angle <= 45: base = 2
    elif angle <= 90: base = 3
    else: base = 4
    if is_abducted: base += 1
    return min(max(base, 1), 6)

def get_lower_arm_score(angle, is_across_midline=False):
    base = 1 if 60 <= angle <= 100 else 2
    if is_across_midline: base += 1
    return min(max(base, 1), 3)

def get_neck_score(angle, is_side_bending=False):
    if angle <= 10: base = 1
    elif angle <= 20: base = 2
    else: base = 3
    if is_side_bending: base += 1
    return min(max(base, 1), 6)

def get_trunk_score(angle, is_side_bending=False):
    if angle <= 5: base = 1
    elif angle <= 20: base = 2
    elif angle <= 60: base = 3
    else: base = 4
    if is_side_bending: base += 1
    return min(max(base, 1), 6)

def compute_rula_side_score(ua, la, nk, tk, m_val, f_val):
    w, wt, lg = 1, 1, 1
    score_a = TABLE_A[ua][la][w][wt]
    score_b = TABLE_B[nk][tk][lg]
    score_c = score_a + m_val + f_val
    score_d = score_b + m_val + f_val
    grand_score = TABLE_C[min(score_c, 8) - 1][min(score_d, 7) - 1]
    al_num = 1 if grand_score <= 2 else (2 if grand_score <= 4 else (3 if grand_score <= 6 else 4))
    return score_a, score_b, score_c, score_d, grand_score, f"AL{al_num}"

# ==================== 【人體解剖約束與骨架校正】 ====================
def correct_hip_position(sh, hip, knee, conf_hip, conf_knee):
    """
    解剖學防禦：當穿著 Oversized 衣服彎腰時，髖關節容易被抓在空氣中。
    若膝蓋清晰可見，強制利用大腿直立方向修正髖骨位置。
    """
    if conf_knee > 0.5 and conf_hip < 0.65:
        # 以膝蓋上方合理的人體軀幹長度估計髖骨
        est_hip = [knee[0], knee[1] - abs(knee[1] - sh[1]) * 0.45]
        return est_hip
    return hip

# ==================== 【側視角高精度分析核心】 ====================
def analyze_side_frame_yolo(frame, model, target_side, front_mods, m_val, f_val):
    h, w = frame.shape[:2]
    # 使用 high-res 推論以提升細節
    results = model.predict(frame, imgsz=960, conf=0.3, verbose=False)
    
    if not results or len(results[0].keypoints) == 0:
        return False, None, frame, 0.0

    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return False, None, frame, 0.0
    
    # 鎖定畫面中最大的受測者
    areas = [(b.xyxy[0][2] - b.xyxy[0][0]) * (b.xyxy[0][3] - b.xyxy[0][1]) for b in boxes]
    best_person_idx = int(np.argmax(areas))

    kpts = results[0].keypoints.data[best_person_idx].cpu().numpy()
    
    # COCO Keypoint 定位
    if target_side == "右手側":
        ear_idx, sh_idx, el_idx, wr_idx, hip_idx, knee_idx = 4, 6, 8, 10, 12, 14
    else:
        ear_idx, sh_idx, el_idx, wr_idx, hip_idx, knee_idx = 3, 5, 7, 9, 11, 13

    ear = [float(kpts[ear_idx][0]), float(kpts[ear_idx][1])]
    sh  = [float(kpts[sh_idx][0]),  float(kpts[sh_idx][1])]
    el  = [float(kpts[el_idx][0]),  float(kpts[el_idx][1])]
    wr  = [float(kpts[wr_idx][0]),  float(kpts[wr_idx][1])]
    hip = [float(kpts[hip_idx][0]), float(kpts[hip_idx][1])]
    knee= [float(kpts[knee_idx][0]),float(kpts[knee_idx][1])]

    conf_list = [kpts[sh_idx][2], kpts[el_idx][2], kpts[wr_idx][2], kpts[hip_idx][2]]
    overall_conf = float(np.mean(conf_list) * 100)

    # 執行解剖學骨盆校正
    hip = correct_hip_position(sh, hip, knee, kpts[hip_idx][2], kpts[knee_idx][2])

    # 1. 前臂屈曲 (Lower Arm)
    ang_lower_arm = abs(180.0 - calculate_2d_angle(sh, el, wr))

    # 2. 上臂屈曲 (Upper Arm): 相對於軀幹軸線 (肩 -> 髖)
    trunk_down = [sh[0] + (hip[0] - sh[0]), sh[1] + (hip[1] - sh[1])]
    ang_upper_arm = calculate_2d_angle(trunk_down, sh, el)

    # 3. 軀幹傾角 (Trunk): 髖-肩連線相對於垂直向上
    vert_up = [hip[0], hip[1] - 100.0]
    ang_trunk = calculate_2d_angle(vert_up, hip, sh)

    # 4. 頸部屈角 (Neck): 軀幹軸線相對於耳-肩連線
    ang_neck = abs(180.0 - calculate_2d_angle(hip, sh, ear))

    # 人因評分
    s_ua = get_upper_arm_score(ang_upper_arm, is_abducted=front_mods.get("is_abducted", False))
    s_la = get_lower_arm_score(ang_lower_arm, is_across_midline=front_mods.get("is_across_midline", False))
    s_tk = get_trunk_score(ang_trunk, is_side_bending=front_mods.get("is_side_bending", False))
    s_nk = get_neck_score(ang_neck)

    sc_a, sc_b, sc_c, sc_d, grand, al = compute_rula_side_score(s_ua, s_la, s_nk, s_tk, m_val, f_val)

    # 繪製穩定骨架線條
    annotated = frame.copy()
    pts = [sh, el, wr, hip, knee]
    for p in pts:
        cv2.circle(annotated, (int(p[0]), int(p[1])), 6, (0, 255, 0), -1)
    cv2.line(annotated, (int(sh[0]), int(sh[1])), (int(el[0]), int(el[1])), (255, 0, 0), 3)
    cv2.line(annotated, (int(el[0]), int(el[1])), (int(wr[0]), int(wr[1])), (255, 100, 0), 3)
    cv2.line(annotated, (int(sh[0]), int(sh[1])), (int(hip[0]), int(hip[1])), (0, 0, 255), 3)
    cv2.line(annotated, (int(hip[0]), int(hip[1])), (int(knee[0]), int(knee[1])), (0, 255, 255), 3)

    side_prefix = "右手" if target_side == "右手側" else "左手"
    data = {
        "頸部角度": round(ang_neck, 1),
        "軀幹角度": round(ang_trunk, 1),
        f"{side_prefix}上臂角度": round(ang_upper_arm, 1),
        f"{side_prefix}前臂角度": round(ang_lower_arm, 1),
        f"{side_prefix}上臂外展補正": "是 (+1分)" if front_mods.get("is_abducted") else "否",
        f"{side_prefix}橫越中線補正": "是 (+1分)" if front_mods.get("is_across_midline") else "否",
        f"{side_prefix} Score A": sc_a,
        f"{side_prefix} Score C": sc_c,
        f"{side_prefix} Grand Score": grand,
        f"{side_prefix} AL": al,
        "最危害側": target_side,
        "最終最高分": grand,
        "最終 AL": al,
        "Overall_Conf": overall_conf
    }
    return True, data, annotated, overall_conf

# ==================== 【正面視角補正分析】 ====================
def analyze_front_frame_yolo(frame, model, target_side):
    results = model.predict(frame, imgsz=640, conf=0.35, verbose=False)
    if not results or len(results[0].keypoints) == 0:
        return False, {}, frame
    
    kpts = results[0].keypoints.data[0].cpu().numpy()
    sh_idx = 6 if target_side == "右手側" else 5
    el_idx = 8 if target_side == "右手側" else 7
    wr_idx = 10 if target_side == "右手側" else 9

    midline_x = (kpts[5][0] + kpts[6][0] + kpts[11][0] + kpts[12][0]) / 4.0
    sh_width  = abs(kpts[5][0] - kpts[6][0])

    el_out_dist = (kpts[el_idx][0] - kpts[sh_idx][0]) if target_side == "右手側" else (kpts[sh_idx][0] - kpts[el_idx][0])
    is_abducted = el_out_dist > (sh_width * 0.35)
    is_across_midline = (kpts[wr_idx][0] < midline_x) if target_side == "右手側" else (kpts[wr_idx][0] > midline_x)
    is_side_bending = abs(kpts[5][1] - kpts[6][1]) > (sh_width * 0.15)

    return True, {
        "is_abducted": is_abducted,
        "is_across_midline": is_across_midline,
        "is_side_bending": is_side_bending
    }, frame

# ==================== 【Streamlit 網頁應用介面】 ====================
st.set_page_config(page_title="RULA 姿態評估系統 (YOLOv8 高精版)", layout="wide")
st.title("🎯 RULA AI 姿勢危害分析系統")
st.markdown("針對彎腰、寬鬆衣物遮擋與肢體重疊進行**解剖學骨盆約束校正**與**純 2D 矢狀面投影量測**。")

target_side = st.radio("1. 請選擇主要評估側:", ["右手側", "左手側"], horizontal=True)

st.header("上傳作業影像")
c1, c2, c3 = st.columns(3)
with c1:
    v_front = st.file_uploader("正面鏡頭 (外展/橫越中線補正)", type=['mp4', 'mov', 'avi'])
with c2:
    v_right = st.file_uploader("右側視角 (量測右手動作)", type=['mp4', 'mov', 'avi'])
with c3:
    v_left  = st.file_uploader("左側視角 (量測左手動作)", type=['mp4', 'mov', 'avi'])

st.header("分析取樣與作業參數")
cp1, cp2, cp3 = st.columns(3)
with cp1:
    num_samples = st.slider("等距抽樣張數", min_value=5, max_value=120, value=30)
with cp2:
    combo_muscle = st.selectbox("肌肉使用 (Muscle):", ["0: 靜態少於1分鐘 / 偶發", "1: 姿勢維持>1分鐘 / 高重複作業"])
with cp3:
    combo_force = st.selectbox("荷重施力 (Force):", ["0: < 2kg (間歇)", "1: 2-10kg (間歇)", "2: 2-10kg (靜態/重複)", "3: > 10kg"])

if st.button("🚀 啟動高精度分析", type="primary", use_container_width=True):
    selected_side_vid = v_right if target_side == "右手側" else v_left
    
    if not v_front:
        st.error("❌ 請上傳【正面鏡頭影片】以供水平面補正！")
    elif not selected_side_vid:
        st.error(f"❌ 請上傳【{target_side}側面影片】進行矢狀面量測！")
    else:
        with st.spinner("載入 YOLOv8-pose 模型中..."):
            model = YOLO("yolov8s-pose.pt")

        with st.spinner("正在進行多視角高精度姿態分析與解剖學約束解算..."):
            tf_f = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tf_f.write(v_front.read())
            cap_f = cv2.VideoCapture(tf_f.name)

            tf_s = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tf_s.write(selected_side_vid.read())
            cap_s = cv2.VideoCapture(tf_s.name)

            fps = cap_s.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames = min(int(cap_f.get(cv2.CAP_PROP_FRAME_COUNT)), int(cap_s.get(cv2.CAP_PROP_FRAME_COUNT)))
            sample_indices = np.linspace(0, total_frames - 1, num_samples, dtype=int).tolist()

            raw_front, raw_side = [], []
            for idx in sample_indices:
                t_sec = round(idx / fps, 2)
                cap_f.set(cv2.CAP_PROP_POS_FRAMES, idx)
                cap_s.set(cv2.CAP_PROP_POS_FRAMES, idx)
                sf, ff = cap_f.read()
                ss, fs = cap_s.read()
                if sf and ss:
                    raw_front.append((t_sec, ff))
                    raw_side.append((t_sec, fs))
            
            cap_f.release()
            cap_s.release()

            m_val = int(combo_muscle.split(":")[0])
            f_val = int(combo_force.split(":")[0])

            records, zip_images, display_images = [], [], []
            pbar = st.progress(0)
            st_text = st.empty()

            for i in range(len(raw_side)):
                sec, f_frame = raw_front[i]
                _, s_frame = raw_side[i]

                _, front_mods, _ = analyze_front_frame_yolo(f_frame, model, target_side)
                succ, data, anno_frame, conf = analyze_side_frame_yolo(s_frame, model, target_side, front_mods, m_val, f_val)

                if succ and data is not None:
                    row = {
                        "樣本編號": i + 1,
                        "時間(秒)": sec,
                        "主要評估側": target_side,
                        "信賴度(%)": round(conf, 1)
                    }
                    row.update(data)
                    records.append(row)

                    anno_frame = put_chinese_text(anno_frame, f"時間: {sec}s | 目標: {target_side}", (15, 15), (0, 255, 255), 18)
                    anno_frame = put_chinese_text(anno_frame, f"Grand Score: {data['最終最高分']} 分 ({data['最終 AL']})", (15, 45), (0, 0, 255) if data['最終最高分'] > 4 else (0, 255, 0), 16)
                    
                    is_ok, buf = cv2.imencode(".jpg", anno_frame)
                    if is_ok:
                        img_bytes = buf.tobytes()
                        zip_images.append((f"Sample_{i+1:03d}_{sec}s.jpg", img_bytes))
                        display_images.append((f"樣本 #{i+1} ({sec}s)", cv2.cvtColor(anno_frame, cv2.COLOR_BGR2RGB)))

                pbar.progress((i + 1) / len(raw_side))
                st_text.text(f"姿態解析與約束校正中: {i+1} / {len(raw_side)}")

            st.success("🎉 分析完成！")

            # 網頁即時指標
            if records:
                df = pd.DataFrame(records)
                avg_score = df["最終最高分"].mean()
                max_score = df["最終最高分"].max()
                high_risk_ratio = (df["最終最高分"] >= 5).mean() * 100

                m1, m2, m3 = st.columns(3)
                m1.metric("平均 RULA 總分", f"{avg_score:.2f} 分")
                m2.metric("最高風險峰值", f"{max_score} 分")
                m3.metric("高危害佔比 (AL3/AL4)", f"{high_risk_ratio:.1f} %")

                st.subheader("📊 詳細稽核報表")
                st.dataframe(df, use_container_width=True)

                st.subheader("🖼️ 標註抽樣影像預覽")
                cols = st.columns(3)
                for idx, (cap_title, rgb_img) in enumerate(display_images):
                    with cols[idx % 3]:
                        st.image(rgb_img, caption=cap_title, use_container_width=True)

            # 打包 ZIP
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w') as zf:
                if records:
                    zf.writestr("RULA多視角分析報表.csv", df.to_csv(index=False).encode('utf-8-sig'))
                for fname, fbytes in zip_images:
                    zf.writestr(f"標註照片/{fname}", fbytes)

            st.download_button(
                label=f"📦 下載 {target_side} RULA 完整稽核包 (ZIP)",
                data=zip_buffer.getvalue(),
                file_name=f"RULA_{target_side}_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                mime="application/zip"
            )
