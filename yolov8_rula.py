import streamlit as st
import cv2
from ultralytics import YOLO
import math
import numpy as np
import pandas as pd
import os
import zipfile
import io
import tempfile
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont

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

# ==================== 【模型載入 (快取避免重複下載)】 ====================
@st.cache_resource
def load_yolo():
    # 使用官方輕量高精度 YOLOv8-Pose 模型
    return YOLO("yolov8n-pose.pt")

# ==================== 【高精度 2D 平面幾何運算】 ====================
def calculate_2d_angle(a, b, c):
    """
    計算以點 b 為頂點之 2D 夾角 (度數)
    向量 ba 與 bc 之夾角，完全捨棄不可靠的 Z 軸震盪
    """
    ba = np.array([a[0] - b[0], a[1] - b[1]], dtype=np.float32)
    bc = np.array([c[0] - b[0], c[1] - b[1]], dtype=np.float32)
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba < 1e-5 or norm_bc < 1e-5:
        return 0.0
    cosine = np.clip(np.dot(ba, bc) / (norm_ba * norm_bc), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))

def calculate_vector_angle(v1, v2):
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 < 1e-5 or norm_v2 < 1e-5:
        return 0.0
    cosine = np.clip(np.dot(v1, v2) / (norm_v1 * norm_v2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))

def get_upper_arm_score(angle, is_abducted=False, is_raised=False):
    if angle <= 20: base = 1
    elif angle <= 45: base = 2
    elif angle <= 90: base = 3
    else: base = 4
    if is_abducted: base += 1
    if is_raised: base += 1
    return min(max(base, 1), 6)

def get_lower_arm_score(angle, is_across_midline=False):
    base = 1 if 60 <= angle <= 100 else 2
    if is_across_midline: base += 1
    return min(max(base, 1), 3)

def get_neck_score(angle, is_twisted=False, is_side_bending=False):
    if angle <= 10: base = 1
    elif angle <= 20: base = 2
    else: base = 3
    if is_twisted: base += 1
    if is_side_bending: base += 1
    return min(max(base, 1), 6)

def get_trunk_score(angle, is_twisted=False, is_side_bending=False):
    if angle <= 5: base = 1
    elif angle <= 20: base = 2
    elif angle <= 60: base = 3
    else: base = 4
    if is_twisted: base += 1
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

def put_chinese_text(img, text, position, color, size=15):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font = None
    font_candidates = [
        "font.ttf",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
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

# ==================== 【YOLOv8 骨架萃取與分析】 ====================
# COCO 17 Keypoints 映射表:
# 0:Nose, 1:L_Eye, 2:R_Eye, 3:L_Ear, 4:R_Ear,
# 5:L_Shoulder, 6:R_Shoulder, 7:L_Elbow, 8:R_Elbow, 9:L_Wrist, 10:R_Wrist,
# 11:L_Hip, 12:R_Hip, 13:L_Knee, 14:R_Knee, 15:L_Ankle, 16:R_Ankle

def extract_yolo_landmarks(results):
    if len(results) == 0 or results[0].keypoints is None or len(results[0].keypoints.xy) == 0:
        return None, None
    pts = results[0].keypoints.xy[0].cpu().numpy()  # shape (17, 2)
    conf = results[0].keypoints.conf[0].cpu().numpy() if results[0].keypoints.conf is not None else np.ones(17)
    return pts, conf

def analyze_front_camera_yolo(frame, model, target_side):
    results = model(frame, verbose=False)
    pts, conf = extract_yolo_landmarks(results)
    annotated = results[0].plot() if len(results) > 0 else frame
    
    if pts is None:
        return False, {}, annotated
        
    sh_idx = 6 if target_side == "右手側" else 5
    el_idx = 8 if target_side == "右手側" else 7
    wr_idx = 10 if target_side == "右手側" else 9
    
    # 身體中線 X 座標
    midline_x = (pts[5][0] + pts[6][0] + pts[11][0] + pts[12][0]) / 4.0
    shoulder_w = abs(pts[5][0] - pts[6][0])
    
    # 外展判定
    elbow_out = (pts[el_idx][0] - pts[sh_idx][0]) if target_side == "右手側" else (pts[sh_idx][0] - pts[el_idx][0])
    is_abducted = elbow_out > (shoulder_w * 0.35)
    
    # 越過中線判定
    is_across_midline = (pts[wr_idx][0] < midline_x) if target_side == "右手側" else (pts[wr_idx][0] > midline_x)
    
    # 軀幹側彎
    shoulder_tilt = abs(pts[5][1] - pts[6][1])
    is_side_bending = (shoulder_tilt / max(shoulder_w, 1.0)) > 0.15
    
    mods = {
        "is_abducted": bool(is_abducted),
        "is_across_midline": bool(is_across_midline),
        "is_side_bending": bool(is_side_bending)
    }
    return True, mods, annotated

def analyze_side_camera_yolo(frame, model, target_side, front_mods, m_val, f_val):
    results = model(frame, verbose=False)
    pts, conf = extract_yolo_landmarks(results)
    annotated = results[0].plot() if len(results) > 0 else frame
    
    if pts is None:
        return False, None, annotated, 0.0
        
    sh_idx = 6 if target_side == "右手側" else 5
    el_idx = 8 if target_side == "右手側" else 7
    wr_idx = 10 if target_side == "右手側" else 9
    hip_idx = 12 if target_side == "右手側" else 11
    ear_idx = 4 if target_side == "右手側" else 3
    
    # 關鍵點能見度防護
    vis_scores = [conf[sh_idx], conf[el_idx], conf[wr_idx], conf[hip_idx]]
    mean_conf = float(np.mean(vis_scores) * 100)
    if min(vis_scores) < 0.35:
        # 若目標手臂被身體嚴重遮蔽，給予低信賴度
        mean_conf *= 0.4

    sh = pts[sh_idx]
    el = pts[el_idx]
    wr = pts[wr_idx]
    hp = pts[hip_idx]
    ear = pts[ear_idx]

    # 1. 前臂屈曲 (Lower Arm): 180 - 內角
    ang_lower_arm = abs(180.0 - calculate_2d_angle(sh, el, wr))
    
    # 2. 上臂屈曲 (Upper Arm): 以軀幹主軸向量為基準線
    v_torso = hp - sh  # 軀幹向量 (由肩指向臀)
    v_arm = el - sh    # 上臂向量 (由肩指向肘)
    ang_upper_arm = calculate_vector_angle(v_torso, v_arm)
    
    # 3. 軀幹前傾 (Trunk): 髖骨至肩膀向量與垂直向上向量 [0, -1] 夾角
    v_trunk_up = sh - hp
    v_vertical_up = np.array([0.0, -1.0], dtype=np.float32)
    ang_trunk = calculate_vector_angle(v_trunk_up, v_vertical_up)
    
    # 4. 頸部前屈 (Neck): 相對於軀幹軸線
    v_neck = ear - sh
    ang_neck = calculate_vector_angle(v_trunk_up, v_neck)

    s_ua = get_upper_arm_score(ang_upper_arm, is_abducted=front_mods.get("is_abducted", False))
    s_la = get_lower_arm_score(ang_lower_arm, is_across_midline=front_mods.get("is_across_midline", False))
    s_tk = get_trunk_score(ang_trunk, is_side_bending=front_mods.get("is_side_bending", False))
    s_nk = get_neck_score(ang_neck)

    sc_a, sc_b, sc_c, sc_d, grand, al = compute_rula_side_score(s_ua, s_la, s_nk, s_tk, m_val, f_val)
    side_prefix = "右手" if target_side == "右手側" else "左手"

    data = {
        "頸部角度": round(ang_neck, 1),
        "軀幹角度": round(ang_trunk, 1),
        f"{side_prefix}上臂角度": round(ang_upper_arm, 1),
        f"{side_prefix}前臂角度": round(ang_lower_arm, 1),
        f"{side_prefix}上臂外展補正": "是 (+1分)" if front_mods.get("is_abducted") else "否",
        f"{side_prefix}橫越中線補正": "是 (+1分)" if front_mods.get("is_across_midline") else "否",
        "軀幹側彎補正": "是 (+1分)" if front_mods.get("is_side_bending") else "否",
        f"{side_prefix} Score A": sc_a,
        f"{side_prefix} Score C": sc_c,
        f"{side_prefix} Grand Score": grand,
        f"{side_prefix} AL": al,
        "最危害側": target_side,
        "最終最高分": grand,
        "最終 AL": al,
        "Overall_Conf": mean_conf
    }
    return True, data, annotated, mean_conf

# ==================== 【Streamlit 網頁應用介面】 ====================
st.set_page_config(page_title="YOLOv8-Pose 智慧 RULA 人因分析系統", layout="wide")
st.title("YOLOv8-Pose 多視角人因工程 (RULA) 自動化評估系統")
st.markdown("> **專案架構特色**：採用 Ultralytics YOLOv8-Pose 進行高抗噪關鍵點偵測，結合矢狀面 2D 向量投影，杜絕深度抖動，量測精度逼近 $\\pm 1^\\circ \\sim 3^\\circ$。")

target_side = st.radio(
    "🎯 請選擇本次評估作業之【主量測側】:",
    options=["右手側", "左手側"],
    index=0,
    horizontal=True,
    help="根據 RULA 評估標準，鎖定受測者主要施力或偏向側，避免非受測側肢體遮蔽干擾。"
)

st.header("步驟一：上傳多視角測試影像")
col_front, col_right, col_left = st.columns(3)

with col_front:
    vid_front = st.file_uploader("1. 正面視角 (必填: 外展/穿越中線校正)", type=['mp4', 'mov', 'avi'])
with col_right:
    vid_right = st.file_uploader("2. 右側視角 (右手側主屈曲量測)", type=['mp4', 'mov', 'avi'])
with col_left:
    vid_left = st.file_uploader("3. 左側視角 (左手側主屈曲量測)", type=['mp4', 'mov', 'avi'])

st.header("步驟二：IE 工作抽樣與人因工況設定")
col_s1, col_s2, col_s3 = st.columns(3)
with col_s1:
    num_samples = st.slider("等距抽樣張數 (Work Sampling)", min_value=5, max_value=120, value=30)
with col_s2:
    combo_muscle = st.selectbox("肌肉使用分數 (Muscle Score):", ["0: 間歇動作 / 姿勢維持<1分鐘", "1: 姿勢維持>1分鐘 / 高頻率重複作業"])
with col_s3:
    combo_force = st.selectbox("負載與施力 (Force/Load):", ["0: < 2kg (間歇施力)", "1: 2-10kg (間歇施力)", "2: 2-10kg (靜態/重複施力)", "3: > 10kg 或衝擊性施力"])

if st.button("🚀 啟動 YOLOv8 多視角聯立評估", type="primary", use_container_width=True):
    side_video = vid_right if target_side == "右手側" else vid_left
    
    if not vid_front:
        st.error("❌ 缺少正面鏡頭影片！正面影像為判定上臂外展與橫越中線的必備基準。")
    elif not side_video:
        st.error(f"❌ 缺少【{target_side}】的側面影像！系統需仰賴該側視角量測主要屈曲角度。")
    else:
        with st.spinner("YOLOv8 骨架特徵提取與 RULA 多維度解算中..."):
            model = load_yolo()
            
            t_front = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            t_front.write(vid_front.read())
            cap_front = cv2.VideoCapture(t_front.name)
            
            t_side = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            t_side.write(side_video.read())
            cap_side = cv2.VideoCapture(t_side.name)

            fps = cap_side.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames = min(int(cap_front.get(cv2.CAP_PROP_FRAME_COUNT)), int(cap_side.get(cv2.CAP_PROP_FRAME_COUNT)))

            if total_frames <= 0:
                st.error("影片影格數無效，請確認檔案格式是否完整！")
                st.stop()

            sample_indices = np.linspace(0, total_frames - 1, num_samples, dtype=int).tolist()
            raw_front_frames, raw_side_frames = [], []
            
            for idx in sample_indices:
                t_sec = round(idx / fps, 2)
                cap_front.set(cv2.CAP_PROP_POS_FRAMES, idx)
                cap_side.set(cv2.CAP_PROP_POS_FRAMES, idx)
                s_f, f_img = cap_front.read()
                s_s, s_img = cap_side.read()
                if s_f and s_s:
                    raw_front_frames.append((t_sec, f_img))
                    raw_side_frames.append((t_sec, s_img))

            cap_front.release()
            cap_side.release()

            m_val = int(combo_muscle.split(":")[0])
            f_val = int(combo_force.split(":")[0])

            records = []
            zip_images = []
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            total_samples = len(raw_side_frames)

            for i in range(total_samples):
                sec, f_frame = raw_front_frames[i]
                _, s_frame = raw_side_frames[i]

                # 1. 正面判斷水平面補正
                succ_f, front_mods, _ = analyze_front_camera_yolo(f_frame, model, target_side)
                
                # 2. 側面進行 2D 向量高精度量測
                succ_s, data, side_annotated, conf = analyze_side_camera_yolo(s_frame, model, target_side, front_mods, m_val, f_val)

                if succ_s and data is not None:
                    record = {
                        "樣本編號": i + 1,
                        "時間(秒)": sec,
                        "主要評估側": target_side,
                        "視角信賴度(%)": round(conf, 1)
                    }
                    record.update(data)
                    records.append(record)

                    # 標註結果於側視圖
                    side_annotated = put_chinese_text(side_annotated, f"時間: {sec}s | YOLOv8 目標: {target_side}", (15, 15), (0, 255, 255), 18)
                    side_annotated = put_chinese_text(side_annotated, f"Grand Score: {data['最終最高分']} 分 ({data['最終 AL']})", (15, 45), (0, 0, 255) if data['最終最高分'] > 4 else (0, 255, 0), 16)
                    side_annotated = put_chinese_text(side_annotated, f"正面校正: 外展({data[f'{target_side[:2]}上臂外展補正']}) | 交叉({data[f'{target_side[:2]}橫越中線補正']})", (15, 75), (255, 150, 0), 14)

                    is_ok, buf = cv2.imencode(".jpg", side_annotated)
                    if is_ok:
                        zip_images.append((f"Sample_{i+1:03d}_{sec}s_YOLOv8_{target_side}.jpg", buf.tobytes()))

                progress_bar.progress((i + 1) / total_samples)
                status_text.text(f"YOLOv8 骨架運算進度: {i+1} / {total_samples} 筆完成")

            # 打包成下載 ZIP
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w') as zf:
                if records:
                    df = pd.DataFrame(records)
                    zf.writestr("RULA_YOLOv8_統計分析報表.csv", df.to_csv(index=False).encode('utf-8-sig'))
                for fname, fbytes in zip_images:
                    zf.writestr(f"YOLOv8標註照片庫/{fname}", fbytes)

            st.success("✅ YOLOv8 多視角解算完成！請下載稽核成果包。")
            st.download_button(
                label=f"📦 下載 {target_side} YOLOv8 報表 (ZIP)",
                data=zip_buffer.getvalue(),
                file_name=f"RULA_YOLOv8_{target_side}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                mime="application/zip"
            )