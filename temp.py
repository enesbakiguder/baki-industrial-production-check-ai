import os
import cv2
import numpy as np
from typing import Tuple, Dict, Any, List

try:
    from matching.viz import plot_matches
    from matching import get_matcher
except ImportError:
    print("[HATA] 'matching' modülü bulunamadı. Lütfen klasör yapısını kontrol edin.")

from ultralytics import YOLO

# 1) ÇALIŞMA DİZİNİ SABİTLEME
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)

# 2) GLOBAL MODELLER
DEVICE = "cpu"
MATCHER_NAME = "sift-lg"

print(f"[INFO] Matcher yükleniyor: {MATCHER_NAME} ({DEVICE})")
matcher = get_matcher(MATCHER_NAME, device=DEVICE)

print("[INFO] YOLO modeli yükleniyor...")
# Model dosyası kodla aynı klasörde olmalı
YOLO_MODEL = YOLO("yolov8n.pt") 

# 3) IŞIK / RENK NORMALİZASYONU
def preprocess_image_for_matching(path: str, use_cache: bool = True) -> str:
    """CLAHE ile gri normalize görüntü üretir ve _norm.png olarak kaydeder."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Görüntü bulunamadı: {path}")
    
    folder, fname = os.path.split(path)
    name, _ = os.path.splitext(fname)
    norm_path = os.path.join(folder, f"{name}_norm.png")
    
    if use_cache and os.path.isfile(norm_path):
        return norm_path
        
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Görüntü okunamadı: {path}")
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)
    
    cv2.imwrite(norm_path, gray_eq)
    return norm_path

# 4) YOLO + GRABCUT İLE ÜRÜN MASKESİ
def get_product_mask(reference_image_path: str, resize: int = 512) -> Tuple[np.ndarray, np.ndarray]:
    img_color = cv2.imread(reference_image_path)
    if img_color is None:
        raise ValueError(f"Ürün resmi okunamadı: {reference_image_path}")
        
    results = YOLO_MODEL(img_color)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    
    if boxes.shape[0] == 0:
        print("[WARN] YOLO ürün bulamadı - tüm görüntü ürün kabul edildi.")
        img_resized = cv2.resize(img_color, (resize, resize))
        mask = np.ones((img_resized.shape[0], img_resized.shape[1]), np.uint8) * 255
        return mask, img_resized
        
    # En büyük kutuyu seç
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    best_idx = int(np.argmax(areas))
    x1, y1, x2, y2 = boxes[best_idx].astype(int)
    
    h0, w0 = img_color.shape[:2]
    img_resized = cv2.resize(img_color, (resize, resize))
    sx = resize / w0
    sy = resize / h0
    
    rect = (int(x1 * sx), int(y1 * sy), int((x2 - x1) * sx), int((y2 - y1) * sy))
    
    mask = np.zeros((resize, resize), np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    
    cv2.grabCut(img_resized, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    product_mask = np.where((mask == 1) | (mask == 3), 255, 0).astype("uint8")
    
    return product_mask, img_resized

# 5) İKİ GÖRÜNTÜYÜ EŞLEŞTİRME
def match_two_images(path1: str, path2: str, resize: int = 512):
    norm1 = preprocess_image_for_matching(path1)
    norm2 = preprocess_image_for_matching(path2)
    
    img0 = matcher.load_image(norm1, resize=resize)
    img1 = matcher.load_image(norm2, resize=resize)
    
    result = matcher(img0, img1)
    num_inliers = int(result.get("num_inliers", 0))
    H = result.get("H", None)
    
    return num_inliers, H, result, norm1, norm2

# 6) BENZERLİK SKORU
def similarity_score(result: Dict[str, Any]) -> float:
    num_inliers = float(result.get("num_inliers", 0))
    all_kpts0 = result.get("all_kpts0", [])
    all_kpts1 = result.get("all_kpts1", [])
    
    n0 = len(all_kpts0)
    n1 = len(all_kpts1)
    denom = max(min(n0, n1), 1)
    
    return num_inliers / denom

# 7) ÜRÜN ÜZERİNDE DEĞİŞİM HARİTASI
def save_product_change_overlay(norm_query, norm_ref, H, ref_color_path, resize=512, diff_threshold=30, out_dir="product_change", base_name="product"):
    if H is None:
        return 1.0, "", ""
        
    q = cv2.imread(norm_query, cv2.IMREAD_GRAYSCALE)
    r = cv2.imread(norm_ref, cv2.IMREAD_GRAYSCALE)
    
    q = cv2.resize(q, (resize, resize))
    r = cv2.resize(r, (resize, resize))
    
    warped_q = cv2.warpPerspective(q, H, (resize, resize))
    diff = cv2.absdiff(warped_q, r)
    _, diff_bin = cv2.threshold(diff, diff_threshold, 255, cv2.THRESH_BINARY)
    
    product_mask, ref_color_resized = get_product_mask(ref_color_path, resize=resize)
    diff_product = cv2.bitwise_and(diff_bin, product_mask)
    
    product_pixels = np.count_nonzero(product_mask)
    change_ratio = float(np.count_nonzero(diff_product)) / float(product_pixels) if product_pixels > 0 else 1.0
    
    os.makedirs(out_dir, exist_ok=True)
    mask_path = os.path.join(out_dir, f"{base_name}_mask.png")
    cv2.imwrite(mask_path, diff_product)
    
    overlay = ref_color_resized.copy()
    overlay[diff_product > 0] = [0, 0, 255] # Değişen yerler KIRMIZI
    overlay_path = os.path.join(out_dir, f"{base_name}_overlay.png")
    cv2.imwrite(overlay_path, overlay)
    
    return change_ratio, overlay_path, mask_path

# 8) 1xN SORGU + ANALİZ + KARAR
if __name__ == "__main__":
   
    QUERY_IMAGE_PATH = "sorgu_resmi.jpg"  # Test etmek istediğin resim
    REF_IMAGES_DIR = "reference_images"   # Referans resimlerinin olduğu klasör
    # -----------------------------------------------

    if not os.path.exists(REF_IMAGES_DIR):
        print(f"[HATA] Referans klasörü bulunamadı: {REF_IMAGES_DIR}")
        exit()

    ref_files = [os.path.join(REF_IMAGES_DIR, f) for f in os.listdir(REF_IMAGES_DIR) 
                 if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    best_score = -1.0
    best_data = None

    print(f"\n--- {len(ref_files)} Referans Arasında En İyi Eşleşme Aranıyor ---")
    for ref_path in ref_files:
        try:
            num_inliers, H, result, n_q, n_r = match_two_images(QUERY_IMAGE_PATH, ref_path)
            score = similarity_score(result)
            print(f"-> {os.path.basename(ref_path)} | Skor: {score:.4f} | Inliers: {num_inliers}")
            
            if score > best_score:
                best_score = score
                best_data = (ref_path, score, result, H, n_q, n_r)
        except Exception as e:
            print(f"[HATA] {ref_path} işlenirken hata oluştu: {e}")

    if best_data:
        ref_path, score, result, H, n_q, n_r = best_data
        print(f"\n=== EN İYİ EŞLEŞME BİLGİLERİ ===")
        print(f"Dosya: {ref_path}")
        print(f"Benzerlik Skoru: {score:.4f}")

        # Değişim analizi
        ratio, over_p, mask_p = save_product_change_overlay(n_q, n_r, H, ref_path)
        
        print(f"\n=== ANALİZ SONUCU ===")
        print(f"Üründeki Değişim Oranı: %{ratio*100:.2f}")
        
        # KARAR MEKANİZMASI
        SIM_THRESHOLD = 0.85
        MAX_CHANGE = 0.10 # %10'dan fazla fark varsa geçersiz say

        if score >= SIM_THRESHOLD and ratio <= MAX_CHANGE:
            print(">>> SONUÇ: ÜRÜN GEÇERLİ (OK) <<<")
        else:
            print(">>> SONUÇ: ÜRÜN GEÇERSİZ (HATALI/FARKLI) <<<")
            
        print(f"Görsel rapor kaydedildi: {over_p}")
    else:
        print("[HATA] Uygun bir referans eşleşmesi bulunamadı.")