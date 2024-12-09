import argparse
import os
import random
import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T

def get_test_transforms():
    return T.Compose([
        T.RandomHorizontalFlip(),
        # T.RandomRotation(degrees=15),
        # T.ColorJitter(contrast=0.2),
        T.ToTensor(),
        # T.Normalize([0.5],[0.5])
    ])

def find_closest_contour_to_center(img_bin, center_x, center_y):
    contours, hierarchy = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None, None
    min_dist = float('inf')
    chosen_contour = None
    for cnt in contours:
        M = cv2.moments(cnt)
        if M['m00'] > 0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            dist = (cx - center_x)**2 + (cy - center_y)**2
            if dist < min_dist:
                min_dist = dist
                chosen_contour = cnt
    return chosen_contour, contours

def crop_and_resize_contour(img_gray, contour, output_size=448):
    x,y,w,h = cv2.boundingRect(contour)
    roi = img_gray[y:y+h, x:x+w]

    max_side = max(w,h)
    square = np.zeros((max_side, max_side), dtype=roi.dtype)

    start_x = (max_side - w)//2
    start_y = (max_side - h)//2
    square[start_y:start_y+h, start_x:start_x+w] = roi

    square_resized = cv2.resize(square, (output_size, output_size), interpolation=cv2.INTER_AREA)
    return square_resized

def main():
    parser = argparse.ArgumentParser(description="Localization with contours using downscaled image first, then rescale bounding box.")
    parser.add_argument('--input_dir', type=str, required=True, help='Path to input images directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to output directory')
    parser.add_argument('--downscale_width', type=int, default=1024, help='Width to scale down the image for contour detection')
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not files:
        raise ValueError(f"No image files found in {input_dir}")

    random_file = random.choice(files)
    img_path = os.path.join(input_dir, random_file)

    # Читаем большое изображение
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read the image: {img_path}")

    orig_h, orig_w = img.shape[:2]

    # Масштабируем по ширине
    downscale_w = args.downscale_width
    scale_factor = downscale_w / orig_w
    downscale_h = int(orig_h * scale_factor)
    img_small = cv2.resize(img, (downscale_w, downscale_h), interpolation=cv2.INTER_AREA)

    center_x_small, center_y_small = downscale_w//2, downscale_h//2

    # Бинаризация на уменьшенном изображении
    _, img_bin = cv2.threshold(img_small, 50, 255, cv2.THRESH_OTSU)

    # Инвертируем результат, если исходное изображение почти всё белое
    img_bin = 255 - img_bin

    # Находим контур ближайший к центру на уменьшенной версии + все контуры
    closest_contour_small, all_contours_small = find_closest_contour_to_center(img_bin, center_x_small, center_y_small)
    if closest_contour_small is None:
        raise ValueError("No contours found on downscaled image!")

    # Преобразуем все контуры в оригинальный масштаб
    all_contours = []
    for cnt_s in all_contours_small:
        cnt = np.array([[[int(pt[0][0]/scale_factor), int(pt[0][1]/scale_factor)]] for pt in cnt_s])
        all_contours.append(cnt)

    # Масштабируем выбранный контур тоже
    closest_contour = np.array([[[int(pt[0][0]/scale_factor), int(pt[0][1]/scale_factor)]] for pt in closest_contour_small])

    # Отрисовываем все контуры
    img_contour_vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(img_contour_vis, all_contours, -1, (0,255,0), 2) # Все контуры зелёным

    # Отрисовываем выбранный контур поверх всех - красным
    cv2.drawContours(img_contour_vis, [closest_contour], -1, (0,0,255), 3)

    contour_vis_path = os.path.join(output_dir, f"contour_vis_{random_file}")
    cv2.imwrite(contour_vis_path, img_contour_vis)
    print(f"Saved contour visualization with all contours and chosen one to {contour_vis_path}")

    # Определяем bounding box выбранного контура в оригинальном масштабе
    x,y,w,h = cv2.boundingRect(closest_contour)

    # Вырезаем и подготавливаем к классификации
    obj_img = crop_and_resize_contour(img, closest_contour, output_size=448)
    obj_img_pil = Image.fromarray(obj_img)
    transform = get_test_transforms()
    obj_tensor = transform(obj_img_pil)

    obj_denorm = obj_tensor * 0.5 + 0.5
    obj_denorm = torch.clamp(obj_denorm, 0, 1)
    obj_denorm_pil = T.ToPILImage()(obj_denorm)

    final_out_path = os.path.join(output_dir, f"final_{random_file}")
    obj_denorm_pil.save(final_out_path)
    print(f"Saved final processed image to {final_out_path}")

if __name__ == '__main__':
    main()