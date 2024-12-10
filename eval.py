import argparse
import os
import random
import cv2
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as T

from model import LegoClassifierModel
from dataset_loader import find_closest_contour_to_center, crop_and_resize_contour


def preprocess_image(img_path, downscale_width=1024):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    orig_h, orig_w = img.shape[:2]

    scale_factor = downscale_width / orig_w
    downscale_h = int(orig_h * scale_factor)
    img_small = cv2.resize(img, (downscale_width, downscale_h), interpolation=cv2.INTER_AREA)

    center_x_small, center_y_small = downscale_width // 2, downscale_h // 2
    # OTSU + инверсия
    _, img_bin = cv2.threshold(img_small, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img_bin = 255 - img_bin

    closest_contour_small = find_closest_contour_to_center(img_bin, center_x_small, center_y_small)
    if closest_contour_small is None:
        # Если контур не найден, вернём чёрный квадрат
        return np.zeros((448, 448), dtype=np.uint8)

    closest_contour = np.array(
        [[[int(pt[0][0] / scale_factor), int(pt[0][1] / scale_factor)]] for pt in closest_contour_small])
    obj_img = crop_and_resize_contour(img, closest_contour, output_size=448)
    return obj_img


def get_inference_transform():
    return T.Compose([
        T.ToTensor(),
        T.Normalize([0.5], [0.5])
    ])


def load_class_mapping(train_dir):
    classes = sorted(os.listdir(train_dir))
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    return class_to_idx, idx_to_class


def pick_examples_from_class(train_dir, class_name, num_examples=3):
    class_path = os.path.join(train_dir, class_name)
    if not os.path.isdir(class_path):
        return []

    images = [f for f in os.listdir(class_path) if f.lower().endswith('.jpg')]
    if len(images) == 0:
        return []
    if len(images) < num_examples:
        # Если меньше, чем нужно, дублируем
        images = images + random.choices(images, k=num_examples - len(images))
    chosen = random.sample(images, num_examples)
    chosen_paths = [os.path.join(class_path, c) for c in chosen]
    return chosen_paths


def load_and_resize_for_display(img_path):
    # Грузим в цвете для визуализации
    img = cv2.imread(img_path)
    if img is None:
        img = np.zeros((448, 448, 3), dtype=np.uint8)
    else:
        img = cv2.resize(img, (448, 448), interpolation=cv2.INTER_AREA)
    return img


def create_collage(top_row_imgs, bottom_row_imgs, text_lines):
    # top_row_imgs и bottom_row_imgs - списки из 3 изображений (np.array)
    # text_lines - список строк, которые надо вывести на итоговое изображение

    # Склеиваем по горизонтали каждую строку
    top_row = np.hstack(top_row_imgs)  # [H, 3*W, 3]
    bottom_row = np.hstack(bottom_row_imgs) if bottom_row_imgs else None

    if bottom_row is not None:
        collage = np.vstack([top_row, bottom_row])  # [2*H, 3*W, 3]
    else:
        collage = top_row

    # Пишем текст сверху слева
    font = cv2.FONT_HERSHEY_SIMPLEX
    y_offset = 30
    for line in text_lines:
        cv2.putText(collage, line, (10, y_offset), font, 0.7, (255, 0, 0), 2, cv2.LINE_AA)
        y_offset += 30

    return collage


def main():
    parser = argparse.ArgumentParser(description="Inference with 3 images + show top class examples")
    parser.add_argument('--img1', type=str, required=True, help='Path to first image')
    parser.add_argument('--img2', type=str, required=True, help='Path to second image')
    parser.add_argument('--img3', type=str, required=True, help='Path to third image')
    parser.add_argument('--train_dir', type=str, required=True, help='Path to train directory for class mapping')
    parser.add_argument('--model_path', type=str, default='model.pth', help='Path to trained model')
    parser.add_argument('--output_file', type=str, required=True, help='Output image file')
    args = parser.parse_args()

    class_to_idx, idx_to_class = load_class_mapping(args.train_dir)
    num_classes = len(class_to_idx)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = LegoClassifierModel(num_classes=num_classes, pretrained=False)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()

    img_paths = [args.img1, args.img2, args.img3]
    processed_imgs = []
    for path in img_paths:
        obj_img = preprocess_image(path)
        if obj_img is None:
            obj_img = np.zeros((448, 448), dtype=np.uint8)
        processed_imgs.append(obj_img)

    transform = get_inference_transform()
    tensors = []
    for img_arr in processed_imgs:
        obj_pil = Image.fromarray(img_arr)
        img_tensor = transform(obj_pil)  # [1,H,W]
        tensors.append(img_tensor)

    # Объединяем в [3,H,W]
    combined_input = torch.cat(tensors, dim=0).unsqueeze(0).to(device)  # [1,3,448,448]

    with torch.no_grad():
        outputs = model(combined_input)
        probs = F.softmax(outputs, dim=1)[0].cpu().numpy()

    top5_indices = np.argsort(probs)[::-1][:5]
    top5_probs = probs[top5_indices]
    top5_classes = [idx_to_class[i] for i in top5_indices]

    # Самый вероятный класс
    final_class = top5_classes[0]
    final_prob = top5_probs[0]

    # Подбираем примеры из final_class
    ref_paths = pick_examples_from_class(args.train_dir, final_class, num_examples=3)
    ref_imgs = [load_and_resize_for_display(p) for p in ref_paths]

    # Загружаем исходные изображения в цвете для визуализации
    input_imgs_color = []
    for p in img_paths:
        img_color = load_and_resize_for_display(p)
        input_imgs_color.append(img_color)

    # Если вдруг не нашли референсные картинки (крайний случай) - создадим пустые
    if len(ref_imgs) < 3:
        needed = 3 - len(ref_imgs)
        for _ in range(needed):
            ref_imgs.append(np.zeros((448, 448, 3), dtype=np.uint8))

    # Подготовим текст
    text_lines = []
    text_lines.append(f"Final chosen class: {final_class} ({final_prob * 100:.2f}%)")
    text_lines.append("Top-5 predictions:")
    for c, p in zip(top5_classes, top5_probs):
        text_lines.append(f"{c}: {p * 100:.2f}%")

    # Создаём коллаж: верхний ряд - входные кадры, нижний ряд - эталонные картинки класса
    collage = create_collage(input_imgs_color, ref_imgs, text_lines)

    cv2.imwrite(args.output_file, collage)
    print(f"Saved result to {args.output_file}")


if __name__ == '__main__':
    main()