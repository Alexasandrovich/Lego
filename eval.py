import argparse
import os
import random
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as T
from model import LegoClassifierModel
from dataset_loader import find_closest_contour_to_center, crop_and_resize_contour  # из предыдущего кода


# Если они в другом файле, импорт скорректировать
# Можно также скопировать эти функции сюда.

def get_augmentation_transforms_for_inference():
    # Те же аугментации, что и при обучении или чуть упрощённый вариант.
    # Здесь можно убрать рандомизации для более стабильного предикта.
    return T.Compose([
        T.ToTensor(),
        T.Normalize([0.5], [0.5])
    ])


def load_class_mapping(train_dir):
    # Предполагаем, что классы - это папки
    classes = sorted(os.listdir(train_dir))
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    return class_to_idx, idx_to_class


def preprocess_image(img_path, downscale_width=1024):
    # Аналогично __getitem__ в датасете: локализация объекта
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read the image: {img_path}")

    orig_h, orig_w = img.shape[:2]
    scale_factor = downscale_width / orig_w
    downscale_h = int(orig_h * scale_factor)
    img_small = cv2.resize(img, (downscale_width, downscale_h), interpolation=cv2.INTER_AREA)

    center_x_small, center_y_small = downscale_width // 2, downscale_h // 2

    # OTSU бинаризация
    _, img_bin = cv2.threshold(img_small, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Инверсия
    img_bin = 255 - img_bin

    closest_contour_small = find_closest_contour_to_center(img_bin, center_x_small, center_y_small)
    if closest_contour_small is None:
        # Если не нашли контур, просто вернём None
        return None

    # Масштабируем контур обратно
    closest_contour = np.array(
        [[[int(pt[0][0] / scale_factor), int(pt[0][1] / scale_factor)]] for pt in closest_contour_small])
    obj_img = crop_and_resize_contour(img, closest_contour, output_size=448)
    return obj_img


def pick_example_from_class(train_dir, class_name):
    class_path = os.path.join(train_dir, class_name)
    images = [f for f in os.listdir(class_path) if f.lower().endswith('.jpg')]
    if not images:
        return None
    return os.path.join(class_path, random.choice(images))


def main():
    parser = argparse.ArgumentParser(
        description="Predict with trained model on 5 random images and show top-5 classes examples.")
    parser.add_argument('--input_path', type=str, required=True,
                        help='Path to a directory with images or a txt file with image paths')
    parser.add_argument('--train_dir', type=str, required=True,
                        help='Path to the train directory with class subfolders')
    parser.add_argument('--model_path', type=str, default='best_model.pth', help='Path to the trained model')
    parser.add_argument('--output_dir', type=str, required=True, help='Where to save the output images')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Получаем список изображений для предсказания
    if os.path.isdir(args.input_path):
        all_files = [os.path.join(args.input_path, f) for f in os.listdir(args.input_path)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    else:
        # Если это файл, считаем построчно пути
        with open(args.input_path, 'r') as f:
            all_files = [line.strip() for line in f if line.strip()]

    if len(all_files) < 5:
        raise ValueError("Need at least 5 images for the test")

    chosen_files = random.sample(all_files, 5)

    # Загрузка модели
    class_to_idx, idx_to_class = load_class_mapping(args.train_dir)
    num_classes = len(class_to_idx)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = LegoClassifierModel(num_classes=num_classes, pretrained=False)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()

    transform = get_augmentation_transforms_for_inference()

    for img_path in chosen_files:
        obj_img = preprocess_image(img_path)
        if obj_img is None:
            print(f"Warning: No contour found for {img_path}, skipping.")
            continue

        # Преобразуем в PIL и затем в тензор
        obj_pil = Image.fromarray(obj_img)
        input_tensor = transform(obj_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_tensor)
            probs = F.softmax(outputs, dim=1)

        top5_probs, top5_indices = torch.topk(probs, 5, dim=1)
        top5_indices = top5_indices[0].cpu().numpy()
        top5_probs = top5_probs[0].cpu().numpy()

        # Предположим, имя класса - это строка директорий, например "1" или "2"
        top5_classes = [idx_to_class[idx] for idx in top5_indices]

        # Сохраняем исходное изображение предсказания
        base_name = os.path.basename(img_path)
        pred_out_path = os.path.join(args.output_dir, f"pred_{base_name}")
        obj_pil.save(pred_out_path)
        print(f"Saved predicted object image: {pred_out_path}")

        # Для каждой из top-5 предсказанных классов возьмём пример изображения
        for rank, cls_name in enumerate(top5_classes, start=1):
            example_path = pick_example_from_class(args.train_dir, cls_name)
            if example_path is not None:
                # Скопируем/сохраним один из примеров в выходную директорию
                example_img = Image.open(example_path).convert('L')
                example_out_path = os.path.join(args.output_dir, f"top{rank}_{cls_name}_{base_name}")
                example_img.save(example_out_path)
                print(f"Saved class example: {example_out_path}")
            else:
                print(f"No example found for class {cls_name}")

        # Выводим классы и вероятности в консоль
        print("Top-5 predictions:")
        for cls_name, p in zip(top5_classes, top5_probs):
            print(f"  Class: {cls_name}, Prob: {p:.4f}")


if __name__ == '__main__':
    main()