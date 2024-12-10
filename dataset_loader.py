import os
import random
import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


def find_closest_contour_to_center(img_bin, center_x, center_y):
    contours, hierarchy = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None
    min_dist = float('inf')
    chosen_contour = None
    for cnt in contours:
        M = cv2.moments(cnt)
        if M['m00'] > 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            dist = (cx - center_x) ** 2 + (cy - center_y) ** 2
            if dist < min_dist:
                min_dist = dist
                chosen_contour = cnt
    return chosen_contour


def crop_and_resize_contour(img_gray, contour, output_size=448):
    x, y, w, h = cv2.boundingRect(contour)
    roi = img_gray[y:y + h, x:x + w]

    max_side = max(w, h)
    square = np.zeros((max_side, max_side), dtype=roi.dtype)

    start_x = (max_side - w) // 2
    start_y = (max_side - h) // 2
    square[start_y:start_y + h, start_x:start_x + w] = roi

    square_resized = cv2.resize(square, (output_size, output_size), interpolation=cv2.INTER_AREA)
    return square_resized


def preprocess_image(img_path, downscale_width=1024):
    # Аналогично логике из предыдущего кода
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    orig_h, orig_w = img.shape[:2]
    scale_factor = downscale_width / orig_w
    downscale_h = int(orig_h * scale_factor)
    img_small = cv2.resize(img, (downscale_width, downscale_h), interpolation=cv2.INTER_AREA)

    center_x_small, center_y_small = downscale_width // 2, downscale_h // 2

    # OTSU бинаризация + инверсия
    _, img_bin = cv2.threshold(img_small, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img_bin = 255 - img_bin

    closest_contour_small = find_closest_contour_to_center(img_bin, center_x_small, center_y_small)
    if closest_contour_small is None:
        return None

    # Масштабируем контур обратно
    closest_contour = np.array(
        [[[int(pt[0][0] / scale_factor), int(pt[0][1] / scale_factor)]] for pt in closest_contour_small])
    obj_img = crop_and_resize_contour(img, closest_contour, output_size=448)
    return obj_img


def get_augmentation_transforms():
    # Аугментации применяем уже к вырезанному 448x448 объекту
    return T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomRotation(degrees=15),
        T.ColorJitter(contrast=0.2),
        T.ToTensor(),
        T.Normalize([0.5], [0.5])
    ])


class LegoDataset(Dataset):
    def __init__(self, root_dir, downscale_width=1024, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.downscale_width = downscale_width
        self.samples = []

        classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

        for cls_name in classes:
            class_dir = os.path.join(root_dir, cls_name)
            if not os.path.isdir(class_dir):
                continue
            # Собираем все файлы этого класса
            images = [f for f in os.listdir(class_dir) if f.lower().endswith('.jpg')]
            for fname in images:
                path = os.path.join(class_dir, fname)
                self.samples.append((path, self.class_to_idx[cls_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Берём один образ
        img_path, label = self.samples[idx]

        # Определяем класс этой картинки
        class_id = label
        # Найдём ещё 2 других кадра того же класса
        class_name = list(self.class_to_idx.keys())[list(self.class_to_idx.values()).index(class_id)]
        class_dir = os.path.join(self.root_dir, class_name)
        all_class_images = [os.path.join(class_dir, f) for f in os.listdir(class_dir) if f.lower().endswith('.jpg')]

        # Убедимся, что есть как минимум 3 кадра
        # Если их меньше 3, то дублируем случайно выбранные
        if len(all_class_images) < 3:
            needed = 3 - len(all_class_images)
            # Просто продублируем исходные
            all_class_images = all_class_images + random.choices(all_class_images, k=needed)

        # Выбираем 2 дополнительных случайных изображения из того же класса, отличных от img_path
        # Если в классе много изображений, выбираем случайно
        other_imgs = [p for p in all_class_images if p != img_path]
        if len(other_imgs) >= 2:
            chosen_others = random.sample(other_imgs, 2)
        else:
            # Если не хватает разных, просто дублируем исходный
            chosen_others = random.choices(all_class_images, k=2)

        three_images = [img_path] + chosen_others

        processed_tensors = []
        for impath in three_images:
            obj_img = preprocess_image(impath, downscale_width=self.downscale_width)
            if obj_img is None:
                # Если не получилось найти контур, попробуем просто пустой чёрный кадр
                obj_img = np.zeros((448, 448), dtype=np.uint8)

            obj_pil = Image.fromarray(obj_img)
            if self.transform:
                img_tensor = self.transform(obj_pil)  # [1,H,W]
            else:
                img_tensor = T.ToTensor()(obj_pil)
                img_tensor = T.Normalize([0.5], [0.5])(img_tensor)
            processed_tensors.append(img_tensor)

        # Стэкаем по каналу: получим [3, H, W]
        # Но сейчас у нас три тензора по [1,H,W]. Можно их сконкатенировать по оси каналов
        combined = torch.cat(processed_tensors, dim=0)  # dim=0 объединит [1,H,W]+[1,H,W]+[1,H,W] = [3,H,W]

        return combined, label