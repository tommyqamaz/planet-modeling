import albumentations as A


def get_train_augmentation(img_size: int) -> A.Compose:
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.HueSaturationValue(
                hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5
            ),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.ShiftScaleRotate(),
            A.Resize(height=img_size, width=img_size),
            A.Normalize(),
        ]
    )


def get_val_augmentation(img_size: int) -> A.Compose:
    return A.Compose([A.Resize(height=img_size, width=img_size), A.Normalize()])
