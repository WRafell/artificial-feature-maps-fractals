from torchvision.utils import save_image
from torchvision.transforms import v2
from PIL import Image, ImageFilter
from pathlib import Path
import pickle
import torch
import random
import torch.nn.functional as F
import torchvision.transforms as T


class GaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            sigma = random.random() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img


class Transforms:
    def __init__(self, resize_int: int = 224):
        self.augment_transforms = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.uint8, scale=True),
            v2.RandomResizedCrop(size=(resize_int, resize_int), scale=(0.5, 1.0), interpolation=Image.BICUBIC),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            v2.RandomApply([v2.RandomChannelPermutation()], p=0.2),
            v2.RandomApply(
                [v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)], 
                p=0.5
            ),
            v2.RandomApply([v2.RandAugment()], p=0.2),
            v2.RandomGrayscale(p=0.1),    
            v2.RandomApply([v2.ElasticTransform(alpha=250.0)], p=0.2),
            v2.RandomApply([v2.RandomEqualize()], p=0.2),
            v2.RandomApply([v2.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.))], p=0.2),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.base_transforms = T.Compose([
            T.Resize((resize_int, resize_int)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])



class PatchDataset(torch.utils.data.Dataset):
    def __init__(self, img_paths: list, transforms: T.Compose):
        super().__init__()
        self.img_paths = img_paths
        self.transforms = transforms
        self.coords = [get_patch_coordinates(img_path) for img_path in img_paths]

    
    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        coords = self.coords[idx]
        with open(img_path, "rb") as f:
            img = Image.open(f).convert("RGB")        
        img = self.transforms(img)
        return img, img_path, coords
    

class FeatureMapDataset(torch.utils.data.Dataset):
    def __init__(self, data: list, is_train: bool):
        self.data = [(Path(file), label) for file, label in data]
        self.transforms = self.get_transforms(is_train)

    @staticmethod
    def get_transforms(is_train: bool):
        transforms = [
            T.CenterCrop(224),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
        if is_train:
            transforms.insert(1, T.RandomHorizontalFlip())
            transforms.insert(2, T.RandomVerticalFlip())
        return T.Compose(transforms)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        file, label = self.data[item]
        features = self.load_features(file)
        features = self.transforms(features)
        return features, label, Path(file).stem

    @staticmethod
    def load_features(file: Path):
        with file.open('rb') as f:
            features = torch.tensor(pickle.load(f)[0])
        return features
    

def patch_loader(img_paths: list, transforms: T.Compose, batch_size: int):

    dataset = PatchDataset(img_paths=img_paths, transforms=transforms)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4)


def base_feature_map(lsize: int, csize: int, num_features: int):
    """
    args:
        lsize (int): height of the feature cube
        csize (int): width of the feature cube
        num_features (int): depth of the feature cube
    return
        empty feature_map (numpy.array)
    """
    return torch.zeros((1, num_features, lsize, csize))


def get_patch_coordinates(img_path: str) -> tuple[int, int]:
    """
    Extract patch coordinates from the image filename.
    
    Args:
        img_path (str): Path of the current patch
    
    Returns:
        tuple[int, int]: Coordinates of the patch as (row, col)
    """
    img_name = Path(img_path).stem
    row, col = map(int, img_name.split('-')[1].split('_'))
    return row, col


def build_feature_map(
        slide_name: str,
        slide_label: int,
        feature_map_name: str,
        use_patch_classifier: bool,
        patch_classifier: torch.nn.Module,
        transforms: T.Compose,
        batch_size: int,
        patches: list,
        feature_map_dir: str,
        lsize: int,
        csize: int,
        num_features: int,
        normal_label: int,
        save_heatmap: bool = False) -> None:
    
    feature_map = base_feature_map(lsize, csize, num_features)

    if not use_patch_classifier:
        if slide_label != normal_label:
            patch_label = [0] * num_features
            patch_label[slide_label] = 1
            for p in patches:
                row, col = get_patch_coordinates(p)
                feature_map[0, :, row, col] = torch.tensor(patch_label)
    else:
        loader = patch_loader(patches, transforms, batch_size)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        patch_classifier.to(device)
        patch_classifier.eval()
        with torch.no_grad():
            for batch in loader:
                imgs = batch[0]
                img_paths = batch[1]
                imgs = imgs.to(device)
                outputs = patch_classifier(imgs)
                scores = F.softmax(outputs, dim=1)

                _, predictions = torch.max(scores, 1)
                for i, prediction in enumerate(predictions):
                    if prediction.item() != normal_label:
                        row, col = get_patch_coordinates(img_paths[i])
                        feature_map[0, :, row, col] = scores[i]
    
    with open(f"{feature_map_dir}/{feature_map_name}.txt", 'wb') as f:
        pickle.dump(feature_map.tolist(), f)

    if save_heatmap:    
        heatmap_dir = feature_map_dir.replace('feature_maps', 'heatmap')
        Path(heatmap_dir).mkdir(exist_ok=True, parents=True)
        save_image(feature_map, f"{heatmap_dir}/{feature_map_name}.png")  

    return











    