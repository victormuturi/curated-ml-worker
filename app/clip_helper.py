#this software is provided as is, and is not guaranteed to work or be suitable for any particular purpose
#for use only in the context of the "Machine Aesthetics" course at GSD. Do not share or distribute
#copyright 2023-2024 Panagiotis Michalatos : pan.michalatos@gmail.com

import torch
from torchvision import  transforms
import os
from PIL import Image
import cv2
import numpy as np
import math
import clip

import ssl
ssl._create_default_https_context = ssl._create_unverified_context


def getAllImagesFromFolder(folder):
    image_files = os.listdir(folder)
    image_files = [f for f in image_files if f.endswith('.png') or f.endswith('.jpg')]
    return image_files


CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


CLIP_TRANSFORM = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(CLIP_MEAN, CLIP_STD)
        ])

CLIP_TENSOR_TRANSFORM = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.Normalize(CLIP_MEAN, CLIP_STD)
        ])

CLIP_NORMALIZATION = transforms.Normalize(CLIP_MEAN, CLIP_STD)


class TextVector:
    def __init__(self, text, vector = None):
        self.text = text
        self.vector = vector

class ClipHelper:
    def __init__(self, device : torch.device):
        self.device = device

        self.model, self.preprocess = clip.load("ViT-B/32", device="cpu")
        self.model = self.model.to(self.device)
        self.model.eval()

    def encodeImageBatchWithGrad(self, image_batch : torch.Tensor) -> torch.Tensor:        
        image_batch : torch.Tensor = CLIP_NORMALIZATION(image_batch)        
        image_features : torch.Tensor = self.model.encode_image(image_batch)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)          

        return image_features
    
    def encodeImage(self, image) -> np.ndarray:
        #if image is opencv image, convert to PIL

        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
        
        with torch.no_grad():
            if isinstance(image, torch.Tensor):
                image = CLIP_TENSOR_TRANSFORM(image.to(self.device))
            else:
                image : torch.Tensor = self.preprocess(image)
                image = image.unsqueeze(0).to(self.device)

            
            image_features : torch.Tensor = self.model.encode_image(image)[0]

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)           

        return image_features.cpu().numpy()
    
    def encodeText(self, text) -> np.ndarray:
        with torch.no_grad():
            text_tensor : torch.Tensor = clip.tokenize(text)
            text_tensor = text_tensor.to(self.device)

            text_features : torch.Tensor = self.model.encode_text(text_tensor)

            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return text_features.cpu().numpy()
    
    def encodeImageToTensor(self, image) -> torch.Tensor:      
        #if image is opencv image, convert to PIL

        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
        
        with torch.no_grad():

            if isinstance(image, torch.Tensor):
                image = CLIP_TENSOR_TRANSFORM(image.to(self.device))
            else:
                image : torch.Tensor = self.preprocess(image)
                image = image.unsqueeze(0).to(self.device)

            
            image_features : torch.Tensor = self.model.encode_image(image)[0]

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)           

        return image_features
    
    def encodeTextToTensor(self, text) -> torch.Tensor:      
        with torch.no_grad():
            text_tensor : torch.Tensor = clip.tokenize(text)
            text_tensor = text_tensor.to(self.device)

            text_features : torch.Tensor = self.model.encode_text(text_tensor)

            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return text_features
    
    def createTextVector(self, text : str) -> TextVector:
        return TextVector(text, self.encodeText(text)[0])
    

    # def classifyAllImagesInFolder(self, folder, max_count = -1) -> list[tuple[str, np.ndarray]]:
    #     image_files = getAllImagesFromFolder(folder)
    #     image_files = [os.path.join(folder, f) for f in image_files]

    #     if max_count > 0 and max_count < len(image_files):
    #         image_files = random.sample(image_files, max_count)

    #     results = []
    #     for i, image_file in enumerate(image_files):
    #         print(f'processing image {i}/{len(image_files)}')
    #         image = Image.open(image_file).convert('RGB')
    #         outputs = self.classify(image)
    #         results.append((image_file, outputs[0].cpu().numpy()))

    #     return results


class Vec2:
    def __init__(self, x:float = 0.0, y:float = 0.0):
        self.x = x
        self.y = y

    def __add__(self, other: 'Vec2'):
        return Vec2(self.x + other.x, self.y + other.y)

    def __sub__(self, other: 'Vec2'):
        return Vec2(self.x - other.x, self.y - other.y)

    def __mul__(self, other: float):
        return Vec2(self.x * other, self.y * other)

    def __truediv__(self, other: float):
        return Vec2(self.x / other, self.y / other)

    def __str__(self):
        return f'({self.x}, {self.y})'

    def __repr__(self):
        return f'({self.x}, {self.y})'

    @property
    def length(self):
        return math.sqrt(self.x * self.x + self.y * self.y)
    
    def distance(self, other : 'Vec2'):
        dx = self.x - other.x
        dy = self.y - other.y
        return math.sqrt(dx * dx + dy * dy)
    
    def distanceSquared(self, other : 'Vec2'):
        dx = self.x - other.x
        dy = self.y - other.y
        return dx * dx + dy * dy
    
    def clone(self):
        return Vec2(self.x, self.y)
    
    def set(self, x, y):
        self.x = x
        self.y = y

    def setMin(self, other : 'Vec2'):
        self.x = min(self.x, other.x)
        self.y = min(self.y, other.y)

    def setMax(self, other : 'Vec2'):
        self.x = max(self.x, other.x)
        self.y = max(self.y, other.y)

    def normalized(self):
        copy = self.clone()
        copy.normalize()
        return copy
    
    def normalize(self):
        l = self.length
        if l > 0:
            self.x /= l
            self.y /= l

    def lerp(self, other : 'Vec2', t : float):
        return Vec2(self.x + (other.x - self.x) * t, self.y + (other.y - self.y) * t)

    def dot(self, other: 'Vec2'):
        return self.x * other.x + self.y * other.y

    def angle(self, other : 'Vec2'):
        return np.arccos(self.dot(other) / (self.length * other.length))

    def rotate(self, angle):
        return Vec2(self.x * np.cos(angle) - self.y * np.sin(angle), self.x * np.sin(angle) + self.y * np.cos(angle))

    def toTuple(self):
        return (self.x, self.y)

    def toIntTuple(self):
        return (int(self.x), int(self.y))

    def toIntList(self):
        return [int(self.x), int(self.y)]

    def toList(self):
        return [self.x, self.y]

    def toNumpy(self):
        return np.array([self.x, self.y], dtype=np.float32)


class Bounds2:
    def __init__(self, min : Vec2 = Vec2(), max : Vec2 = Vec2()):
        self.min = min
        self.max = max

    @property
    def size(self):
        return self.max - self.min

    @property
    def width(self):
        return self.max.x - self.min.x
    
    @property
    def height(self):
        return self.max.y - self.min.y
    
    @property
    def center(self):
        return (self.min + self.max) / 2
    
    def contains(self, point : Vec2):
        return point.x >= self.min.x and point.x <= self.max.x and point.y >= self.min.y and point.y <= self.max.y
    
    def containsBounds(self, other : 'Bounds2'):
        return self.contains(other.min) and self.contains(other.max)
    
    def expand(self, amount : float):
        self.min -= Vec2(amount, amount)
        self.max += Vec2(amount, amount)

    def expandVec(self, amount : Vec2):
        self.min -= amount
        self.max += amount

    def expandToInclude(self, point : Vec2):
        self.min.setMin(point)
        self.max.setMax(point)

    def expandToIncludeBounds(self, other : 'Bounds2'):
        self.min.setMin(other.min)
        self.max.setMax(other.max)

    def intersects(self, other : 'Bounds2'):
        return self.min.x <= other.max.x and self.max.x >= other.min.x and self.min.y <= other.max.y and self.max.y >= other.min.y
    
    def intersection(self, other : 'Bounds2'):
        result = Bounds2()
        result.min.x = max(self.min.x, other.min.x)
        result.min.y = max(self.min.y, other.min.y)
        result.max.x = min(self.max.x, other.max.x)
        result.max.y = min(self.max.y, other.max.y)
        return result
    
    def union(self, other : 'Bounds2'):
        result = Bounds2()
        result.min.x = min(self.min.x, other.min.x)
        result.min.y = min(self.min.y, other.min.y)
        result.max.x = max(self.max.x, other.max.x)
        result.max.y = max(self.max.y, other.max.y)
        return result
    

class FolderBrowser:
    def __init__(self, root_folder : str, extensions : list[str] = None, folder_browser : bool = False):
        self.root_folder = root_folder
        self.extensions = extensions 
        self.folder_browser = folder_browser
        self.files = []
        self.folders = []
        self.stack = []
        self.current_folder = root_folder
        self.selected : str = None
        self.update()

    @property
    def selected_full_path(self):
        if self.selected is not None:
            return self.getFullPath(self.selected)
        else:
            return None

    def home(self):
        self.unselect()
        self.stack = []
        self.current_folder = self.root_folder
        self.update()

    def up(self):
        self.unselect()
        if len(self.stack) > 0:
            self.current_folder = self.stack.pop()
            self.update()
        else:
            self.current_folder = os.path.dirname(self.current_folder)
            self.update()

    def openFolder(self, folder : str):
        self.stack.append(self.current_folder)
        self.current_folder = self.getFullPath(folder)
        self.unselect()
        self.update()

    def onSelcected(self, file : str):
        self.selected = file

    def unselect(self):
        self.selected = None

    def onClose(self):
        self.unselect()

    def update(self):
        self.files = []
        self.folders = []
        
        if os.path.isdir(self.current_folder):
            if self.folder_browser:
                #self.folders.append('..')
                for f in os.listdir(self.current_folder):
                    full_path = os.path.join(self.current_folder, f)
                    if os.path.isdir(full_path):
                        self.folders.append(f)
            else:
                for f in os.listdir(self.current_folder):
                    full_path = os.path.join(self.current_folder, f)
                    if os.path.isdir(full_path):
                        self.folders.append(f)
                    else:
                        if self.extensions is None or f.endswith(tuple(self.extensions)):
                            self.files.append(f)
        else:
            print(f'{self.current_folder} is not a folder')

    def getFullPath(self, file : str):
        return os.path.join(self.current_folder, file)
    