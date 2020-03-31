import torchvision.transforms as transforms
from RandAugment import RandAugment
from cv2 import cv2
from PIL import Image
from segmentation.dataset import NerveSegmentationDataset


def crop_square(im, tsize):
    size = im.shape

    new_width, new_height = tsize, tsize

    left = (size[0] - new_width)/2
    top = (size[1] - new_height)/2
    right = (size[0] + new_width)/2
    bottom = (size[1] + new_height)/2
    
    im = im[int(left):int(right), int(top):int(bottom)]

    return im

def resize(im, size):
    im = cv2.resize(im, dsize=(int(size), int(size)), interpolation=cv2.INTER_AREA)
    return im

def preprocessing(image, mask, res):
    image = image / 255.0
    image, mask = resize(image, res), resize(mask, res)
    # image, mask = Image.fromarray(image), Image.fromarray(mask)
    image_transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    # image_transform.transforms.insert(0, RandAugment(2, 14))
    
    return image_transform(image).float(), image_transform(mask).float()

def real_preprocessing(image, index):
    image = image / 255.0
    image = crop_square(image, 400)

    image_transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    # image_transform.transforms.insert(0, RandAugment(2, 14))
    
    return image_transform(image).float(), index


if __name__ == '__main__':
    ds_train = NerveSegmentationDataset(root='./data/', train=True, transform=preprocessing)
    ds_test = NerveSegmentationDataset(root='./data/', train=False, transform=preprocessing)

    print(ds_train.__getitem__(10))
    print(ds_train.__getitem__(10)[0].shape)
    print(ds_train.__getitem__(10)[1].shape)
    print(ds_train.__len__())
    print(ds_test.__len__())
    
    print("DATA LOADED")