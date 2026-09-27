import torch
import random
from torchvision import transforms 

def get_device(force_cpu):
	try:
		if force_cpu:
			device= torch.device("cpu")
			print("force using CPU")
		elif torch.backends.mps.is_available():
			device = torch.device("mps")
			print("using Apple MX chipset")
		elif torch.cuda.is_available():
			device = torch.device("cuda")
			print("using Nvidia GPU")
		else:
			device = torch.device("cpu")
			print("using CPU")
		return device
	except:
		print("MPS is not supported for this version of PyTorch")
		if torch.cuda.is_available():
			device = torch.device("cuda")
			print("using Nvidia GPU")
		else:
			device = torch.device("cpu")
			print("using CPU")
		return device
def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

def setup_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_transforms(model_name):
    """Get model-specific image transformations.
    
    Args:
        model_name: 'resnet' or 'inception' or 'transformer'
    
    Returns:
        tuple: (train_transforms, test_transforms)
    """
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    if model_name == 'pretrained':
        transform_train = transforms.Compose(
                    [
                        transforms.RandomHorizontalFlip(p=0.5),
                        transforms.RandomVerticalFlip(p=0.5),
                        transforms.PILToTensor(),
                    ]
                )
        transform_test = transforms.PILToTensor()
        return transform_train, transform_test
    elif model_name in {'resnet', 'resnet101'}:
        size = 224
    elif model_name in {'inception', 'inception_v3'}:
        size = 299
    elif model_name == 'convnext_base':
        size = 224
    elif model_name == 'efficientnet_v2_m':
        size = 480
    else:
        raise ValueError(f"Unknown architecture: {model_name}")
    
    trs = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])
    
    tss = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])
    
    return trs, tss
