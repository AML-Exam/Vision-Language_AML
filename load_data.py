from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import json

CATEGORIES = {
    'dog': 0,
    'elephant': 1,
    'giraffe': 2,
    'guitar': 3,
    'horse': 4,
    'house': 5,
    'person': 6,
}

class PACSDatasetBaseline(Dataset):
    def __init__(self, examples, transform):
        self.examples = examples
        self.transform = transform

    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, index):
        img_path, y = self.examples[index]
        x = self.transform(Image.open(img_path).convert('RGB'))
        return x, y

class PACSDatasetClip(Dataset):
    def __init__(self, examples, transform):
        self.examples = examples
        self.transform = transform

    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, index):
        example, y = self.examples[index]
        x = self.transform(Image.open(example[0]).convert('RGB'))
        return (x, example[1]), y

def read_lines(data_path, domain_name):
    examples = {}
    with open(f'{data_path}/{domain_name}.txt') as f:
        lines = f.readlines()

    for line in lines: 
        line = line.strip().split()[0].split('/')
        category_name = line[3]
        category_idx = CATEGORIES[category_name]
        image_name = line[4]
        image_path = f'{data_path}/kfold/{domain_name}/{category_name}/{image_name}'
        if category_idx not in examples.keys():
            examples[category_idx] = [image_path]
        else:
            examples[category_idx].append(image_path)
    return examples

def read_lines_clip(data_path, domain_name):
    examples = {}
    #with open(f'{data_path}/{domain_name}.txt') as f:
    #    lines = f.readlines()
#
    #for line in lines: 
    #    line = line.strip().split()[0].split('/')
    #    category_name = line[3]
    #    category_idx = CATEGORIES[category_name]
    #    image_name = line[4]
    #    image_path = f'{data_path}/kfold/{domain_name}/{category_name}/{image_name}'
    #    if category_idx not in examples.keys():
    #        examples[category_idx] = [image_path]
    #    else:
    #        examples[category_idx].append(image_path)

    
    with open(f'{data_path}/descriptions/{domain_name}_descriptions.txt') as f:
        for line in f.readlines(): 
            dict_tmp = json.loads(line.strip())
            path = dict_tmp['image_name'].split('/')
            category_name = path[1]
            category_idx = CATEGORIES[category_name]
            image_name = path[2]
            image_path = f'{data_path}/kfold/{domain_name}/{category_name}/{image_name}'
            description = dict_tmp['descriptions']
            if category_idx not in examples.keys():
                examples[category_idx] = [(image_path, description)]
            else:
                examples[category_idx].append((image_path, description))

    return examples

def build_splits_baseline(opt):
    source_domain = 'art_painting'
    target_domain = opt['target_domain']

    source_examples = read_lines(opt['data_path'], source_domain)
    target_examples = read_lines(opt['data_path'], target_domain)

    # Compute ratios of examples for each category
    source_category_ratios = {category_idx: len(examples_list) for category_idx, examples_list in source_examples.items()}
    source_total_examples = sum(source_category_ratios.values())
    source_category_ratios = {category_idx: c / source_total_examples for category_idx, c in source_category_ratios.items()}

    # Build splits - we train only on the source domain (Art Painting)
    val_split_length = source_total_examples * 0.2 # 20% of the training split used for validation

    train_examples = []
    val_examples = []
    test_examples = []

    for category_idx, examples_list in source_examples.items():
        split_idx = round(source_category_ratios[category_idx] * val_split_length)
        for i, example in enumerate(examples_list):
            if i > split_idx:
                train_examples.append([example, category_idx]) # each pair is [path_to_img, class_label]
            else:
                val_examples.append([example, category_idx]) # each pair is [path_to_img, class_label]
    
    for category_idx, examples_list in target_examples.items():
        for example in examples_list:
            test_examples.append([example, category_idx]) # each pair is [path_to_img, class_label]
    
    # Transforms
    normalize = T.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ResNet18 - ImageNet Normalization

    train_transform = T.Compose([
        T.Resize(256),
        T.RandAugment(3, 15),
        T.CenterCrop(224),
        T.ToTensor(),
        normalize
    ])

    eval_transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        normalize
    ])

    # Dataloaders
    train_loader = DataLoader(PACSDatasetBaseline(train_examples, train_transform), batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=True)
    val_loader = DataLoader(PACSDatasetBaseline(val_examples, eval_transform), batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=False)
    test_loader = DataLoader(PACSDatasetBaseline(test_examples, eval_transform), batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=False)

    return train_loader, val_loader, test_loader

def build_splits_domain_disentangle(opt):
    source_domain = 'art_painting'
    target_domain = opt['target_domain']

    source_examples = read_lines(opt['data_path'], source_domain)
    target_examples = read_lines(opt['data_path'], target_domain)

    # Compute ratios of examples for each category in source data
    source_category_ratios = {category_idx: len(examples_list) for category_idx, examples_list in source_examples.items()}
    source_total_examples = sum(source_category_ratios.values())
    source_category_ratios = {category_idx: c / source_total_examples for category_idx, c in source_category_ratios.items()}

    # Compute ratios of examples for each category in target data
    target_category_ratios = {category_idx: len(examples_list) for category_idx, examples_list in target_examples.items()}
    target_total_examples = sum(target_category_ratios.values())
    target_category_ratios = {category_idx: c / target_total_examples for category_idx, c in target_category_ratios.items()}

    # Build splits - we train and validate on the source domain (Art Painting)
    source_val_split_length = source_total_examples * 0.2 # 20% of the source used for validation
    
    # Build splits - we train and test on the target domain
    target_test_split_length = target_total_examples * 0.2 # 20% of the target used for test

    source_train_examples = []
    target_train_examples = []
    source_val_examples = []
    #target_val_examples = []
    test_examples = []

    for category_idx, examples_list in source_examples.items():
        split_idx_s = round(source_category_ratios[category_idx] * source_val_split_length)
        for i, example in enumerate(examples_list):
            if i < split_idx_s:
                source_val_examples.append([example, category_idx]) # each pair is [path_to_img, class_label]
            else:
                source_train_examples.append([example, category_idx]) # each pair is [path_to_img, class_label]

    for category_idx, examples_list in target_examples.items():
        split_idx_t = round(target_category_ratios[category_idx] * target_test_split_length)
        for i, example in enumerate(examples_list):
            if i < split_idx_t:
                test_examples.append([example, category_idx]) # each pair is [path_to_img, class_label]
            else:
                target_train_examples.append([example, category_idx]) # each pair is [path_to_img, class_label]
    
    #for category_idx, examples_list in target_examples.items():
    #    for example in examples_list:
    #        test_examples.append([example, category_idx]) # each pair is [path_to_img, class_label]

    #val_examples = source_val_examples + target_val_examples
    
    # Transforms
    normalize = T.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ResNet18 - ImageNet Normalization

    train_transform = T.Compose([
        T.Resize(256),
        T.RandAugment(3, 15),
        T.CenterCrop(224),
        T.ToTensor(),
        normalize
    ])

    eval_transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        normalize
    ])

    # Dataloaders
    source_train_loader = DataLoader(PACSDatasetBaseline(source_train_examples, train_transform), batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=True)
    source_val_loader = DataLoader(PACSDatasetBaseline(source_val_examples, eval_transform), batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=False)
    target_train_loader = DataLoader(PACSDatasetBaseline(target_train_examples, train_transform), batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=True)
    #target_val_loader = DataLoader(PACSDatasetBaseline(target_val_examples, eval_transform), batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=False)
    test_loader = DataLoader(PACSDatasetBaseline(test_examples, eval_transform), batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=False)

    return source_train_loader, target_train_loader, source_val_loader, test_loader #source_val_loader, target_val_loader 
    #raise NotImplementedError('[TODO] Implement build_splits_domain_disentangle') #TODO

def build_splits_clip_disentangle(opt):
    source_domain = 'art_painting'
    target_domain = opt['target_domain']

    source_examples = read_lines_clip(opt['data_path'], source_domain)
    target_examples = read_lines_clip(opt['data_path'], target_domain)

    # Compute ratios of examples for each category in source data
    source_category_ratios = {category_idx: len(examples_list) for category_idx, examples_list in source_examples.items()}
    source_total_examples = sum(source_category_ratios.values())
    source_category_ratios = {category_idx: c / source_total_examples for category_idx, c in source_category_ratios.items()}

    # Compute ratios of examples for each category in target data
    target_category_ratios = {category_idx: len(examples_list) for category_idx, examples_list in target_examples.items()}
    target_total_examples = sum(target_category_ratios.values())
    target_category_ratios = {category_idx: c / target_total_examples for category_idx, c in target_category_ratios.items()}

    # Build splits - we train and validate on the source domain (Art Painting)
    source_val_split_length = source_total_examples * 0.2 # 20% of the source used for validation
    
    # Build splits - we train and test on the target domain
    target_test_split_length = target_total_examples * 0.2 # 20% of the target used for test

    source_train_examples = []
    target_train_examples = []
    source_val_examples = []
    #target_val_examples = []
    test_examples = []

    for category_idx, examples_list in source_examples.items():
        split_idx_s = round(source_category_ratios[category_idx] * source_val_split_length)
        for i, example in enumerate(examples_list):
            if i < split_idx_s:
                source_val_examples.append([example, category_idx]) # each pair is [path_to_img, class_label]
            else:
                source_train_examples.append([example, category_idx]) # each pair is [path_to_img, class_label]

    for category_idx, examples_list in target_examples.items():
        split_idx_t = round(target_category_ratios[category_idx] * target_test_split_length)
        for i, example in enumerate(examples_list):
            if i < split_idx_t:
                test_examples.append([example, category_idx]) # each pair is [path_to_img, class_label]
            else:
                target_train_examples.append([example, category_idx]) # each pair is [path_to_img, class_label]
    
    # Transforms
    normalize = T.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ResNet18 - ImageNet Normalization

    train_transform = T.Compose([
        T.Resize(256),
        T.RandAugment(3, 15),
        T.CenterCrop(224),
        T.ToTensor(),
        normalize
    ])

    eval_transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        normalize
    ])

    # Dataloaders
    source_train_loader = DataLoader(PACSDatasetClip(source_train_examples, train_transform), batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=True)
    source_val_loader = DataLoader(PACSDatasetClip(source_val_examples, eval_transform), batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=False)
    target_train_loader = DataLoader(PACSDatasetClip(target_train_examples, train_transform), batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=True)
    test_loader = DataLoader(PACSDatasetClip(test_examples, eval_transform), batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=False)

    return source_train_loader, target_train_loader, source_val_loader, test_loader
    #raise NotImplementedError('[TODO] Implement build_splits_clip_disentangle') #TODO
