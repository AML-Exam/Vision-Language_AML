from load_data import CATEGORIES, PACSDatasetBaseline, PACSDatasetTuple, read_lines
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import json

DOMAINS = ['art_painting', 'cartoon', 'photo', 'sketch']

def read_lines_dg(data_path, domain_name, dom_idx = None):
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
            examples[category_idx] = [(image_path, dom_idx)]
        else:
            examples[category_idx].append((image_path, dom_idx))
    return examples

def build_splits_baseline_dg(opt):
    
    target_domain = opt['target_domain']
    source_domains = list(filter(lambda dom : dom != opt['target_domain'], DOMAINS))

    source_examples = {}
    for dom in source_domains:
        tmp_examples = read_lines(opt['data_path'], dom)
        for key in tmp_examples.keys():
            if key in source_examples:
                source_examples[key].extend(tmp_examples[key])
            else:
                source_examples[key] = tmp_examples[key]

    target_examples = read_lines(opt['data_path'], target_domain)

    train_examples = []
    val_examples = []
    test_examples = []

    for category_idx, examples_list in source_examples.items():
        for i, example in enumerate(examples_list):
            if i % 5 != 0:
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

    
def build_splits_domain_disentangle_dg(opt):

    target_domain = opt['target_domain']
    source_domains = list(filter(lambda dom : dom != opt['target_domain'], DOMAINS))

    source_examples = {}
    for dom_idx, dom in enumerate(source_domains):
        tmp_examples = read_lines_dg(opt['data_path'], dom, dom_idx)
        for key in tmp_examples.keys():
            if key in source_examples:
                source_examples[key].extend(tmp_examples[key])
            else:
                source_examples[key] = tmp_examples[key]

    target_examples = read_lines(opt['data_path'], target_domain)

    train_examples = []
    val_examples = []
    test_examples = []

    for category_idx, examples_list in source_examples.items():
        for i, example in enumerate(examples_list):
            if i % 5 != 0:
                train_examples.append([example, category_idx]) # each pair is [(path_to_img, domain_idx), class_label]
            else:
                val_examples.append([example, category_idx]) # each pair is [(path_to_img, domain_idx), class_label]
    
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
    train_loader = DataLoader(PACSDatasetTuple(train_examples, train_transform), batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=True)
    val_loader = DataLoader(PACSDatasetTuple(val_examples, eval_transform), batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=False)
    test_loader = DataLoader(PACSDatasetBaseline(test_examples, eval_transform), batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=False)

    return train_loader, val_loader, test_loader

def build_splits_clip_disentangle_dg(opt):
    return None