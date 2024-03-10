import os, yaml, pickle, shutil, tarfile, glob
import cv2
import albumentations
import PIL
import numpy as np
import torchvision.transforms.functional as TF
from omegaconf import OmegaConf
from functools import partial
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, Subset
import pandas as pd
from taming.data.base import ImagePaths, NumpyPaths
from torchvision import transforms
from sklearn.preprocessing import StandardScaler

import pandas as pd
import torch
from torch.nn.functional import one_hot
from torchvision.datasets.folder import ImageFolder
from torchvision.transforms import Resize, Compose, ToTensor

import taming.data.utils as tdu
from taming.data.imagenet import (
    str_to_indices,
    give_synsets_from_indices,
    download,
    retrieve,
)
from taming.data.imagenet import ImagePaths

from ldm.modules.image_degradation import degradation_fn_bsr, degradation_fn_bsr_light


def synset2idx(path_to_yaml="data/age_index_synset.yaml"):
    with open(path_to_yaml) as f:
        di2s = yaml.load(f)
    return dict((v, k) for k, v in di2s.items())


class AgeNetBase(Dataset):
    def __init__(self, csv_file, data_root, size, keys=None, transform=None):
        super().__init__()
        data_info = pd.read_csv(csv_file)
        self.image_paths = [os.path.join(data_root, img_path) for img_path in data_info['image_path']]
        self.ages = data_info['age'].tolist()
        self.size = size
        self.keys = keys
        self.transform = transform or transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.data = [{'image': img_path, 'age': age} for img_path, age in zip(self.image_paths, self.ages)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        example = self.data[i]
        image = Image.open(example['image']).convert('RGB')
        age = example['age']
        if self.transform:
            image = self.transform(image)
        #print("ÍBS: Image tensor shape in dataset:", image.size()) 
        #print("ÍBS: Image shape in dataset:", image.shape) 
        example = {'image': image, 'age': age}
        if self.keys is not None:
            return {key: example[key] for key in self.keys}
        return example
    
class AgeDBTrain(AgeNetBase):
    def __init__(self, size, keys=None):
        csv_file = "data/AgeDB_train.csv"
        data_root = "data/AgeDB_train"

        self.data = ImagePaths(paths=paths, size=size, random_crop=False, labels=self.labels)

        super().__init__(csv_file=csv_file, data_root=data_root, size=size, keys=keys)

    #def __init__(self, size, keys=None):
    #    super().__init__()
        
    #    root = "data/AgeDB_train"
        
    #    csv_file = "data/AgeDB_train.csv"

    #    data_info = pd.read_csv(csv_file)

    #    image_paths = data_info['image_path'].tolist()
    #    age_labels = data_info['age'].tolist()

    #    paths = [os.path.join(root, img_path) for img_path in image_paths]

     #   self.data = ImagePaths(paths=paths, size=size, random_crop=False, labels=age_labels)
        
      #  self.keys = keys if keys is not None else ['image', 'age']

class AgeDBValidation(AgeNetBase):
    def __init__(self, size, keys=None):
        csv_file = "data/AgeDB_val.csv"
        data_root = "data/AgeDB_val"
        super().__init__(csv_file=csv_file, data_root=data_root, size=size, keys=keys)

    #def __init__(self, size, keys=None):
    #    super().__init__()
        

    #    root = "data/AgeDB_val"
        

    #    csv_file = "data/AgeDB_val.csv"

    #    data_info = pd.read_csv(csv_file)

    #    image_paths = data_info['image_path'].tolist()
    #    age_labels = data_info['age'].tolist()

    #    paths = [os.path.join(root, img_path) for img_path in image_paths]

    #    self.data = ImagePaths(paths=paths, size=size, random_crop=False, labels=age_labels)
        
    #    self.keys = keys if keys is not None else ['image', 'age']

class FacesBase(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.data = None
        self.keys = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        example = self.data[i]
        ex = {}
        if self.keys is not None:
            for k in self.keys:
                ex[k] = example[k]
        else:
            ex = example
        return ex
    

# FROM Marcel(supervisor)
#class AgeTrain(FacesBase):
#    def __init__(self, size, keys=None):
#        super().__init__()
#        root = "/mnt2/PhD-Marcel/ldm-face-manipulation/latent-diffusion/taming-transformers/data/morph_db"
#        meta_data = pd.read_csv(os.path.join(root, "data.csv"))
#        relpaths = meta_data['Name']
#        paths = [os.path.join(root, f"reference/{relpath}") for relpath in relpaths
#                 if relpath.endswith(".png")]
#        create_path = lambda x: os.path.join(root, f"reference-flame-shape-codes/{x}")
#        shape_code_paths = [f"{create_path(relpath).split('.')[0]}.pt" for relpath in relpaths if relpath.endswith(".png")]
#        self.labels = {'shape_codes':  [torch.load(shape_code_path).to('cpu') for shape_code_path in shape_code_paths]}
#        #self.labels = {'shape_codes':  [randn((1, 100)) for _ in range(len(relpaths))]}
#        self.data = ImagePaths(paths=paths, size=size, random_crop=False, labels=self.labels)

class AgeTrainMagV2(FacesBase):
    def __init__(self, size, keys=None):
        super().__init__()
        root = "data"  
        csv_file = "/work3/s212461/data/meta_data_final_train.csv"#"AgeDB_train.csv" 
        image_dir = "/work3/s212461/data/all_data_small_final_train" 
        embeddings_dir = "/work3/s212461/data/face_embeddings_final_train" #ÍBS:03.01.2024

        meta_data = pd.read_csv(os.path.join(root, csv_file))
        relpaths = meta_data['image_path']
        ages = meta_data['age']

        ages = meta_data['age'].astype(int)

        age_indices = ages - 1
        age_one_hot = torch.nn.functional.one_hot(torch.tensor(age_indices), num_classes=101).float()

        age_one_hot_scaled = age_one_hot * 10


        paths = [os.path.join(root, image_dir, f"{relpath}") for relpath in relpaths]

        face_embeddings = [np.load(os.path.join(embeddings_dir, f"{os.path.splitext(os.path.basename(relpath))[0]}.npy")) for relpath in relpaths] #ÍBS:03.01.2024:Added

        scaler = StandardScaler()
        normalized_embeddings = scaler.fit_transform(np.vstack(face_embeddings))

        #combined_labels = [torch.cat((torch.tensor(age_one_hot[i]), torch.tensor(face_embeddings[i])), dim=-1) for i in range(len(relpaths))]
        #combined_labels = [torch.cat((age_one_hot[i], torch.tensor(face_embeddings[i])), dim=0) for i in range(len(age_one_hot))]
        
        combined_labels = torch.zeros((len(age_one_hot_scaled), age_one_hot_scaled.shape[1] + len(normalized_embeddings[0])))

        for i in range(len(age_one_hot_scaled)):
            combined_labels[i] = torch.cat((age_one_hot_scaled[i], torch.tensor(face_embeddings[i])), dim=0)

        #print(combined_labels.shape)

        #print('ÍBS START')
        #print(age_one_hot.shape) 
        #print(face_embeddings[0].shape)
        #print(combined_labels[0].shape)  
        #print('ÍBS END')
        
        self.data = ImagePaths(paths=paths, size=size, random_crop=False, labels={'age': combined_labels})#labels=self.labels)

        self.keys = keys

class AgeValidationMagV2(FacesBase):
    def __init__(self, size, keys=None):
        super().__init__()
        root = "data"  
        csv_file = "/work3/s212461/data/meta_data_final_val.csv"  
        image_dir = "/work3/s212461/data/all_data_small_final_val"  
        embeddings_dir = "/work3/s212461/data/face_embeddings_final_val" #ÍBS:03.01.2024

        meta_data = pd.read_csv(os.path.join(root, csv_file))
        relpaths = meta_data['image_path']
        ages = meta_data['age']

        ages = meta_data['age'].astype(int)

        age_indices = ages - 1
        age_one_hot = torch.nn.functional.one_hot(torch.tensor(age_indices), num_classes=101).float()

        age_one_hot_scaled = age_one_hot * 10

        paths = [os.path.join(root, image_dir, f"{relpath}") for relpath in relpaths]

        #self.labels = {'age': age_one_hot}

        face_embeddings = [np.load(os.path.join(embeddings_dir, f"{os.path.splitext(os.path.basename(relpath))[0]}.npy")) for relpath in relpaths] #ÍBS:03.01.2024:Added

        scaler = StandardScaler()
        normalized_embeddings = scaler.fit_transform(np.vstack(face_embeddings))


        #combined_labels = [torch.cat((torch.tensor(age_one_hot[i]), torch.tensor(face_embeddings[i])), dim=-1) for i in range(len(relpaths))]
        combined_labels = torch.zeros((len(age_one_hot_scaled), age_one_hot_scaled.shape[1] + len(normalized_embeddings[0])))

        for i in range(len(age_one_hot_scaled)):
            combined_labels[i] = torch.cat((age_one_hot_scaled[i], torch.tensor(face_embeddings[i])), dim=0)

        #print(combined_labels.shape)

        #print('ÍBS START')
        #print(age_one_hot.shape) 
        #print(face_embeddings[0].shape) 
        #print(combined_labels[0].shape)  
        #print('ÍBS END')
        
        self.data = ImagePaths(paths=paths, size=size, random_crop=False, labels={'age': combined_labels})#labels=self.labels)

        self.keys = keys 

class AgeTrain(FacesBase):
    def __init__(self, size, keys=None):
        super().__init__()
        root = "data"  
        csv_file = "/work3/s212461/data/meta_data_final_train.csv"#"AgeDB_train.csv" 
        image_dir = "/work3/s212461/data/all_data_small_final_train" 
        embeddings_dir = "/work3/s212461/data/face_embeddings_final_train" #ÍBS:03.01.2024

        meta_data = pd.read_csv(os.path.join(root, csv_file))
        relpaths = meta_data['image_path']
        ages = meta_data['age']

        ages = meta_data['age'].astype(int)

        age_indices = ages - 1
        age_one_hot = torch.nn.functional.one_hot(torch.tensor(age_indices), num_classes=101).float()

        age_one_hot_scaled = age_one_hot

        paths = [os.path.join(root, image_dir, f"{relpath}") for relpath in relpaths]

        #self.labels = {'age': age_one_hot}#ÍBS03.01.2024: Commented out

        face_embeddings = [np.load(os.path.join(embeddings_dir, f"{os.path.splitext(os.path.basename(relpath))[0]}.npy")) for relpath in relpaths] #ÍBS:03.01.2024:Added

        #combined_labels = [torch.cat((torch.tensor(age_one_hot[i]), torch.tensor(face_embeddings[i])), dim=-1) for i in range(len(relpaths))]
        #combined_labels = [torch.cat((age_one_hot[i], torch.tensor(face_embeddings[i])), dim=0) for i in range(len(age_one_hot))]
        
        combined_labels = torch.zeros((len(age_one_hot_scaled), age_one_hot_scaled.shape[1] + len(face_embeddings[0])))

        for i in range(len(age_one_hot_scaled)):
            combined_labels[i] = torch.cat((age_one_hot_scaled[i], torch.tensor(face_embeddings[i])), dim=0)

        #print(combined_labels.shape)

        #print('ÍBS START')
        #print(age_one_hot.shape)
        #print(face_embeddings[0].shape)
        #print(combined_labels[0].shape)  
        #print('ÍBS END')
        
        self.data = ImagePaths(paths=paths, size=size, random_crop=False, labels={'age': combined_labels})#labels=self.labels)

        self.keys = keys

class AgeValidation(FacesBase):
    def __init__(self, size, keys=None):
        super().__init__()
        root = "data"  
        csv_file = "/work3/s212461/data/meta_data_final_val.csv"  
        image_dir = "/work3/s212461/data/all_data_small_final_val"  
        embeddings_dir = "/work3/s212461/data/face_embeddings_final_val" #ÍBS:03.01.2024

        meta_data = pd.read_csv(os.path.join(root, csv_file))
        relpaths = meta_data['image_path']
        ages = meta_data['age']

        ages = meta_data['age'].astype(int)

        age_indices = ages - 1
        age_one_hot = torch.nn.functional.one_hot(torch.tensor(age_indices), num_classes=101).float()

        age_one_hot_scaled = age_one_hot

        paths = [os.path.join(root, image_dir, f"{relpath}") for relpath in relpaths]

        #self.labels = {'age': age_one_hot}

        face_embeddings = [np.load(os.path.join(embeddings_dir, f"{os.path.splitext(os.path.basename(relpath))[0]}.npy")) for relpath in relpaths] #ÍBS:03.01.2024:Added
        #combined_labels = [torch.cat((torch.tensor(age_one_hot[i]), torch.tensor(face_embeddings[i])), dim=-1) for i in range(len(relpaths))]
        combined_labels = torch.zeros((len(age_one_hot_scaled), age_one_hot_scaled.shape[1] + len(face_embeddings[0])))

        for i in range(len(age_one_hot_scaled)):
            combined_labels[i] = torch.cat((age_one_hot_scaled[i], torch.tensor(face_embeddings[i])), dim=0)

        #print(combined_labels.shape)

        #print('ÍBS START')
        #print(age_one_hot.shape) 
        #print(face_embeddings[0].shape) 
        #print(combined_labels[0].shape)  
        #print('ÍBS END')
        
        self.data = ImagePaths(paths=paths, size=size, random_crop=False, labels={'age': combined_labels})#labels=self.labels)

        self.keys = keys 


########################################################################
########################One-Hot Encoder#################################
########################################################################
########################################################################

class AgeTrainTEMP(FacesBase):
    def __init__(self, size, keys=None):
        super().__init__()
        root = "data"  
        csv_file = "/work3/s212461/data/meta_data_final_val.csv"  
        image_dir = "/work3/s212461/data/all_data_small_final_val"  

        meta_data = pd.read_csv(os.path.join(root, csv_file))
        relpaths = meta_data['image_path']
        ages = meta_data['age']

        ages = meta_data['age'].astype(int)

        age_indices = ages - 1
        age_one_hot = torch.nn.functional.one_hot(torch.tensor(age_indices), num_classes=101).float()

        paths = [os.path.join(root, image_dir, f"{relpath}") for relpath in relpaths]
        #self.labels = {'age': age_one_hot}#ÍBS03.01.2024: Commented out

        #face_embeddings = [np.load(os.path.join(embeddings_dir, f"{os.path.splitext(os.path.basename(relpath))[0]}.npy")) for relpath in relpaths] #ÍBS:03.01.2024:Added

        #combined_labels = [torch.cat((torch.tensor(age_one_hot[i]), torch.tensor(face_embeddings[i])), dim=-1) for i in range(len(relpaths))]
        #combined_labels = [torch.cat((age_one_hot[i], torch.tensor(face_embeddings[i])), dim=0) for i in range(len(age_one_hot))]
        
        #combined_labels = torch.zeros((len(age_one_hot), age_one_hot.shape[1] + len(face_embeddings[0])))

        #for i in range(len(age_one_hot)):
        #    combined_labels[i] = torch.cat((age_one_hot[i], torch.tensor(face_embeddings[i])), dim=0)

        #print(combined_labels.shape)

        #print('ÍBS START')
        #print(age_one_hot.shape)  
        #print(face_embeddings[0].shape)
        #print(combined_labels[0].shape)  
        #print('ÍBS END')
        
        self.data = ImagePaths(paths=paths, size=size, random_crop=False, labels={'age': age_one_hot})#labels=self.labels)

        self.keys = keys 

class AgeValidationTEMP(FacesBase):
    def __init__(self, size, keys=None):
        super().__init__()
        root = "data"  
        csv_file = "/work3/s212461/data/meta_data_final_train.csv"#"AgeDB_train.csv" 
        image_dir = "/work3/s212461/data/all_data_small_final_train" 
        #embeddings_dir = "/work3/s212461/data/face_embeddings_final_train" #ÍBS:03.01.2024

        meta_data = pd.read_csv(os.path.join(root, csv_file))
        relpaths = meta_data['image_path']
        ages = meta_data['age']

        ages = meta_data['age'].astype(int)

        age_indices = ages - 1
        age_one_hot = torch.nn.functional.one_hot(torch.tensor(age_indices), num_classes=101).float()

        paths = [os.path.join(root, image_dir, f"{relpath}") for relpath in relpaths]
        #self.labels = {'age': age_one_hot}#ÍBS03.01.2024: Commented out

        #face_embeddings = [np.load(os.path.join(embeddings_dir, f"{os.path.splitext(os.path.basename(relpath))[0]}.npy")) for relpath in relpaths] #ÍBS:03.01.2024:Added

        #combined_labels = [torch.cat((torch.tensor(age_one_hot[i]), torch.tensor(face_embeddings[i])), dim=-1) for i in range(len(relpaths))]
        #combined_labels = [torch.cat((age_one_hot[i], torch.tensor(face_embeddings[i])), dim=0) for i in range(len(age_one_hot))]
        
        #combined_labels = torch.zeros((len(age_one_hot), age_one_hot.shape[1] + len(face_embeddings[0])))

        #for i in range(len(age_one_hot)):
        #    combined_labels[i] = torch.cat((age_one_hot[i], torch.tensor(face_embeddings[i])), dim=0)

        #print(combined_labels.shape)

        #print('ÍBS START')
        #print(age_one_hot.shape)
        #print(face_embeddings[0].shape)
        #print(combined_labels[0].shape)  
        #print('ÍBS END')
        
        self.data = ImagePaths(paths=paths, size=size, random_crop=False, labels={'age': age_one_hot})#labels=self.labels)

        self.keys = keys


######################################################################
####################### DRAFTS #######################################
######################################################################

#class AgeNetBase(Dataset):
#    def __init__(self, config=None):
#        self.config = config or OmegaConf.create()
#        if not type(self.config) == dict:
#            self.config = OmegaConf.to_container(self.config)
#        self.keep_orig_class_label = self.config.get("keep_orig_class_label", False)
#        self.process_images = True 
#        self._prepare()
        # self._prepare_synset_to_human()
        # self._prepare_idx_to_synset()
#        self._prepare_human_to_integer_label()
#        self._load()

#    def __len__(self):
#        return len(self.data)

#    def __getitem__(self, i):
#        return self.data[i]

#    def _prepare(self):
#        raise NotImplementedError()

#    def _filter_relpaths(self, relpaths):
#        ignore = set(
#            [
#                "n06596364_9591.JPEG",
#            ]
#        )
#        relpaths = [rpath for rpath in relpaths if not rpath.split("/")[-1] in ignore]
#        if "sub_indices" in self.config:
#            indices = str_to_indices(self.config["sub_indices"])
#            synsets = give_synsets_from_indices(
#                indices, path_to_yaml=self.idx2syn
#            ) 
#            self.synset2idx = synset2idx(path_to_yaml=self.idx2syn)
#            files = []
#            for rpath in relpaths:
#                syn = rpath.split("/")[0]
#                if syn in synsets:
#                    files.append(rpath)
#            return files
#        else:
#            return relpaths

    # def _prepare_synset_to_human(self):
    #    SIZE = 2655750
    #    URL = "https://heibox.uni-heidelberg.de/f/9f28e956cd304264bb82/?dl=1"
    #    self.human_dict = os.path.join(self.root, "synset_human.txt")
    #    if (
    #        not os.path.exists(self.human_dict)
    #        or not os.path.getsize(self.human_dict) == SIZE
    #    ):
    #        download(URL, self.human_dict)

    # ÍBS: I created this file myself so it's ready for now
    # def _prepare_idx_to_synset(self):
    #    URL = "https://heibox.uni-heidelberg.de/f/d835d5b6ceda4d3aa910/?dl=1"
    #    self.idx2syn = os.path.join(self.root, "index_synset.yaml")
    #    if not os.path.exists(self.idx2syn):
    #        download(URL, self.idx2syn)

    # ÍBS: Don't need this
    # def _prepare_human_to_integer_label(self):
    #    URL = "https://heibox.uni-heidelberg.de/f/2362b797d5be43b883f6/?dl=1"
    #    self.human2integer = os.path.join(
    #        self.root, "imagenet1000_clsidx_to_labels.txt"
    #    )
    #    if not os.path.exists(self.human2integer):
    #        download(URL, self.human2integer)
    #    with open(self.human2integer, "r") as f:
    #        lines = f.read().splitlines()
    #        assert len(lines) == 1000
    #        self.human2integer_dict = dict()
    #        for line in lines:
    #            value, key = line.split(":")
    #            self.human2integer_dict[key] = int(value)

#    def _load(self):
#        with open(self.txt_filelist, "r") as f:
#            self.relpaths = f.read().splitlines()
#            l1 = len(self.relpaths)
#            self.relpaths = self._filter_relpaths(self.relpaths)
#            print(
#                "Removed {} files from filelist during filtering.".format(
#                    l1 - len(self.relpaths)
#                )
#            )

#        self.synsets = [p.split(os.sep)[0] for p in self.relpaths]
#        self.abspaths = [os.path.join(self.datadir, p) for p in self.relpaths]

#        unique_synsets = np.unique(self.synsets)
#        class_dict = dict((synset, i) for i, synset in enumerate(unique_synsets))
#        if not self.keep_orig_class_label:
#            self.class_labels = [class_dict[s] for s in self.synsets]
#        else:
#            self.class_labels = [self.synset2idx[s] for s in self.synsets]#

#       with open(self.human_dict, "r") as f:
#            human_dict = f.read().splitlines()
#            human_dict = dict(line.split(maxsplit=1) for line in human_dict)

#        self.human_labels = [human_dict[s] for s in self.synsets]
#
#        labels = {
#            "relpath": np.array(self.relpaths),
#            "synsets": np.array(self.synsets),
#            "class_label": np.array(self.class_labels),
#            "human_label": np.array(self.human_labels),
#        }

#        if self.process_images:
#            self.size = retrieve(self.config, "size", default=256)
#            self.data = ImagePaths(
#                self.abspaths,
#                labels=labels,
#                size=self.size,
#                random_crop=self.random_crop,
#            )
#        else:
#            self.data = self.abspaths
#class AgeNetTrain(AgeNetBase):
#    NAME = "ILSVRC2012_train"
#    URL = "http://www.image-net.org/challenges/LSVRC/2012/"
#    AT_HASH = "a306397ccf9c2ead27155983c254227c0fd938e2"
#    FILES = [
#        "ILSVRC2012_img_train.tar",
#    ]
#    SIZES = [
#        147897477120,
#    ]

#    def __init__(self, process_images=True, data_root=None, **kwargs):
#        self.process_images = process_images
#        self.data_root = data_root
#        super().__init__(**kwargs)#

#    def _prepare(self):
#        if self.data_root:
#            self.root = os.path.join(self.data_root, self.NAME)
#        else:
#            cachedir = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
#            self.root = os.path.join(cachedir, "autoencoders/data", self.NAME)

#        self.datadir = os.path.join(self.root, "data")
#        self.txt_filelist = os.path.join(self.root, "filelist.txt")
#        self.expected_length = 1281167
#        self.random_crop = retrieve(
#            self.config, "ImageNetTrain/random_crop", default=True
#        )
#        if not tdu.is_prepared(self.root):
#            # prep
#            print("Preparing dataset {} in {}".format(self.NAME, self.root))

#            datadir = self.datadir
#            if not os.path.exists(datadir):
#                path = os.path.join(self.root, self.FILES[0])
#                if (
#                    not os.path.exists(path)
#                    or not os.path.getsize(path) == self.SIZES[0]
#                ):
#                    import academictorrents as at

#                    atpath = at.get(self.AT_HASH, datastore=self.root)
#                    assert atpath == path

#                print("Extracting {} to {}".format(path, datadir))
#                os.makedirs(datadir, exist_ok=True)
#                with tarfile.open(path, "r:") as tar:
#                    tar.extractall(path=datadir)

#                print("Extracting sub-tars.")
#                subpaths = sorted(glob.glob(os.path.join(datadir, "*.tar")))
#                for subpath in tqdm(subpaths):
#                    subdir = subpath[: -len(".tar")]
#                    os.makedirs(subdir, exist_ok=True)
#                    with tarfile.open(subpath, "r:") as tar:
#                        tar.extractall(path=subdir)

#            filelist = glob.glob(os.path.join(datadir, "**", "*.JPEG"))
#            filelist = [os.path.relpath(p, start=datadir) for p in filelist]
#            filelist = sorted(filelist)
#            filelist = "\n".join(filelist) + "\n"
#            with open(self.txt_filelist, "w") as f:
#                f.write(filelist)

#            tdu.mark_prepared(self.root)

#class ImageNetValidation(ImageNetBase):
#    NAME = "ILSVRC2012_validation"
#    URL = "http://www.image-net.org/challenges/LSVRC/2012/"
#    AT_HASH = "5d6d0df7ed81efd49ca99ea4737e0ae5e3a5f2e5"
#    VS_URL = "https://heibox.uni-heidelberg.de/f/3e0f6e9c624e45f2bd73/?dl=1"
#    FILES = [
#        "ILSVRC2012_img_val.tar",
#        "validation_synset.txt",
#    ]
#    SIZES = [
#        6744924160,
#        1950000,
#    ]
#
 #   def __init__(self, process_images=True, data_root=None, **kwargs):
 #       self.data_root = data_root
 #       self.process_images = process_images
 #       super().__init__(**kwargs)

 #   def _prepare(self):
  #      if self.data_root:
  #          self.root = os.path.join(self.data_root, self.NAME)
  #      else:
  #          cachedir = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
  #          self.root = os.path.join(cachedir, "autoencoders/data", self.NAME)
   #     self.datadir = os.path.join(self.root, "data")
   #     self.txt_filelist = os.path.join(self.root, "filelist.txt")
   #     self.expected_length = 50000
   #     self.random_crop = retrieve(
 #           self.config, "ImageNetValidation/random_crop", default=False
  #      )
  #      if not tdu.is_prepared(self.root):
  #          # prep
  #          print("Preparing dataset {} in {}".format(self.NAME, self.root))

 #           datadir = self.datadir
  #          if not os.path.exists(datadir):
  #              path = os.path.join(self.root, self.FILES[0])
  #              if (
  #                  not os.path.exists(path)
  #                  or not os.path.getsize(path) == self.SIZES[0]
   #             ):
   #                 import academictorrents as at
#
   #                 atpath = at.get(self.AT_HASH, datastore=self.root)
   #                 assert atpath == path
#
    #            print("Extracting {} to {}".format(path, datadir))
 #               os.makedirs(datadir, exist_ok=True)
    #            with tarfile.open(path, "r:") as tar:
#                    tar.extractall(path=datadir)
#
#                vspath = os.path.join(self.root, self.FILES[1])
#                if (
#                    not os.path.exists(vspath)
#                    or not os.path.getsize(vspath) == self.SIZES[1]
#                ):
#                    download(self.VS_URL, vspath)
#
#                with open(vspath, "r") as f:
#                    synset_dict = f.read().splitlines()
#                    synset_dict = dict(line.split() for line in synset_dict)
#
#                print("Reorganizing into synset folders")
#                synsets = np.unique(list(synset_dict.values()))
#                for s in synsets:
#                    os.makedirs(os.path.join(datadir, s), exist_ok=True)
#                for k, v in synset_dict.items():
#                    src = os.path.join(datadir, k)
#                    dst = os.path.join(datadir, v)
#                    shutil.move(src, dst)
#
#            filelist = glob.glob(os.path.join(datadir, "**", "*.JPEG"))
#            filelist = [os.path.relpath(p, start=datadir) for p in filelist]
#            filelist = sorted(filelist)
#            filelist = "\n".join(filelist) + "\n"
#            with open(self.txt_filelist, "w") as f:
#                f.write(filelist)
#
#            tdu.mark_prepared(self.root)
#
#
#class ImageNetSR(Dataset):
#    def __init__(
#        self,
#        size=None,
#        degradation=None,
#        downscale_f=4,
#        min_crop_f=0.5,
#        max_crop_f=1.0,
#        random_crop=True,
#    ):
#        """
#        Imagenet Superresolution Dataloader
#        Performs following ops in order:
#        1.  crops a crop of size s from image either as random or center crop
#        2.  resizes crop to size with cv2.area_interpolation
#        3.  degrades resized crop with degradation_fn
#
#        :param size: resizing to size after cropping
#        :param degradation: degradation_fn, e.g. cv_bicubic or bsrgan_light
#        :param downscale_f: Low Resolution Downsample factor
#        :param min_crop_f: determines crop size s,
#          where s = c * min_img_side_len with c sampled from interval (min_crop_f, max_crop_f)
#        :param max_crop_f: ""
#        :param data_root:
#        :param random_crop:
#        """
#        self.base = self.get_base()
#        assert size
#        assert (size / downscale_f).is_integer()
#        self.size = size
#        self.LR_size = int(size / downscale_f)
#        self.min_crop_f = min_crop_f
#        self.max_crop_f = max_crop_f
#        assert max_crop_f <= 1.0
#        self.center_crop = not random_crop
#
#        self.image_rescaler = albumentations.SmallestMaxSize(
#            max_size=size, interpolation=cv2.INTER_AREA
#        )
#
#        self.pil_interpolation = (
#            False  # gets reset later if incase interp_op is from pillow
#        )
#
#        if degradation == "bsrgan":
#            self.degradation_process = partial(degradation_fn_bsr, sf=downscale_f)
#
#        elif degradation == "bsrgan_light":
#            self.degradation_process = partial(degradation_fn_bsr_light, sf=downscale_f)
#
#        else:
#            interpolation_fn = {
#                "cv_nearest": cv2.INTER_NEAREST,
#                "cv_bilinear": cv2.INTER_LINEAR,
#                "cv_bicubic": cv2.INTER_CUBIC,
#                "cv_area": cv2.INTER_AREA,
#                "cv_lanczos": cv2.INTER_LANCZOS4,
#                "pil_nearest": PIL.Image.NEAREST,
#                "pil_bilinear": PIL.Image.BILINEAR,
#                "pil_bicubic": PIL.Image.BICUBIC,
#                "pil_box": PIL.Image.BOX,
#                "pil_hamming": PIL.Image.HAMMING,
#                "pil_lanczos": PIL.Image.LANCZOS,
#            }[degradation]
#
#            self.pil_interpolation = degradation.startswith("pil_")
#
#            if self.pil_interpolation:
#                self.degradation_process = partial(
#                    TF.resize, size=self.LR_size, interpolation=interpolation_fn
#                )
#
#            else:
#                self.degradation_process = albumentations.SmallestMaxSize(
#                    max_size=self.LR_size, interpolation=interpolation_fn
#                )
#
#    def __len__(self):
#        return len(self.base)
#
#    def __getitem__(self, i):
#        example = self.base[i]
#        image = Image.open(example["file_path_"])
#
#        if not image.mode == "RGB":
#            image = image.convert("RGB")
#
#        image = np.array(image).astype(np.uint8)
#
#        min_side_len = min(image.shape[:2])
#        crop_side_len = min_side_len * np.random.uniform(
#            self.min_crop_f, self.max_crop_f, size=None
#        )
#        crop_side_len = int(crop_side_len)
#
#        if self.center_crop:
#            self.cropper = albumentations.CenterCrop(
#                height=crop_side_len, width=crop_side_len
#            )
#
#        else:
#            self.cropper = albumentations.RandomCrop(
#                height=crop_side_len, width=crop_side_len
#            )
#
#        image = self.cropper(image=image)["image"]
#        image = self.image_rescaler(image=image)["image"]
#
#        if self.pil_interpolation:
#            image_pil = PIL.Image.fromarray(image)
#            LR_image = self.degradation_process(image_pil)
#            LR_image = np.array(LR_image).astype(np.uint8)
#
#        else:
#            LR_image = self.degradation_process(image=image)["image"]
#
#        example["image"] = (image / 127.5 - 1.0).astype(np.float32)
#        example["LR_image"] = (LR_image / 127.5 - 1.0).astype(np.float32)
#
#        return example
#
#
#class ImageNetSRTrain(ImageNetSR):
#    def __init__(self, **kwargs):
#        super().__init__(**kwargs)
#
#    def get_base(self):
#        with open("data/imagenet_train_hr_indices.p", "rb") as f:
#            indices = pickle.load(f)
#        dset = ImageNetTrain(
#            process_images=False,
#        )
#        return Subset(dset, indices)
#
#
#class ImageNetSRValidation(ImageNetSR):
#    def __init__(self, **kwargs):
#        super().__init__(**kwargs)
#
#    def get_base(self):
#        with open("data/imagenet_val_hr_indices.p", "rb") as f:
#            indices = pickle.load(f)
#        dset = ImageNetValidation(
#            process_images=False,
#        )
#        return Subset(dset, indices)