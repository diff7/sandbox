import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms

# from torchvision import transforms
import torchvision.transforms.functional as F


def get_low_variance_objects(frame, th=None):
    if th is None:
        th = frame.shape[0] * 0.01

    low_var = set(frame.T[frame.nunique() < th].index.to_list())
    objects = set(frame.select_dtypes(include='object').columns.to_list())
    print(f'Low variance items all: {len(low_var)}, Objects all: {len(objects)}')
    low_var_objects = low_var.intersection(objects)
    high_var_objects  = objects.difference(low_var)
    print(f'Low variance objects: {len(low_var_objects)}, High var objects: {len(high_var_objects)}')
    return low_var_objects, high_var_objects,  low_var.difference(objects)


def get_feature_sizes(frame, th=None, params):
    low_var_objects, high_var_objects,  low_var_real = get_low_variance_objects(frame, th=None)
    categoical = low_var_objects.union(low_var_real)
    low_var_n_objects = categoical.union(high_var_objects)
    real_features = set(frame.columns).difference(low_var_n_objects)
    real_features_size = len(real_features)
    cat_embed_sizes = dict()
    input_concat_vector_size = real_features_size

    for cat in categoical:
        cat_embed_sizes[cat_name] = int(frame[cat].nunique()*params['embed_scaler'])
        input_concat_vector_size += cat_embed_sizes[cat_name]
    params.update({'real_features_size':real_features_size,
            'cat_embed_sizes':cat_embed_sizes,
            'real_features':real_features,
            'input_concat_vector_size':input_concat_vector_size})
    return params


class PandasCatLoader(Dataset):
    """ Input data: list of dicts path_orig, path_mask
        Returns: tensor, label, age """

    def __init__(self, data, params):
        self.data = data
        self.params = params

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        real_vector = self.data.loc[idx, params['real_features']].values
        real_vector = torch.tensor(real_vector, dtype = torch.float64)
        cat_vector = dict()
        for name in params['cat_embed_sizes']:
            cat_vec = self.data.loc[idx, params['real_features']].values
            cat_vec = torch.tensor(cat_vec, dtype = torch.float64)
            cat_vector[name] = cat_vec
        target =  self.data.loc[idx, params['TARGET']].values
        target = torch.tensor(target, dtype = torch.float64)

        return {'real_vector':real_vector,
                'cat_vector':cat_vector,
                'target':target}
