import os
os.environ["OMP_NUM_THREADS"] = "4"

import pandas as pd
import numpy as np
import torch
from functools import partial
import pytorch_lightning as pl
import warnings
import pickle
warnings.filterwarnings("ignore")

from torch.utils.data import DataLoader

from ptls.data_load.datasets import MemoryMapDataset
from ptls.data_load.iterable_processing.iterable_seq_len_limit import ISeqLenLimit
from ptls.data_load.iterable_processing.to_torch_tensor import ToTorch
from ptls.data_load.iterable_processing.feature_filter import FeatureFilter
from ptls.nn import TrxEncoder, RnnSeqEncoder
from ptls.frames.coles import CoLESModule
from ptls.data_load.iterable_processing import SeqLenFilter
from ptls.frames.coles import ColesIterableDataset
from ptls.frames.coles.split_strategy import SampleSlices
from ptls.frames import PtlsDataModule
from ptls.preprocessing import PandasDataPreprocessor
from ptls.data_load.utils import collate_feature_dict
from ptls.data_load.iterable_processing_dataset import IterableProcessingDataset

from tqdm import tqdm
from pandas.tseries.offsets import MonthBegin
from datetime import datetime


def prepare_train_dl(processed_train, processed_test, min_seq_len=64, batch_size=256):
    train = MemoryMapDataset(
        data=processed_train.to_dict("records"),
        i_filters=[
            FeatureFilter(drop_feature_names=['client_id', 'target_1', 'target_2', 'target_3', 'target_4']),
            SeqLenFilter(min_seq_len=min_seq_len),
            ISeqLenLimit(max_seq_len=4096),
            ToTorch()
        ]
    )

    test = MemoryMapDataset(
        data=processed_test.to_dict("records"),
        i_filters=[
            FeatureFilter(drop_feature_names=['client_id', 'target_1', 'target_2', 'target_3', 'target_4']),
            SeqLenFilter(min_seq_len=min_seq_len),
            ISeqLenLimit(max_seq_len=4096),
            ToTorch()
        ]
    )
    train_ds = ColesIterableDataset(
        data=train,
        splitter=SampleSlices(
            split_count=5,
            cnt_min=32,
            cnt_max=180
        )
    )

    valid_ds = ColesIterableDataset(
        data=test,
        splitter=SampleSlices(
            split_count=5,
            cnt_min=32,
            cnt_max=180
        )
    )
    
    train_dl = PtlsDataModule(
        train_data=train_ds,
        train_num_workers=8,
        train_batch_size=batch_size,
        valid_data=valid_ds,
        valid_num_workers=8,
        valid_batch_size=batch_size
    )
    return train_dl


class GetSplit(IterableProcessingDataset):
    def __init__(
        self,
        months,
        col_id='client_id',
        col_time='event_time'
    ):
        super().__init__()
        self.months = months
        self._col_id = col_id
        self._col_time = col_time

    def __iter__(self):
        for rec in self._src:
            for i, month in enumerate(self.months):
                features = rec[0] if type(rec) is tuple else rec
                features = features.copy()

                month_event_time = int((pd.to_datetime(month, yearfirst=True, dayfirst=False) - MonthBegin(1)).to_datetime64()) / 1e9
                mask = features[self._col_time] < month_event_time

                for key, tensor in features.items():
                    if key.startswith('target'):
                        features[key] = tensor[i].tolist()
                    elif key != self._col_id:
                        features[key] = tensor[mask]

                features[self._col_id] += '__' + str(month)

                yield features


def collate_feature_dict_with_target(batch, col_id='client_id', targets=False):
    batch_ids = []
    target_cols = []
    for sample in batch:
        batch_ids.append(sample[col_id])
        del sample[col_id]

        if targets:
            target_cols.append([sample[f'target_{i}'] for i in range(1, 5)])
            del sample['target_1']
            del sample['target_2']
            del sample['target_3']
            del sample['target_4']

    padded_batch = collate_feature_dict(batch)
    if targets:
        return padded_batch, batch_ids, target_cols
    return padded_batch, batch_ids


def to_pandas(x):
    with torch.no_grad():
        expand_cols = []
        scalar_features = {}
        for k, v in x.items():
            if type(v) is torch.Tensor:
                v = v.cpu().detach().numpy()
            if type(v) is list or len(v.shape) == 1:
                scalar_features[k] = v
            elif len(v.shape) == 2:
                expand_cols.append(k)
            else:
                scalar_features[k] = None
        dataframes = [pd.DataFrame(scalar_features)]
        for col in expand_cols:
            v = x[col].cpu().detach().numpy()
            dataframes.append(pd.DataFrame(v, columns=[f'{col}_{i:04d}' for i in range(v.shape[1])]))
        return pd.concat(dataframes, axis=1)
    

def make_prediction(model, inference_dl, device):
    dfs = []
    for x in tqdm(inference_dl):
        x_len = len(x)
        if x_len == 3:
            x, batch_ids, target_cols = x
        else:
            x, batch_ids = x
        out = model(x.to(device))
        if x_len == 3:
            target_cols = torch.tensor(target_cols)
            x_out = {
                'client_id': batch_ids,
                'target_1': target_cols[:, 0],
                'target_2': target_cols[:, 1],
                'target_3': target_cols[:, 2],
                'target_4': target_cols[:, 3],
                'emb': out
            }
        else:
            x_out = {
                'client_id': batch_ids,
                'emb': out
            }
        torch.cuda.empty_cache()
        dfs.append(to_pandas(x_out))
    return pd.concat(dfs, axis='rows')


def get_train_dataset(processed_data, model, months, device):
    train = MemoryMapDataset(
        data=processed_data.to_dict("records"),
        i_filters=[
            ISeqLenLimit(max_seq_len=4096),
            FeatureFilter(keep_feature_names=['client_id', 'target_1', 'target_2', 'target_3', 'target_4']),
            GetSplit(months=months),
            ToTorch(),
        ]
    )

    inference_train_dl = DataLoader(
            dataset=train,
            collate_fn=collate_feature_dict_with_target,
            shuffle=False,
            num_workers=0,
            batch_size=256,
        )
    
    train_emb_df = make_prediction(model, inference_train_dl, device)
    train_emb_df[['client_id', 'month']] = train_emb_df['client_id'].str.split('__', n=1, expand=True)
    return train_emb_df


def get_val_dataset(processed_data, model, months, device):
    val = MemoryMapDataset(
        data=processed_data.to_dict("records"),
        i_filters=[
            ISeqLenLimit(max_seq_len=4096),
            FeatureFilter(keep_feature_names=['client_id', 'target_1', 'target_2', 'target_3', 'target_4']),
            GetSplit(months=months),
            ToTorch(),
        ]
    )
    inference_val_dl = DataLoader(
            dataset=val,
            collate_fn=collate_feature_dict_with_target,
            shuffle=False,
            num_workers=0,
            batch_size=256,
        )
    
    val_emb_df = make_prediction(model, inference_val_dl, device)
    val_emb_df[['client_id', 'month']] = val_emb_df['client_id'].str.split('__', n=1, expand=True)
    return val_emb_df


def get_test_dataset(processed_data, model, device):
    test = MemoryMapDataset(
        data=processed_data.to_dict("records"),
        i_filters=[
            ISeqLenLimit(max_seq_len=4096),
            FeatureFilter(keep_feature_names=['client_id', 'target_1', 'target_2', 'target_3', 'target_4']),
            ToTorch(),
        ]
    )

    inference_test_dl = DataLoader(
            dataset=test,
            collate_fn=collate_feature_dict_with_target,
            shuffle=False,
            num_workers=0,
            batch_size=256,
        )
    
    test_emb_df = make_prediction(model, inference_test_dl, device)
    return test_emb_df