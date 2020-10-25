# -*- coding: utf-8 -*-

import pandas as pd
import torch.utils.data as data
import torch


class FaceLandmarksDataset(data.Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmarks_frame = pd.read_csv(csv_file)

    def __len__(self):
        # print len(self.landmarks_frame)
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        print(idx)
        # landmarks = self.landmarks_frame.get_chunk(128).as_matrix().astype('float')
        landmarks = self.landmarks_frame.pure_rtn[idx, -17:-3].as_matrix().astype('float')

        return landmarks


filename = './data/result_store1.csv'
dataset = FaceLandmarksDataset(filename)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
for data in train_loader:
    print(data)
