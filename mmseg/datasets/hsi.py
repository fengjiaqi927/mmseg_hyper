# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class HsiABDataset(BaseSegDataset):
    """Cityscapes dataset.

    The ``img_suffix`` is fixed to '_leftImg8bit.png' and ``seg_map_suffix`` is
    fixed to '_gtFine_labelTrainIds.png' for Cityscapes dataset.
    """
    METAINFO = dict(
        classes=('Background','Surface water','Street','Urban Fabric',
                 'Industrial, commercial and transport','Mine, dump, and construction sites',
                 'Artificial, vegetated areas','Arable Land','Permanent Crops',
                 'Pastures','Forests','Shrub','Open spaces with no vegetation','Inland wetlands'),
        palette=[[0, 0, 0], [244, 35, 232], [70, 70, 70], [102, 102, 156],
                 [190, 153, 153], [153, 153, 153],
                 [220, 220, 0],[107, 142, 35], [152, 251, 152], 
                 [70, 130, 180],[220, 20, 60], [255, 0, 0], [0, 0, 142],[0, 0, 70]])

    
    def __init__(self,
                 img_suffix='.tiff',
                 seg_map_suffix='.tiff',
                 reduce_zero_label=False,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)



    # def __init__(self,
    #              img_suffix='_leftImg8bit.png',
    #              seg_map_suffix='_gtFine_labelTrainIds.png',
    #              **kwargs) -> None:
    #     super().__init__(
    #         img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)
    #     # super().__init__(
    #     #     img_suffix=img_suffix, seg_map_suffix=seg_map_suffix)
    #     dataset_dict = kwargs.get('dataset_info', None)
    #     self.hsi = dataset_dict['hsi']
    #     self.msi = dataset_dict['msi']
    #     self.sar = dataset_dict['sar']
    #     self.label = dataset_dict['label']
        
    # # def __len__(self):
    # #     return len(self.label)
    
    # # def __getitem__(self, index):
    # #     hsi_out = self.hsi[index]
    # #     msi_out = self.msi[index]
    # #     sar_out = self.sar[index]
    # #     label_out = self.label[index]
    # #     return hsi_out, msi_out, sar_out, label_out
