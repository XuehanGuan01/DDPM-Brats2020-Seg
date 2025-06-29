import torch
import os
import nibabel

class BRATSDataset(torch.utils.data.Dataset):
    def __init__(self, directory, test_flag=True):
        '''
        directory is expected to contain folders like:
            BraTS20_Training_001/
            ├── BraTS20_Training_001_flair.nii.gz
            ├── BraTS20_Training_001_seg.nii.gz
            ├── BraTS20_Training_001_t1.nii.gz
            ├── BraTS20_Training_001_t1ce.nii.gz
            └── BraTS20_Training_001_t2.nii.gz
        '''
        super().__init__()
        self.directory = os.path.expanduser(directory)
        self.test_flag = test_flag
        
        if test_flag:
            self.seqtypes = ['t1', 't1ce', 't2', 'flair']
        else:
            self.seqtypes = ['t1', 't1ce', 't2', 'flair', 'seg']
        self.seqtypes_set = set(self.seqtypes)
        self.database = []
        
        print(f"Scanning directory: {self.directory}")
        for root, dirs, files in os.walk(self.directory):
            # 跳过隐藏目录（如 .ipynb_checkpoints）
            if os.path.basename(root).startswith('.'):
                print(f"Skipping hidden directory: {root}")
                continue
            if not dirs:  # 叶目录，包含数据文件
                print(f"Processing directory: {root}")
                files = [f for f in files if f.endswith('.nii.gz')]
                print(f"Found files: {files}")
                datapoint = {}
                for f in files:
                    base_name = os.path.splitext(os.path.splitext(f)[0])[0]  # 去掉 .nii.gz
                    # 提取模态名称（假设文件名以 _t1, _t2 等结尾）
                    parts = base_name.split('_')
                    if len(parts) >= 2 and parts[-1] in self.seqtypes_set:
                        seqtype = parts[-1]
                        datapoint[seqtype] = os.path.join(root, f)
                    else:
                        print(f"Skipping file {f}: unrecognized modality")
                
                if set(datapoint.keys()) == self.seqtypes_set:
                    self.database.append(datapoint)
                    print(f"Valid datapoint: {datapoint.keys()}")
                else:
                    print(f"Skipping incomplete datapoint in {root}: keys are {datapoint.keys()}")

        print(f"Total samples loaded: {len(self.database)}")
        if len(self.database) == 0:
            print(f"Warning: No valid samples found in {self.directory}")
            
    def __getitem__(self, x):
        out = []
        filedict = self.database[x]
        for seqtype in self.seqtypes:
            nib_img = nibabel.load(filedict[seqtype])
            data = torch.tensor(nib_img.get_fdata())
            # 选择中间切片（假设深度轴是最后一个维度）
            mid_slice = data.shape[-1] // 2
            data = data[..., mid_slice]  # 提取 2D 切片，形状 [C, H, W]
            out.append(data)
        out = torch.stack(out)
    
        if self.test_flag:
            image = out[..., 8:-8, 8:-8]  # 裁剪到 (224, 224)
            return image, filedict[self.seqtypes[0]]
        else:
            image = out[:-1, ...]
            label = out[-1, ...][None, ...]
            image = image[..., 8:-8, 8:-8]
            label = label[..., 8:-8, 8:-8]
            label = torch.where(label > 0, 1, 0).float()
            return image, label



    def __len__(self):
        return len(self.database)