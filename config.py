# config.py  

import torch  
import numpy as np  

class Config:  
    def __post_init__(self):  
        # 确定是否使用 GPU  
        self.use_gpu = self.use_gpu and torch.cuda.is_available()  

        if self.use_gpu and self.use_multi_gpu:  
            self.devices = self.devices.replace(' ', '')  
            device_ids = self.devices.split(',')  
            self.device_ids = [int(id_) for id_ in device_ids]  
            self.gpu = self.device_ids[0]  
        else:  
            self.device_ids = [self.gpu]  

        # 根据 layer_nums 重塑 patch_size_list  
        if len(self.patch_size_list) != self.layer_nums * 4:  
            raise ValueError(f"预期 patch_size_list 有 {self.layer_nums * 4} 个元素，但实际有 {len(self.patch_size_list)} 个")  
        self.patch_size_list = np.array(self.patch_size_list).reshape(self.layer_nums, -1).tolist()  

    def __init__(self):  
        # 基本配置  
        self.is_training = 1  
        self.model = 'MTCL'  # 选项: ['MTCL']  
        self.model_id = "test"  

        # 数据加载器  
        self.data = 'ALFA_ad'  # 数据集类型  
        self.root_path = './dataset/ALFA_dataset1/'  # 数据文件根路径  
        self.data_path = 'ETTh1.csv'  # 数据文件  
        self.loss_save_name = 'test.csv'  # 保存路径  
        self.features = 'M'  # 选项: ['M', 'S']  
        self.target = 'OT'  # S 或 MS 任务中的目标特征  
        self.freq = 's'  # 时间特征编码频率  
        self.checkpoints = './checkpoints/'  # 模型检查点位置  

        # 预测任务  
        self.seq_len = 96  # 输入序列长度  
        self.pred_len = 96  # 预测序列长度  
        self.individual = False  # DLinear：为每个变量（通道）单独使用线性层  

        # 模型参数  
        self.d_model = 16  
        self.d_ff = 16  
        self.num_nodes = 18  
        self.layer_nums = 3  
        self.k = 4  # 每一层选择的 Top K patch 大小  
        self.num_experts_list = [4, 4, 4]  
        self.patch_size_list = [16, 12, 8, 32, 12, 8, 6, 4, 8, 6, 4, 2]  
        self.do_predict = False  # 是否预测未见的未来数据  
        self.revin = 1  # 是否应用 RevIN  
        self.drop = 0.1  # dropout 比例  
        self.embed = 'timeF'  # 时间特征编码，选项: ['timeF', 'fixed', 'learned']  
        self.residual_connection = 1  
        self.metric = 'mse'  
        self.batch_norm = 1  
        self.temp = 200  
        self.lambda_contrastive = 0.1  
        self.no_inter_atten = 0  # 0: 选择 inter_atten, 1: 放弃 inter_atten  
        self.no_intra_atten = 0  # 0: 选择 intra_atten, 1: 放弃 intra_atten  
        self.no_contrastive = 0  # 0: 添加对比损失, 1: 放弃对比损失  

        # 异常检测任务  
        self.anomaly_ratio = 5.0  # 先验异常比率 (%)  

        # 优化  
        self.num_workers = 10  # 数据加载器工作线程数  
        self.itr = 1  # 实验次数  
        self.train_epochs = 1  # 训练轮数  
        self.batch_size = 128  # 训练输入数据的 batch 大小  
        self.patience = 30  # 早停耐心值  
        self.learning_rate = 0.001  # 优化器学习率  
        self.lradj = 'TST'  # 调整学习率  
        self.use_amp = False  # 使用自动混合精度训练  
        self.pct_start = 0.4  # pct_start  

        # GPU 设置  
        self.use_gpu = True  # 使用 GPU  
        self.gpu = 0  # GPU 设备 ID  
        self.use_multi_gpu = False  # 使用多 GPU  
        self.devices = '0'  # 多 GPU 的设备 ID  
        self.test_flop = False  # 使用查看 utils/tools  

        # 后处理  
        self.__post_init__()  

    def __repr__(self):  
        return f"{self.__dict__}"  

# 实例化配置  
config = Config()