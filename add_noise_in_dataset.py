import pandas as pd  
import numpy as np  

def add_noise_to_dataset(df, noise_type='gaussian', **kwargs):  
    """  
    为CSV文件中的特征列添加噪声。  

    参数：  
    - input_file: 输入CSV文件路径  
    - output_file: 输出添加噪声后的CSV文件路径  
    - noise_type: 噪声类型，可选 'gaussian', 'uniform', 'salt_pepper'  
    - kwargs: 各种噪声类型的参数，例如标准差、噪声级别等  
    """

    # 确定特征列的范围（排除第一列和最后一列）  
    feature_cols = df.columns[1:-1]  

    # 创建副本以添加噪声  
    df_noisy = df.copy()  

    if noise_type == 'gaussian':  
        noise_level = kwargs.get('noise_level', 0.01)  
        noise = np.random.normal(0, noise_level, df[feature_cols].shape)  
        df_noisy[feature_cols] += noise  
        print(f"添加高斯噪声，标准差={noise_level}")  

    elif noise_type == 'uniform':  
        low = kwargs.get('low', -0.01)  
        high = kwargs.get('high', 0.01)  
        noise = np.random.uniform(low, high, df[feature_cols].shape)  
        df_noisy[feature_cols] += noise  
        print(f"添加均匀噪声，范围=({low}, {high})")  

    elif noise_type == 'salt_pepper':  
        amount = kwargs.get('amount', 0.01)  
        df_noisy[feature_cols] = add_salt_pepper_noise(df[feature_cols], amount)  
        print(f"添加盐碱噪声，比例={amount}")  

    else:  
        raise ValueError("Unsupported noise type. Choose from 'gaussian', 'uniform', 'salt_pepper'.")  

    # 保存添加噪声后的数据到新的CSV文件  
    return df_noisy

def add_salt_pepper_noise(data, amount=0.01):  
    """  
    为DataFrame添加盐碱噪声。  

    参数：  
    - data: 需要添加噪声的DataFrame  
    - amount: 噪声比例  
    """  
    noisy_data = data.copy()  
    num_rows, num_cols = data.shape  
    total_elements = num_rows * num_cols  
    num_salt = np.ceil(amount * total_elements * 0.5).astype(int)  
    num_pepper = np.ceil(amount * total_elements * 0.5).astype(int)  

    for col in data.columns:  
        # 添加盐噪声（设为最大值）  
        salt_indices = (  
            np.random.randint(0, num_rows, num_salt),  
            [col] * num_salt  
        )  
        noisy_data.loc[salt_indices[0], col] = data[col].max()  

        # 添加胡椒噪声（设为最小值）  
        pepper_indices = (  
            np.random.randint(0, num_rows, num_pepper),  
            [col] * num_pepper  
        )  
        noisy_data.loc[pepper_indices[0], col] = data[col].min()  

    return noisy_data  

if __name__ == "__main__":
    
    src_path = 'dataset/ALFA_dataset3/'  
    # 读取数据
    train = pd.read_csv(src_path + 'train.csv')
    test = pd.read_csv(src_path + 'test.csv')
    noise_level=0.1
    
    # 先合并数据集，添加噪声后再重新分开
    data = pd.concat([train, test])

    # 添加高斯噪声
    gau_data = add_noise_to_dataset(data, noise_type='gaussian', noise_level=noise_level)

    # 重新分开数据集
    gau_train = gau_data[:len(train)]
    gau_test = gau_data[len(train):]

    # 保存数据集,文件名包含噪声等级
    gau_train.to_csv(src_path + 'train_gau_{}.csv'.format(noise_level), index=False)
    gau_test.to_csv(src_path + 'test_gau_{}.csv'.format(noise_level), index=False)
    
    # 添加均匀噪声
    uni_data = add_noise_to_dataset(data, noise_type='uniform', noise_level=noise_level)
    # 重新分开数据集
    uni_train = uni_data[:len(train)]
    uni_test = uni_data[len(train):]

    # 保存数据集,文件名包含噪声等级
    uni_train.to_csv(src_path + 'train_uni_{}.csv'.format(noise_level), index=False)
    uni_test.to_csv(src_path + 'test_uni_{}.csv'.format(noise_level), index=False)
    