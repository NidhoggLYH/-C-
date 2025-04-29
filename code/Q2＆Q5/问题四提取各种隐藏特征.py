import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis, entropy
from scipy.fft import fft

# 载入数据
file_path = "ori_data/附件三（测试集）.xlsx"
data = pd.read_excel(file_path)

# 提取时域特征
flux_density_data = data.iloc[:, 5:]
mean_values = flux_density_data.mean(axis=1)
std_dev_values = flux_density_data.std(axis=1)
max_values = flux_density_data.max(axis=1)
min_values = flux_density_data.min(axis=1)
peak_to_peak_values = max_values - min_values
rms_values = np.sqrt(np.mean(np.square(flux_density_data), axis=1))
waveform_factor = rms_values / mean_values.abs()
crest_factor = max_values / rms_values
skewness = flux_density_data.apply(skew, axis=1)
kurtosis_values = flux_density_data.apply(kurtosis, axis=1)


# 提取频域特征
def extract_frequency_domain_features(series):
    fft_features = fft(series.to_numpy())  # 确保输入为numpy数组
    magnitude = np.abs(fft_features)

    # 计算频域特征
    fft_max_freq = np.max(magnitude)
    fft_energy = np.sum(magnitude ** 2)
    spectral_centroid = np.sum(magnitude * np.arange(len(magnitude))) / np.sum(magnitude)
    spectral_flatness = 10 * np.log10(entropy(magnitude) / np.log(len(magnitude)))
    spectral_entropy = -np.sum((magnitude / np.sum(magnitude)) * np.log(magnitude / np.sum(magnitude)))

    return pd.Series([fft_max_freq, fft_energy, spectral_centroid, spectral_flatness, spectral_entropy],
                     index=['最大频率', '能量', '频谱中心', '频谱平坦度', '频谱熵'])


freq_features = flux_density_data.apply(extract_frequency_domain_features, axis=1)

# 合并特征
features_df = pd.concat([
    mean_values, std_dev_values, max_values, min_values,
    peak_to_peak_values, rms_values, waveform_factor,
    crest_factor, skewness, kurtosis_values, freq_features
], axis=1)
features_df.columns = [
    '平均值', '标准差', '最大值', '最小值', '峰峰值', '均方根值', '波形因子',
    '峰值因子', '偏度', '峰度', '最大频率', '能量', '频谱中心', '频谱平坦度', '频谱熵'
]

result_data = pd.concat([data.iloc[:, :5], features_df], axis=1)

# 保存为Excel
output_file_path = "data/镜像测试集.xlsx"
result_data.to_excel(output_file_path, index=False)