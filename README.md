# 原文说明

该代码为文章《Attention-Based Multi-Scale Feature Fusion for Enhanced Birdsong Recognition》的实现。提出了一种新的鸟鸣识别方法AMDG-NET,它将高效网络V2作为主干网络，结合了基于注意力的特征融合、多尺度特征提取和门禁分类头。我们的模型通过动态聚焦关键特征通道和有效捕捉多尺度特征来增强表示能力。对自建的38种鸟类数据集进行的实验，其准确率为98.17%。该模型在北京鸟类数据集和UrbanSound8K数据集上的准确率分别为97.56%和95.88%。

为了方便实验，本实验对鸟鸣片段进行特征提取后的特征进行存储，在后续调试方法和参数时加载就行，不需要每次实验都进行特征提取，但是这个.apk文件很大无法上传到Github上。

本来想把训练好的.pth模型放上来，但是超过100mb了也不让上传，后续看看网盘可不可以上传，如果可以上传我会把链接放这里。

# 数据说明
本实验使用了三个数据集，其中北京百鸟集与UrbanSound8K泛化数据集为公共数据集。
北京百鸟集下载地址为：https://data.baai.ac.cn/datadetail

UrbanSound8K泛化数据集下载地址为：https://zenodo.org/records/401395

自建数据集所使用的鸟声数据均来自于Xeno-Canto网站，地址为：https://xeno-canto.org

由于数据过大无法上传至Github，如有需要联系作者:zh1324520575@gmail.com。

