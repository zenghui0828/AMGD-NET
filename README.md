# 原文说明

该代码为文章《Attention-Based Multi-Scale Feature Fusion for Enhanced Birdsong Recognition》的实现。提出了一种新的鸟鸣识别方法AMDG-NET,它将高效网络V2作为主干网络，结合了基于注意力的特征融合、多尺度特征提取和门禁分类头。我们的模型通过动态聚焦关键特征通道和有效捕捉多尺度特征来增强表示能力。对自建的38种鸟类数据集进行的实验，其准确率为98.17%。该模型在北京鸟类数据集和UrbanSound8K数据集上的准确率分别为97.56%和95.88%。

# 数据说明
本实验使用了三个数据集，其中北京百鸟集与UrbanSound8K泛化数据集为公共数据集。
北京百鸟集下载地址为：https://data.baai.ac.cn/datadetail

UrbanSound8K泛化数据集下载地址为：https://zenodo.org/records/401395

自建数据集所使用的鸟声数据均来自于Xeno-Canto网站，地址为：https://xeno-canto.org

由于数据过大无法上传至Github，如有需要联系作者。

