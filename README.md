# LogDP: Combining Dependency and Proximity for Log-based Anomaly Detection
[LogDP: Combining Dependency and Proximity for Log-based Anomaly Detection] Yongzheng Xie, Hongyu Zhang, Bo Zhang, Muhammad Ali Babar, Sha Lu.<br> 
LogDP is a semi-supervised log anomaly detection approach, which utilizes the dependency relationships among log events and proximity among log sequences to detect the anomalies in massive unlabeled log data. 

LogDP divides log events into dependent and independent events, then learns normal patterns of dependent events using dependency and independent events using proximity. Events violating any normal pattern are identified as anomalies. 

By combining dependency and proximity, LogDP is able to achieve high detection accuracy. Extensive experiments have been conducted on real-world datasets, and the results show that LogDP outperforms six state-of-the-art methods.

## Datasets

Three public log datasets, HDFS, BGL and Spirit, are used in ourexperiments, which are available from [11]. From the three datasets, we generateseven datasets using different log grouping strategies. The HDFS is generatedusing session, and BGL and Spirit are generated using 1-hour logs, 100 logs, and20 logs windows. The names of the datasets of BGL and Spirit are denoted asDataset-Window, e.g., BGL-100logs as shown in Table 1. For LogDP, the first2/3 sequences of the training set are used for training, and the remaining 1/3sequences are used as a validation set.

## Benchmark Methods

Six state-of-the-art log-based anomaly detection meth-ods  are  selected  as  the  benchmark  methods,  including  three  proximity-based

## Contributing
Please feel free to contribute any kind of functions or enhancements.

## License
This project is licensed under the MIT License. See LICENSE for more details.


## Citation
If you use this code for your research, please cite our paper.
```
@article{xie2021LogDP,
  title={LogDP: Combining Dependency and Proximityfor Log-based Anomaly Detection},
  author={Li, Wenbin and Dong, Chuanqi and Tian, Pinzhuo and Qin, Tiexin and Yang, Xuesong and Wang, Ziyi and Huo Jing and Shi, Yinghuan and Wang, Lei and Gao, Yang and Luo, Jiebo Xie, Yongzheng and Zhang, Hongyu and Zhang, Bo and Muhammad Ali Babar and Sha Lu},
  year={2021}
}
```
