# LogDP: Combining Dependency and Proximity for Log-based Anomaly Detection
LogDP is a semi-supervised log anomaly detection approach, which utilizes the dependency relationships among log events and proximity among log sequences to detect the anomalies in massive unlabeled log data. 

LogDP divides log events into dependent and independent events, then learns normal patterns of dependent events using dependency and independent events using proximity. Events violating any normal pattern are identified as anomalies. 

By combining dependency and proximity, LogDP is able to achieve high detection accuracy. Extensive experiments have been conducted on real-world datasets, and the results show that LogDP outperforms six state-of-the-art methods.

## Datasets

Three public log datasets, HDFS, BGL and Spirit, are used in ourexperiments, which are available from [LOGPAI](https://github.com/logpai). From the three datasets, we generateseven datasets using different log grouping strategies. The HDFS is generatedusing session, and BGL and Spirit are generated using 1-hour logs, 100 logs, and20 logs windows. For LogDP, the first2/3 sequences of the training set are used for training, and the remaining 1/3sequences are used as a validation set.

## Benchmark Methods

Six state-of-the-art log-based anomaly detection methods  are  selected  as  the  benchmark  methods

### Benchmark methods
#### Proximity-based methods
+ [OneClassSVM](https://direct.mit.edu/neco/article-abstract/13/7/1443/6529)
+ [PCA(SOSP2009)](https://dl.acm.org/doi/abs/10.1145/1629575.1629587)
+ [LogCluster(ICSE-C2016)](https://ieeexplore.ieee.org/abstract/document/7883294)
#### Sequential-based methods
+ [DeepLog(CCS2017)](https://dl.acm.org/doi/abs/10.1145/3133956.3134015)
#### Invariant relation-based methods
+ [InvariantMining(USENIX2010)](https://www.usenix.org/event/atc10/tech/full_papers/Lou.pdf)
+ [ADR(SRDS2020)](https://ieeexplore.ieee.org/abstract/document/9252062)


## Contributing
Please feel free to contribute any kind of functions or enhancements.

## License
This project is licensed under the MIT License. See LICENSE for more details.


## Citation
If you use this code for your research, please cite our paper.
```
@article{xie2021LogDP,
  title={LogDP: Combining Dependency and Proximityfor Log-based Anomaly Detection},
  author={Xie, Yongzheng and Zhang, Hongyu and Zhang, Bo and Babar, Muhammad Ali and Lu, Sha},
  year={2021}
}
```
