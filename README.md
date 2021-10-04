# LogDP: Combining Dependency and Proximity for Log-based Anomaly Detection
LogDP is a semi-supervised log anomaly detection approach, which utilizes the dependency relationships among log events and proximity among log sequences to detect the anomalies in massive unlabeled log data. 
LogDP divides log events into dependent and independent events, then learns normal patterns of dependent events using dependency and independent events using proximity. Events violating any normal pattern are identified as anomalies. 
By combining dependency and proximity, LogDP is able to achieve high detection accuracy. Extensive experiments have been conducted on real-world datasets, and the results show that LogDP outperforms six state-of-the-art methods.
