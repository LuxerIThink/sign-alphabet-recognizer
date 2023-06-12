# Sign alphabet recognizer
*Project created for Selected Topics of Machine Learning subject
on Poznan University of Technology in Poland.*

This project used model training method to recognize sign alphabet.
Training and test datasets are made with 
[Data Gatherer Repo](https://github.com/kamilmlodzikowski/WZUM_2023_DataGatherer).

## Used solutions
- Remove ordinal numbers, handness and world landmarks columns from dataset
- IQR filter with factor = 5
- SVC classifier with linear kernel, C = 250 and gamma = auto