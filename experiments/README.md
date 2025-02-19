## Experiments folder
This folder contains the job files used to run the experiments, and has the following structure:

```
experiments
├──distance
|    └── contains files and code for the Sinkhorn distance analysis
├── evaluation
|    └── contains files to report the results of the models on the test set
├── fairface_dataset
|    └── contains files to report the results on the fairface dataset
├── finetuning_fairclip_jobs
|    └── contains files to finetune clip models with the fairclip objective
├── lambdatest
|    └── contains file to test the different fairness weights
└── linear_probing
     └── contains files used to train the linear probes
```