In the previous stage, there is no validation set label, the validation set used is made by ourselves according to the result returned by the submission, and there are some label errors, so the result in the test log is not completely correct.

**On February 15, 2019, we reduced the ACER to 0 on the validation set, and we made the exact correct validation set label**

The performance of the model used for ensemble in the validation set can be viewed in file ./logs/ensemble_model_val_log.txt