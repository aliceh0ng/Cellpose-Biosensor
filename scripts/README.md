# Scripts to run the whole project pipeline from splitting the training data to downstream analysis!

Order:

1. run `make_splits.py` - this splits the raw data into train/val/test. do once in the beginning
2. run `inference_norm.py` - this normalizes the image, creates nxn patches, runs base cpsam to create initial masks
3. Fix annotation masks created by inference.py, update `annotations.json` with stacks/patches that are finished
4. run `finetune.py` - trains model using train.train_seg(), validates against internal validation patches
5. Run external validation & testing

If happy with results, the model is saved & you want to do downstream analysis of MFI measurements, etc., then run `pipeline.py`