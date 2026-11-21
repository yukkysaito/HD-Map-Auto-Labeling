"""Reference camera-only BEV segmentation network trained on the exported dataset (docs/DATASET.md).

geometry  BEV grids and the projection of BEV points into the cameras
data      dataset with BEV rotation / mirror augmentation, collate
model     BEVSegNet: frozen foundation ViT + CNN -> multi-height lifting -> BEV U-Net -> 0.1 m detail head
losses    Lovasz-softmax
metrics   prediction (with abstention), IoU and boundary F-score at the label resolution
train     command line training / evaluation
evaluate  scores of a trained model on a dataset (e.g. re-made labels)
video     videos of a trained model on the val scenes
"""
