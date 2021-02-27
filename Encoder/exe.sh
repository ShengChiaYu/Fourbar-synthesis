# Classification
# time python3 cnn_main.py --class_num 4 --save_dir 'cnn_models' --net LeNet
# time python3 cnn_main.py --class_num 12 --save_dir 'cnn_models' --net LeNet
# time python3 cnn_main.py --class_num 4 --save_dir 'cnn_models' --net resnet50
# time python3 cnn_main.py --class_num 12 --save_dir 'cnn_models' --net resnet50

# Train to mechanism dimensions
# time python3 cnn_main.py --select_dir 'GCRR' --save_dir 'cnn_models' --net resnet50
# time python3 cnn_main.py --select_dir 'GCRR' --save_dir 'cnn_models' --net resnet152
# time python3 cnn_main.py --select_dir 'GCRR' --save_dir 'cnn_models' --net inception_v3 --img_size 299

# Feature extractor
# time python3 feature_extractor.py --select_dir 'GCRR' --save_dir 'features' --net resnet152

# Image matching
time python3 image_matching.py --select_dir 'GCRR'

# autoencoder training
# time python3 ae_main.py --select_dir 'GCRR' --net autoencoder --bs 32 --lr 5e-4

# Feature extractor
# time python3 feature_extractor.py --select_dir 'GCRR' --save_dir 'features' --net autoencoder
