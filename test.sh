python model_inference.py -config configs/kelp_segs.py \
                            -ckpt kelp-me/prithvi-proper-means/prithvi-proper-means/iter_5000.pth\
                            -input  /home/ziggy/devel/kelp_data/test_satellite \
                            -output /home/ziggy/devel/kelp_data/test_inference \
                            -input_type tif -bands 0 1 2 3 4