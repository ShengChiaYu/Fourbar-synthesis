# time python3 train.py --dataset "data_e" --data_dir "data/GCRR" --save_dir "models/GCRR/model_GCRR_data_e_14560" --o "adam" --net Net_2 --bs 16
# time python3 train.py --dataset "data_1215" --data_dir "data/GCRR" --save_dir "models/GCRR/model_GCRR_data_1215_14560" --o "adam" --net Net_2 --bs 16

# for i in {2..15..1}
#   do
#      time python3 train.py --dataset "data_b_1215" --data_dir "data/GCRR" --save_dir 'models/GCRR' --o "adam" --net Net_2 --bs 16 --n_clst $i
#   done
#
# time python3 test.py --select_dir 'GCRR' --data_dir "data/GCRR" --save_dir "models/GCRR" --net Net_2


# time python3 train.py --dataset "data_1215" --select_dir 'GRRC' --data_dir "data/GRRC" --save_dir "models/GRRC/model_GRRC_data_1215_14560" --o "adam" --net Net_2 --bs 16
# time python3 train.py --dataset "data_e" --select_dir 'GRRC' --data_dir "data/GRRC" --save_dir "models/GRRC/model_GRRC_data_e_14560" --o "adam" --net Net_2 --bs 16
#
# for i in {5..15..1}
#   do
#      time python3 train.py --dataset "data_b_1215"  --select_dir 'GRRC' --data_dir "data/GRRC" --save_dir 'models/GRRC' --o "adam" --net Net_2 --bs 16 --n_clst $i
#   done
#
# time python3 test.py --select_dir 'GRRC' --data_dir "data/GRRC" --save_dir "models/GRRC" --net Net_2


# time python3 train.py --dataset "data_1215" --select_dir 'RRR1' --data_dir "data/RRR1" --save_dir "models/RRR1/model_RRR1_data_1215_12896" --o "adam" --net Net_2 --bs 16
# time python3 train.py --dataset "data_e" --select_dir 'RRR1' --data_dir "data/RRR1" --save_dir "models/RRR1/model_RRR1_data_e_12896" --o "adam" --net Net_2 --bs 16
#
# for i in {2..15..1}
#   do
#      time python3 train.py --dataset "data_b_1215"  --select_dir 'RRR1' --data_dir "data/RRR1" --save_dir 'models/RRR1' --o "adam" --net Net_2 --bs 16 --n_clst $i
#   done
#
# time python3 test.py --select_dir 'RRR1' --data_dir "data/RRR1" --save_dir "models/RRR1" --net Net_2


# time python3 train.py --dataset "data_1215" --select_dir 'GCCC' --data_dir "data/GCCC" --save_dir "models/GCCC/model_GCCC_data_1215_14560" --o "adam" --net Net_2 --bs 16
# time python3 train.py --dataset "data_e" --select_dir 'GCCC' --data_dir "data/GCCC" --save_dir "models/GCCC/model_GCCC_data_e_14560" --o "adam" --net Net_2 --bs 16
#
# for i in {2..15..1}
#   do
#      time python3 train.py --dataset "data_b_1215"  --select_dir 'GCCC' --data_dir "data/GCCC" --save_dir 'models/GCCC' --o "adam" --net Net_2 --bs 16 --n_clst $i
#   done

# time python3 test.py --select_dir 'GCCC' --data_dir "data/GCCC" --save_dir "models/GCCC" --net Net_2


# time python3 train.py --dataset "data_1215" --select_dir 'GRCR' --data_dir "data/GRCR" --save_dir "models/GRCR/model_GRCR_data_1215_14560" --o "adam" --net Net_2 --bs 16
# time python3 train.py --dataset "data_e" --select_dir 'GRCR' --data_dir "data/GRCR" --save_dir "models/GRCR/model_GRCR_data_e_14560" --o "adam" --net Net_2 --bs 16

# for i in {2..15..1}
#   do
#      time python3 train.py --dataset "data_b_1215"  --select_dir 'GRCR' --data_dir "data/GRCR" --save_dir 'models/GRCR' --o "adam" --net Net_2 --bs 16 --n_clst $i
#   done
#
# time python3 test.py --select_dir 'GRCR' --data_dir "data/GRCR" --save_dir "models/GRCR" --net Net_2


# time python3 train.py --dataset "data_1215" --select_dir 'RRR2' --data_dir "data/RRR2" --save_dir "models/RRR2/model_RRR2_data_1215_12896" --o "adam" --net Net_2 --bs 16
# time python3 train.py --dataset "data_e" --select_dir 'RRR2' --data_dir "data/RRR2" --save_dir "models/RRR2/model_RRR2_data_e_12896" --o "adam" --net Net_2 --bs 16
#
#
# time python3 train.py --dataset "data_1215" --select_dir 'RRR3' --data_dir "data/RRR3" --save_dir "models/RRR3/model_RRR3_data_1215_12896" --o "adam" --net Net_2 --bs 16
# time python3 train.py --dataset "data_e" --select_dir 'RRR3' --data_dir "data/RRR3" --save_dir "models/RRR3/model_RRR3_data_e_12896" --o "adam" --net Net_2 --bs 16
#
#
# time python3 train.py --dataset "data_1215" --select_dir 'RRR4' --data_dir "data/RRR4" --save_dir "models/RRR4/model_RRR4_data_1215_12896" --o "adam" --net Net_2 --bs 16
# time python3 train.py --dataset "data_e" --select_dir 'RRR4' --data_dir "data/RRR4" --save_dir "models/RRR4/model_RRR4_data_e_12896" --o "adam" --net Net_2 --bs 16


# for i in {3..6..1}
#   do
#      time python3 train.py --dataset "data_b_1215"  --select_dir 'RRR2' --data_dir "data/RRR2" --save_dir 'models/RRR2' --o "adam" --net Net_2 --bs 16 --n_clst $i
#   done
#
# time python3 test.py --select_dir 'RRR2' --data_dir "data/RRR2" --save_dir "models/RRR2" --net Net_2
#
#
# for i in {2..8..1}
#   do
#      time python3 train.py --dataset "data_b_1215"  --select_dir 'RRR3' --data_dir "data/RRR3" --save_dir 'models/RRR3' --o "adam" --net Net_2 --bs 16 --n_clst $i
#   done
#
# time python3 test.py --select_dir 'RRR3' --data_dir "data/RRR3" --save_dir "models/RRR3" --net Net_2
#
#
# for i in {2..5..1}
#   do
#      time python3 train.py --dataset "data_b_1215"  --select_dir 'RRR4' --data_dir "data/RRR4" --save_dir 'models/RRR4' --o "adam" --net Net_2 --bs 16 --n_clst $i
#   done
#
# time python3 test.py --select_dir 'RRR4' --data_dir "data/RRR4" --save_dir "models/RRR4" --net Net_2


# time python3 train.py --dataset "data_b_1215"  --select_dir 'RRR3' --data_dir "data/RRR3" --save_dir 'models/RRR3' --o "adam" --net Net_2 --bs 16 --n_clst 9
# time python3 train.py --dataset "data_b_1215"  --select_dir 'RRR4' --data_dir "data/RRR4" --save_dir 'models/RRR4' --o "adam" --net Net_2 --bs 16 --n_clst 6
# time python3 train.py --dataset "data_b_1215"  --select_dir 'RRR4' --data_dir "data/RRR4" --save_dir 'models/RRR4' --o "adam" --net Net_2 --bs 16 --n_clst 7


# time python3 test.py --select_dir 'RRR3' --data_dir "data/RRR3" --save_dir "models/RRR3" --net Net_2
# time python3 test.py --select_dir 'RRR4' --data_dir "data/RRR4" --save_dir "models/RRR4" --net Net_2


# time python3 train_cls.py --save_dir "models/model_classifier" --o "adam" --net classifier --bs 16

# time python3 test.py --select_dir 'GCCC' --data_dir "data/GCCC" --save_dir "models/GCCC" --net Net_2
time python3 test.py --select_dir 'GCRR' --data_dir "data/GCRR" --save_dir "models/GCRR" --net Net_2
# time python3 test.py --select_dir 'GRCR' --data_dir "data/GRCR" --save_dir "models/GRCR" --net Net_2
# time python3 test.py --select_dir 'GRRC' --data_dir "data/GRRC" --save_dir "models/GRRC" --net Net_2
#
# time python3 test.py --select_dir 'RRR1' --data_dir "data/RRR1" --save_dir "models/RRR1" --net Net_2
# time python3 test.py --select_dir 'RRR2' --data_dir "data/RRR2" --save_dir "models/RRR2" --net Net_2
# time python3 test.py --select_dir 'RRR3' --data_dir "data/RRR3" --save_dir "models/RRR3" --net Net_2
# time python3 test.py --select_dir 'RRR4' --data_dir "data/RRR4" --save_dir "models/RRR4" --net Net_2
