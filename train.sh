##!/bin/bash

#python main.py  \
#--config config/nturgbd120-cross-subject/joint.yaml --model model.Hyperformer.Model --work-dir work_dir/ntu120/csub/Hyperformer_joint --device 2 3 --base-lr 2.5e-2 

#python main.py  \
#--config config/nturgbd120-cross-set/joint.yaml --model model.Hyperformer.Model --work-dir work_dir/ntu120/cset/Hyperformer_joint --device 2 3 --base-lr 2.5e-2 

#python main.py  \
#--config config/nturgbd-cross-view/joint.yaml --model model.Hyperformer.Model --work-dir work_dir/ntu60/cview/Hyperformer_joint  --device 0 1 --base-lr 2.5e-2

#python main.py  \
#--config config/nturgbd-cross-subject/joint.yaml --model model.Hyperformer.Model --work-dir work_dir/ntu60/csub/Hyperformer_joint  --device 0 1 --base-lr 2.5e-2

#python main.py  \
#--config config/nturgbd-cross-subject/joint.yaml --model model.Hyperformer.Model --work-dir work_dir/ntu60/csub/Hyperformer_joint  --device 0 3 --base-lr 2.5e-2 --batch-size 128 --num-worker 64

#python main.py  \
#--config config/ucla/joint.yaml --model model.Hyperformer.Model --work-dir work_dir/ucla/Hyperformer_joint  --device 0 1 --base-lr 2.5e-2

#python main.py  \
#--config config/nturgbd-cross-subject/coupled_joint.yaml --model model.coupledformerv2.CoupledFormerV2 --work-dir /data/users/li/work_dir/ntu60/csub/CoupledFormerV2_joint  --device 0 --base-lr 5e-2 --batch-size 128

python main.py  \
--config config/nturgbd-cross-subject/coupled_joint.yaml --model model.coupledformerv2.CoupledFormerV2 --work-dir /data/users/li/work_dir/ntu60/csub/CoupledFormerV2_joint  --device 0 --base-lr 2.5e-2