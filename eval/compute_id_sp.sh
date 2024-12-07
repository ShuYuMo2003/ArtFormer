# python eval/instantiation_distance_sp.py \
#     --gt_dir eval/testgt \
#     --data_dir eval/testin \
#     --output_dir eval/ID_output \
#     --N_states 10 \
#     --N_pcl 4096 \
#     --world_size 4

python eval/instantiation_distance_sp.py \
    --gt_dir StorageFurniture_128/gt/data_gt \
    --data_dir 6_ours_obj_dats \
    --output_dir output/ours \
    --sample_file_path eval/test_files.json \
    --N_states 10 \
    --N_pcl 4096 \
    --world_size 4