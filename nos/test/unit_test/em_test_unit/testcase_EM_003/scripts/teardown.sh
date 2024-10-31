#!/bin/bash
app_names=("em_proc_a" "em_proc_b" "em_proc_c" "em_proc_d" "em_proc_e" \
           "em_proc_f" "em_proc_g" "em_proc_h" "em_proc_i" "em_proc_j" \
           "em_proc_k" "em_proc_l" "em_proc_m" "em_proc_n"  "em_proc_o")

../../em_test_common/teardown.sh "${app_names[*]}"