#!/bin/bash
app_names=("em_proc_a" "em_proc_b")
source ../../sm_test_common/setup.sh "${app_names[*]}"

../../sm_test_common/start_apps.sh "${app_names[*]}"