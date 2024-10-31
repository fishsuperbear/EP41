#!/bin/bash

ulimit -c unlimited
export LD_LIBRARY_PATH=/app/lib:/usr/lib:/svp/lib:../lib:$LD_LIBRARY_PATH
/app/bin/notify_main &
/app/bin/execution_manager &
/app/bin/extwdg &
/app/bin/remote_diag &
/app/bin/system_monitor &