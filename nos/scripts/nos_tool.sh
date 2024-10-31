#!/bin/bash

exeparam=${@:2}
execommd=$@
commands='bag dbg log topic devm stmm fm cfg tsync perfhelper easy_package'

usage(){
        echo "Nos Tools Usage:"
        echo "          Toolname: nos , Commands: dbg|bag|topic|log|devm|stmm|easy_package}"
        echo "          { dbg },   -- EM & SM Debug Tool"
        echo "          { bag },   -- Data Record Playback Conversion Tool"
        echo "          { topic }, -- CM Topic Query Tool"
        echo "          { log },   -- Log Configuration Setup Tool"
        echo "          { devm },  -- Devm get devices data Tool"
        echo "          { stmm },  -- System Monitor Tool"
        echo "          { fm },    -- Fm Fault Process Tool"
        echo "          { cfg },   -- configserver Tool"
        echo "          { tsync }, -- time getter/setter Tool"
        echo "          { perfhelper }, -- perf + flamegraph combined tool"
        echo "          { easy_package }, -- Collect files(logs,version,fm,etc) and compress"
}

svcname=$1
if [ x"$1" = x ]; then
    usage
    exit 1
fi

isCommand=$(echo $commands | grep "$1")
if [[ "$isCommand" = "" ]]; then
    echo 'invalid command'
    echo 'usage: nos dbg | bag | topic | log | devm | stmm | fm | cfg | tsync | perfhelper | easy_package'
    exit 0
fi

case "$1" in
    dbg )
        smdbg $exeparam
        ;;
    bag | topic )
        $execommd
        ;;
    log )
        log $exeparam
        ;;
    fm )
        fm
        ;;
    devm )
        devm $exeparam
        ;;
    stmm )
        stmm $exeparam
        ;;
    cfg )
        cfg $exeparam
        ;;
    tsync)
        tsync $exeparam
        ;;
    perfhelper)
        $NOS_ROOT/scripts/perf_helper.sh $exeparam
        ;;
    easy_package)
        $NOS_ROOT/scripts/res_collect.sh $exeparam
        ;;
    *)
        usage
        ;;
esac
exit

