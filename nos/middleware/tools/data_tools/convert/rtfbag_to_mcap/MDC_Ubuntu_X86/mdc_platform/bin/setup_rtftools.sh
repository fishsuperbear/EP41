#!/bin/sh
# ===============================================================================
# Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
# Description: rtftoolsbash - functions to support VRTF_RTFTOOLS users
# Create: 2020-03-05
# ===============================================================================
_init_param() {
    local cur_option bag_opts opts event_opts bagfile cur secondcmd
    COMPREPLY=()
}
# command line completion rtftools -> rtfevent
_complete_func_rtfevent() {
    cur_option="${COMP_WORDS[COMP_CWORD]}"
    if [[ $COMP_CWORD -eq 1 ]]
    then
        COMPREPLY=( $(compgen -W 'list info hz show echo latency' -- $cur_option) )
        return 0
    elif [[ $COMP_CWORD -eq 2 ]]
    then
        secondcmd="${COMP_WORDS[1]}"
        case "$secondcmd" in
        'info')
              _complete_rtfevent_info
              return 0;;
        'list')
              _complete_rtfevent_list
              return 0;;
        'show')
              _complete_rtfevent_show
              return 0;;
        'hz')
              _complete_rtfevent_hz
              return 0;;
        'echo')
              _complete_rtfevent_echo
              return 0;;
        'latency')
              _complete_rtfevent_latency
              return 0;;
        '*')
              ;;
        esac
    elif [[ $COMP_CWORD -gt 2 ]]
    then
        secondcmd="${COMP_WORDS[1]}"
        case "$secondcmd" in
        'hz')
              _complete_rtfevent_hz
              return 0;;
        'echo')
              _complete_rtfevent_echo
              return 0;;
        'latency')
              _complete_rtfevent_latency
              return 0;;
        '*')
              ;;
        esac
    fi
    return 0
}

_complete_rtfevent_info() {
    cur_option="${COMP_WORDS[COMP_CWORD]}"
    case "${cur_option}" in
    -)
        COMPREPLY=( $(compgen -W '-h' -- ${cur_option}) )
        return 0;;
    --*)
        COMPREPLY=( $(compgen -W '--help' -- ${cur_option}) )
        return 0;;
    *)
        event_opts=$(rtfevent list 2> /dev/null | grep -v WARNING | grep -e '\/.')
        COMPREPLY=( $(compgen -W "${event_opts}" -- ${cur_option}) )
        return 0;;
    esac
}

_complete_rtfevent_list() {
    play_short_opts_witharg="-h -p -s -a -c"
    play_long_opts_witharg="--help --publisher --subscriber --all --communicable"
    cur_option="${COMP_WORDS[COMP_CWORD]}"
    case "${cur_option}" in
    -)
        COMPREPLY=( $(compgen -W "${play_short_opts_witharg}" -- ${cur_option}) )
        return 0;;
    --*)
        COMPREPLY=( $(compgen -W "${play_long_opts_witharg}" -- ${cur_option}) )
        return 0;;
    *)
        ;;
    esac
}

_complete_rtfevent_show() {
    play_short_opts_witharg="-h"
    play_long_opts_witharg="--help"
    cur_option="${COMP_WORDS[COMP_CWORD]}"
    case "${cur_option}" in
    -)
        COMPREPLY=( $(compgen -W "${play_short_opts_witharg}" -- ${cur_option}) )
        return 0;;
    --*)
        COMPREPLY=( $(compgen -W "${play_long_opts_witharg}" -- ${cur_option}) )
        return 0;;
    *)
        opts=$(rtfevent list 2> /dev/null | grep -v WARNING | grep -e '\/.')
        COMPREPLY=( $(compgen -W "${opts}" -- ${cur_option}) )
        return 0;;
    esac
}

_complete_rtfevent_hz() {
    play_short_opts_witharg="-h -w"
    play_long_opts_witharg="--help --window --someip-network --dds-network"
    cur_option="${COMP_WORDS[COMP_CWORD]}"
    case "${cur_option}" in
    -)
        COMPREPLY=( $(compgen -W "${play_short_opts_witharg}" -- ${cur_option}) )
        return 0;;
    --*)
        COMPREPLY=( $(compgen -W "${play_long_opts_witharg}" -- ${cur_option}) )
        return 0;;
    *)
        opts=$(rtfevent list 2> /dev/null | grep -v WARNING | grep -e '\/.')
        COMPREPLY=( $(compgen -W "${opts}" -- ${cur_option}) )
        return 0;;
    esac
}

_complete_rtfevent_latency() {
    play_short_opts_witharg="-h -w"
    play_long_opts_witharg="--help --window --start --stop --status"
    cur_option="${COMP_WORDS[COMP_CWORD]}"
    case "${cur_option}" in
    -)
        COMPREPLY=( $(compgen -W "${play_short_opts_witharg}" -- ${cur_option}) )
        return 0;;
    --*)
        COMPREPLY=( $(compgen -W "${play_long_opts_witharg}" -- ${cur_option}) )
        return 0;;
    *)
        opts=$(rtfevent list 2> /dev/null | grep -v WARNING | grep -e '\/.')
        COMPREPLY=( $(compgen -W "${opts}" -- ${cur_option}) )
        return 0;;
    esac
}

_complete_rtfevent_echo() {
    play_short_opts_witharg="-h"
    play_long_opts_witharg="--help --someip-network --dds-network --untypeset --noarr"
    cur="${COMP_WORDS[COMP_CWORD]}"
    case "${cur_option}" in
    -)
        COMPREPLY=( $(compgen -W "${play_short_opts_witharg}" -- ${cur_option}) )
        return 0;;
    --*)
        COMPREPLY=( $(compgen -W "${play_long_opts_witharg}" -- ${cur_option}) )
        return 0;;
    *)
        opts=$(rtfevent list 2> /dev/null | grep -v WARNING | grep -e '\/.')
        COMPREPLY=( $(compgen -W "${opts}" -- ${cur_option}) )
        return 0;;
    esac
}

# command line completion rtftools -> rtfbag
_complete_func_rtfbag() {
    cur_option="${COMP_WORDS[COMP_CWORD]}"
    if [[ $COMP_CWORD -eq 1 ]]
    then
        COMPREPLY=( $(compgen -W 'help info play record extract fix' -- $cur_option) )
        return 0
    elif [[ $COMP_CWORD -ge 2 ]]
    then
        secondcmd="${COMP_WORDS[1]}"
        case "$secondcmd" in
        'info')
              _complete_rtfbag_info
              return 0;;
        'play')
              _complete_rtfbag_play
              return 0;;
        'record')
              _complete_rtfbag_record
              return 0;;
        'extract')
              _complete_rtfbag_extract
              return 0;;
        'fix')
              _complete_rtfbag_fix
              return 0;;
        '*')
              ;;
        esac
    fi
    return 0
}

_complete_rtfbag_info() {
    cur_option="${COMP_WORDS[COMP_CWORD]}"
    case "${cur_option}" in
    -)
        COMPREPLY=( $(compgen -W '-h' -- ${cur_option}) )
        return 0;;
    --*)
        COMPREPLY=( $(compgen -W '--help --freq' -- ${cur_option}) )
        return 0;;
    *)
        bag_opts=$(ls *.bag 2> /dev/null)
        COMPREPLY=( $(compgen -W "${bag_opts}" -- ${cur_option}) )
        return 0;;
    esac
}

_complete_rtfbag_play() {
    play_short_opts_witharg="-h -q -i -l -d -r -s -u"
    play_long_opts_witharg="--help --quiet --pause --immediate --queue --hz --delay
                            --rate --start --duration --skip-empty --loop --port --events
                            --pause-events --skip-events --dds-network --someip-network --force-play-events --change-mode
                            --adjust-clock"
    cur_option="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"
    int=0
    while(( $int <= $COMP_CWORD ))
    do
        if [[ ${COMP_WORDS[${int}]} == *.bag ]]
        then
            bagfile=${COMP_WORDS[${int}]}
        fi
        let "int++"
    done
    case "${prev},${cur_option}" in
    --events,*)
        cmd_output=$(rtfbag info ${bagfile} 2> /dev/null | grep -a '\/.' | grep -n events | cut -d ":" -f 1)
        if [[ $cmd_output -eq "" ]]
        then
            cmd_output=15
        fi
        event_opts=$(rtfbag info ${bagfile} 2> /dev/null | grep -a '\/.' | awk 'NR=='${cmd_output}'{print $2} NR>'${cmd_output}'{print $1}')
        COMPREPLY=( $(compgen -W "${event_opts}" -- ${cur_option}) )
        return 0;;
    --skip-events,*)
        cmd_output=$(rtfbag info ${bagfile} 2> /dev/null | grep -a '\/.' | grep -n events | cut -d ":" -f 1)
        if [[ $cmd_output -eq "" ]]
        then
            cmd_output=15
        fi
        event_opts=$(rtfbag info ${bagfile} 2> /dev/null | grep -a '\/.' | awk 'NR=='${cmd_output}'{print $2} NR>'${cmd_output}'{print $1}')
        COMPREPLY=( $(compgen -W "${event_opts}" -- ${cur_option}) )
        return 0;;
    --pause-events,*)
        cmd_output=$(rtfbag info ${bagfile} 2> /dev/null | grep -a '\/.' | grep -n events | cut -d ":" -f 1)
        if [[ $cmd_output -eq "" ]]
        then
            cmd_output=15
        fi
        event_opts=$(rtfbag info ${bagfile} 2> /dev/null | grep -a '\/.' | awk 'NR=='${cmd_output}'{print $2} NR>'${cmd_output}'{print $1}')
        COMPREPLY=( $(compgen -W "${event_opts}" -- ${cur_option}) )
        return 0;;
    *,-)
        COMPREPLY=( $(compgen -W "${play_short_opts_witharg}" -- ${cur_option}) )
        return 0;;
    *,--*)
        COMPREPLY=( $(compgen -W "${play_long_opts_witharg}" -- ${cur_option}) )
        return 0;;
    *,*)
        bag_opts=$(ls *.bag 2> /dev/null)
        COMPREPLY=( $(compgen -W "${bag_opts}" -- ${cur_option}) )
        return 0;;
    esac
}

_complete_rtfbag_record() {
    play_short_opts_witharg="-h -a -o -O -b -l -p"
    play_long_opts_witharg="--help --all --output-prefix --output-name
                            --split --max-splits --size --duration --buffsize
                            --limit --path --dds-network --someip-network --skip-frame --compression"
    cur_option="${COMP_WORDS[COMP_CWORD]}"
    case "${cur_option}" in
    -)
        COMPREPLY=( $(compgen -W "${play_short_opts_witharg}" -- ${cur_option}) )
        return 0;;
    --*)
        COMPREPLY=( $(compgen -W "${play_long_opts_witharg}" -- ${cur_option}) )
        return 0;;
    *)
        event_opts=$(rtfevent list 2> /dev/null | grep -v WARNING | grep -e '\/.')
        COMPREPLY=( $(compgen -W "${event_opts}" -- ${cur_option}) )
        return 0;;
    esac
}

_complete_rtfbag_extract() {
    cur="${COMP_WORDS[COMP_CWORD]}"
    case "${cur}" in
    -)
        COMPREPLY=( $(compgen -W '-h -s -e' -- ${cur}) )
        return 0;;
    --*)
        COMPREPLY=( $(compgen -W '--help --start-time --end-time' -- ${cur}) )
        return 0;;
    *)
        opts=$(ls *.bag 2> /dev/null)
        COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
        return 0;;
    esac
}

_complete_rtfbag_fix() {
    cur_option="${COMP_WORDS[COMP_CWORD]}"
    case "${cur_option}" in
    -)
        COMPREPLY=( $(compgen -W '-h' -- ${cur_option}) )
        return 0;;
    --*)
        COMPREPLY=( $(compgen -W '--help --path' -- ${cur_option}) )
        return 0;;
    *)
        bag_opts=$(ls *.bag.active 2> /dev/null)
        COMPREPLY=( $(compgen -W "${bag_opts}" -- ${cur_option}) )
        return 0;;
    esac
}

# command line completion rtftools -> rtfnode
_complete_func_rtfnode() {
    cur_option="${COMP_WORDS[COMP_CWORD]}"
    if [[ $COMP_CWORD -eq 1 ]]
    then
        COMPREPLY=( $(compgen -W 'list info' -- $cur_option) )
        return 0
    elif [[ $COMP_CWORD -eq 2 ]]
    then
        secondcmd="${COMP_WORDS[1]}"
        case "$secondcmd" in
        'info')
              _complete_rtfnode_info
              return 0;;
        'list')
              _complete_rtfnode_list
              return 0;;
        '*')
              ;;
        esac
    fi
    return 0
}

_complete_rtfnode_info() {
    cur_option="${COMP_WORDS[COMP_CWORD]}"
    case "${cur_option}" in
    -)
        COMPREPLY=( $(compgen -W '-h' -- ${cur_option}) )
        return 0;;
    --*)
        COMPREPLY=( $(compgen -W '--help' -- ${cur_option}) )
        return 0;;
    *)
        node_opts=$(rtfnode list 2> /dev/null | grep -v WARNING | grep -E '^/+')
        COMPREPLY=( $(compgen -W "${node_opts}" -- ${cur_option}) )
        return 0;;
    esac
}

_complete_rtfnode_list() {
    play_short_opts_witharg="-h"
    play_long_opts_witharg="--help"
    cur_option="${COMP_WORDS[COMP_CWORD]}"
    case "${cur_option}" in
    -)
        COMPREPLY=( $(compgen -W "${play_short_opts_witharg}" -- ${cur_option}) )
        return 0;;
    --*)
        COMPREPLY=( $(compgen -W "${play_long_opts_witharg}" -- ${cur_option}) )
        return 0;;
    *)
        ;;
    esac
}

# command line completion rtftools -> rtfmethod
_complete_func_rtfmethod() {
    cur_option="${COMP_WORDS[COMP_CWORD]}"
    if [[ $COMP_CWORD -eq 1 ]]
    then
        COMPREPLY=( $(compgen -W 'list info type call' -- $cur_option) )
        return 0
    elif [[ $COMP_CWORD -eq 2 ]]
    then
        secondcmd="${COMP_WORDS[1]}"
        case "$secondcmd" in
        'info')
              _complete_rtfmethod_info
              return 0;;
        'list')
              _complete_rtfmethod_list
              return 0;;
        'type')
              _complete_rtfmethod_type
              return 0;;
        'call')
              _complete_rtfmethod_call
              return 0;;
        '*')
              ;;
        esac
    elif [[ $COMP_CWORD -gt 2 ]]
    then
        secondcmd="${COMP_WORDS[1]}"
        case "$secondcmd" in
        'call')
              _complete_rtfmethod_call
              return 0;;
        '*')
              ;;
        esac
    fi
    return 0
}

_complete_rtfmethod_info() {
    cur_option="${COMP_WORDS[COMP_CWORD]}"
    case "${cur_option}" in
    -)
        COMPREPLY=( $(compgen -W '-h' -- ${cur_option}) )
        return 0;;
    --*)
        COMPREPLY=( $(compgen -W '--help' -- ${cur_option}) )
        return 0;;
    *)
        event_opts=$(rtfmethod list 2> /dev/null | grep -v WARNING | grep -e '\/.')
        COMPREPLY=( $(compgen -W "${event_opts}" -- ${cur_option}) )
        return 0;;
    esac
}

_complete_rtfmethod_list() {
    play_short_opts_witharg="-h"
    play_long_opts_witharg="--help"
    cur_option="${COMP_WORDS[COMP_CWORD]}"
    case "${cur_option}" in
    -)
        COMPREPLY=( $(compgen -W "${play_short_opts_witharg}" -- ${cur_option}) )
        return 0;;
    --*)
        COMPREPLY=( $(compgen -W "${play_long_opts_witharg}" -- ${cur_option}) )
        return 0;;
    *)
        ;;
    esac
}

_complete_rtfmethod_type() {
    play_short_opts_witharg="-h"
    play_long_opts_witharg="--help"
    cur_option="${COMP_WORDS[COMP_CWORD]}"
    case "${cur_option}" in
    -)
        COMPREPLY=( $(compgen -W "${play_short_opts_witharg}" -- ${cur_option}) )
        return 0;;
    --*)
        COMPREPLY=( $(compgen -W "${play_long_opts_witharg}" -- ${cur_option}) )
        return 0;;
    *)
        opts=$(rtfmethod list 2> /dev/null | grep -v WARNING | grep -e '\/.')
        COMPREPLY=( $(compgen -W "${opts}" -- ${cur_option}) )
        return 0;;
    esac
}

_complete_rtfmethod_call() {
    play_short_opts_witharg="-h -p"
    play_long_opts_witharg="--help --path --someip-network --dds-network --untypeset --noarr"
    cur_option="${COMP_WORDS[COMP_CWORD]}"

    if [[ $COMP_CWORD -eq 2 ]]
    then
        opts=$(rtfmethod list 2> /dev/null | grep -v WARNING | grep -e '\/.')
        COMPREPLY=( $(compgen -W "${opts}" -- ${cur_option}) )
        return 0
    fi
    case "${cur_option}" in
    -)
        COMPREPLY=( $(compgen -W "${play_short_opts_witharg}" -- ${cur_option}) )
        return 0;;
    --*)
        COMPREPLY=( $(compgen -W "${play_long_opts_witharg}" -- ${cur_option}) )
        return 0;;
    *)
        return 0;;
    esac
}

complete -F "_complete_func_rtfbag" -o default "rtfbag"
complete -F "_complete_func_rtfnode" "rtfnode"
complete -F "_complete_func_rtfevent" "rtfevent"
complete -F "_complete_func_rtfmethod" -o default "rtfmethod"
