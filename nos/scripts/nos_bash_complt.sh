#!/bin/bash

_neta_bag()
{
    local cur prev
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"
    options="record play info convert save stat attachment"
    opts_stat="check_seq"
    if ((COMP_CWORD == 1)); then
        COMPREPLY=( $(compgen -W "${options}" -- ${cur}) )
    elif ((COMP_CWORD >= 2)); then
        if [[ ${prev} == "stat" ]]; then
            COMPREPLY=( $(compgen -W "${opts_stat}" -- ${cur}) )
        else
            compopt -o default
        fi
    fi
}

complete -F _neta_bag bag


_neta_topic()
{
    local cur prev
    cur="${COMP_WORDS[COMP_CWORD]}"
    options="echo hz monitor list latency"

    if ((COMP_CWORD == 1)); then
        COMPREPLY=( $(compgen -W "${options}" -- ${cur}) )
    elif ((COMP_CWORD >= 2)); then
        compopt -o default
    fi
}

complete -F _neta_topic topic


_neta_tsync()
{
    local cur prev
    cur="${COMP_WORDS[COMP_CWORD]}"
    options="DP MP DP_AND_MP"

    if ((COMP_CWORD == 1)); then
        COMPREPLY=( $(compgen -W "${options}" -- ${cur}) )
    elif ((COMP_CWORD >= 2)); then
        compopt -o default
    fi
}

complete -F _neta_tsync tsync

_neta_cfg()
{
    local cur prev
    cur="${COMP_WORDS[COMP_CWORD]}"
    options="set get del setdefault reset convertcalib getmonitorclients"
    if ((COMP_CWORD == 1)); then
        COMPREPLY=( $(compgen -W "${options}" -- ${cur}) )
    elif ((COMP_CWORD >= 2)); then
        compopt -o default
    fi
}

complete -F _neta_cfg cfg

_hz_log_tools_()
{
    local cur prev opts
    COMPREPLY=()
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"
    opts="continue setloglevel convert"
    opts_appid_ctx="IGNORE.IGNORE:kCritical IGNORE.IGNORE:kError IGNORE.IGNORE:kWarn IGNORE.IGNORE:kInfo IGNORE.IGNORE:kDebug IGNORE.IGNORE:kTrace"

    # Complete the options
    if [ $COMP_CWORD -eq 1 ]; then
        COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
    elif [ $COMP_CWORD -eq 2 ]; then
        if [[ ${prev} == "setloglevel" ]]; then
            COMPREPLY=( $(compgen -W "${opts_appid_ctx}" -- ${cur}) )
        else
            local IFS=$'\n'
            compopt -o filenames
            COMPREPLY=( $(compgen -f -- ${cur}) )
        fi
    else
        local IFS=$'\n'
        compopt -o filenames
        COMPREPLY=( $(compgen -f -- ${cur}) )
    fi
}

complete -F _hz_log_tools_ log


_neta_dbg()
{
    local cur prev
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"
    options="set query restart request reboot reset"
    selta="modeList processStatus"
    seltb="<processName>"
    seltc="mode"
    itemc="<modeName>"

    if ((COMP_CWORD == 1)); then
        COMPREPLY=( $(compgen -W "${options}" -- ${cur}) )
    fi

    case "${prev}" in query )
            COMPREPLY=( $(compgen -W "${selta}" -- ${cur}) )
            return 0
        ;;
        restart )
            COMPREPLY=( $(compgen -W "${seltb}" -- ${cur}) )
            return 0
        ;;
        request | switch)
            COMPREPLY=( $(compgen -W "${seltc}" -- ${cur}) )
            return 0
        ;;
    esac

    if ((COMP_CWORD == 2)); then
        COMPREPLY=( $(compgen -W "${seltc}" -- ${cur}) )
    fi

    case "${prev}" in mode )
            COMPREPLY=( $(compgen -W "${itemc}" -- ${cur}) )
            return 0
        ;;
    esac
}

complete -F _neta_dbg smdbg

_neta_perfhelper()
{
    local cur prev sub_cmds
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"
    sub_cmds="perf sched stat"

    # echo -e "\nCOMP_CWORD: ${COMP_CWORD}, cur: ${cur}, prev: ${prev}"
    if [ "$prev" = "perfhelper" ]; then
        COMPREPLY=( $(compgen -W "${sub_cmds}" -- ${cur}))
    fi
}

complete -F _neta_perfhelper perfhelper

_nos_()
{
    local cur prev
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"
    pprev="${COMP_WORDS[COMP_CWORD-2]}"
    options="dbg bag topic log devm stmm fm cfg tsync perfhelper easy_package"
    sel_d="set query restart request reboot reset"
    sel_b="record play info convert attachment"
    sel_t="echo hz monitor"
    sel_l="continue setloglevel convert"
    sel_s="help all cpu mem disk file mnand network process temp voltage"
    sel_c="set get del setdefault reset convertcalib getmonitorclients getparaminfolist getclientinfolist"
    itm_d1="startupMode"
    itm_d2="modeList processStatus"
    itm_d3="<processName>"
    itm_d4="mode"
    itm_d4_1="<modeName>"
    opts_log_appid_ctx="IGNORE.IGNORE:kCritical IGNORE.IGNORE:kError IGNORE.IGNORE:kWarn IGNORE.IGNORE:kInfo IGNORE.IGNORE:kDebug IGNORE.IGNORE:kTrace"
    local sel_tsync="DP MP DP_AND_MP"
    local sel_perfhelper="perf sched stat"

    if ((COMP_CWORD == 1)); then
        COMPREPLY=( $(compgen -W "${options}" -- ${cur}) )
    elif ((COMP_CWORD == 2)); then
    case "${prev}" in
        dbg )
        if ((COMP_CWORD == 2)); then
            COMPREPLY=( $(compgen -W "${sel_d}" -- ${cur}) )
        fi
                return 0
            ;;
        bag )
            if ((COMP_CWORD == 2)); then
            COMPREPLY=( $(compgen -W "${sel_b}" -- ${cur}) )
            elif ((COMP_CWORD >= 3)); then
                compopt -o default
            fi
                return 0
            ;;
        topic)
               if ((COMP_CWORD == 2)); then
            COMPREPLY=( $(compgen -W "${sel_t}" -- ${cur}) )
            elif ((COMP_CWORD >= 3)); then
                compopt -o default
            fi
        return 0
            ;;
        cfg)
               if ((COMP_CWORD == 2)); then
            COMPREPLY=( $(compgen -W "${sel_c}" -- ${cur}) )
            elif ((COMP_CWORD >= 3)); then
                compopt -o default
            fi
        return 0
            ;;
        log)
               if ((COMP_CWORD == 2)); then
            COMPREPLY=( $(compgen -W "${sel_l}" -- ${cur}) )
            fi
        return 0
            ;;
        devm)
            if ((COMP_CWORD == 2)); then
                devm_commands="cpu-info dev-info dev-status ifdata iostat read-did upgrade"
                COMPREPLY=( $(compgen -W "${devm_commands}" -- ${cur}) )
            fi
            return 0
            ;;
        stmm )
            if ((COMP_CWORD == 2)); then
            COMPREPLY=( $(compgen -W "${sel_s}" -- ${cur}) )
            elif ((COMP_CWORD >= 3)); then
                compopt -o default
            fi
                return 0
            ;;
        fm)
            if ((COMP_CWORD == 2)); then
                COMPREPLY=( $(compgen -W "" -- ${cur}) )
            fi
            return 0
            ;;
        tsync)
            if ((COMP_CWORD == 2)); then
                COMPREPLY=( $(compgen -W "${sel_tsync}" -- ${cur}) )
            elif ((COMP_CWORD >= 3)); then
                compopt -o default
            fi
            return 0
            ;;
        perfhelper)
            # COMP_CWORD=$((COMP_CWORD + 1))
            _neta_perfhelper
            # if ((COMP_CWORD == 2)); then
            #     COMPREPLY=( $(compgen -W "${sel_perfhelper}" -- ${cur}))
            # fi
            return 0
            ;;
        esac

    elif ((COMP_CWORD == 3)); then
        if [ ${pprev} == "dbg" ]; then
            case "${prev}" in query )
                COMPREPLY=( $(compgen -W "${itm_d2}" -- ${cur}) )
                ;;
            restart )
                COMPREPLY=( $(compgen -W "${itm_d3}" -- ${cur}) )
                ;;
            request )
                COMPREPLY=( $(compgen -W "${itm_d4}" -- ${cur}) )
                ;;
            set )
                COMPREPLY=( $(compgen -W "${itm_d1}" -- ${cur}) )
                ;;
            esac
        elif [ ${pprev} == "log" ]; then
            case "${prev}" in setloglevel )
                COMPREPLY=( $(compgen -W "${opts_log_appid_ctx}" -- ${cur}) )
                ;;
            continue )
                local IFS=$'\n'
                compopt -o filenames
                COMPREPLY=( $(compgen -f -- ${cur}) )
                ;;
            * )
                local IFS=$'\n'
                compopt -o filenames
                COMPREPLY=( $(compgen -f -- ${cur}) )
                ;;
            esac
        elif [ ${pprev} == "devm" ]; then
            case "${prev}" in
            read-did)
                read_did_options="0x0110 0x900F 0xF170 0xF180 0xF186 0xF187 0xF188 0xF18A \
                                0xF18B 0xF18C 0xF190 0xF191 0xF198 0xF199 0xF19D 0xF1B0 \
                                0xF1BF 0xF1C0 0xF1D0 0xF1E0 0xF1E1 0xF1E2 0xF1E3"
                COMPREPLY=( $(compgen -W "${read_did_options}" -- ${cur}) )
                ;;
            ifdata)
                ifdata_options="si bips bops"
                COMPREPLY=( $(compgen -W "${ifdata_options}" -- ${cur}) )
                ;;
            upgrade)
                update_options="status precheck progress update version finish result cur_partition help"
                COMPREPLY=( $(compgen -W "${update_options}" -- ${cur}) )
                ;;
            esac
        fi

    elif ((COMP_CWORD >= 4)); then
        case "${prev}" in mode | startupMode )
                    COMPREPLY=( $(compgen -W "${itm_d4_1}" -- ${cur}) )
                ;;
            'si' | 'bips' | 'bops')
                ifdata_options="<interface>"
                COMPREPLY=( $(compgen -W "${ifdata_options}" -- ${cur}) )
                ;;
            'update')
                local IFS=$'\n'
                compopt -o filenames
                COMPREPLY=( $(compgen -f -- ${cur}) )
                ;;
            * )
                local IFS=$'\n'
                compopt -o filenames
                COMPREPLY=( $(compgen -f -- ${cur}) )
                ;;
            esac

    fi
}

complete -F _nos_ nos

_devm_completion() {
    local cur prev opts

    COMPREPLY=()
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"

    # 一级命令列表
    top_commands="cpu-info dev-info dev-status ifdata iostat read-did upgrade"

    # 定义各个一级命令的二级命令、选项和参数
    case "${prev}" in
        'devm')
            COMPREPLY=( $(compgen -W "${top_commands}" -- ${cur}) )
            return 0
            ;;
        'read-did')
            read_did_options="0x0110 0x900F 0xF170 0xF180 0xF186 0xF187 0xF188 0xF18A \
                            0xF18B 0xF18C 0xF190 0xF191 0xF198 0xF199 0xF19D 0xF1B0 \
                            0xF1BF 0xF1C0 0xF1D0 0xF1E0 0xF1E1 0xF1E2 0xF1E3"
            COMPREPLY=( $(compgen -W "${read_did_options}" -- ${cur}) )
            return 0
            ;;
        'ifdata')
            ifdata_options="si bips bops"
            COMPREPLY=( $(compgen -W "${ifdata_options}" -- ${cur}) )
            return 0
            ;;
        'si' | 'bips' | 'bops')
            ifdata_options="<interface>"
            COMPREPLY=( $(compgen -W "${ifdata_options}" -- ${cur}) )
            return 0
            ;;
        'upgrade')
            update_options="status precheck progress update version finish result cur_partition help"
            COMPREPLY=( $(compgen -W "${update_options}" -- ${cur}) )
            return 0
            ;;
        'update')
            local IFS=$'\n'
            compopt -o filenames
            COMPREPLY=( $(compgen -f -- ${cur}) )
            return 0
            ;;
        # 可继续添加其他一级命令的补全规则
        *)
            ;;
    esac
}
complete -F _devm_completion devm


