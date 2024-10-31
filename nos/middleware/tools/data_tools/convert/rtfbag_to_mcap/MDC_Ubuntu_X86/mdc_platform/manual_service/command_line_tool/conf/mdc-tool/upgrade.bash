function _get_upgrade_files() {
    local file_dir=$1
    local place_holder_space=$2

    local cmd_elements
    local element

    # get all reguler files in specfic directory
    for file in "${file_dir}"/*
    do
        if [ -f "${file}" ] ; then
            element=$(sed "s/ /${place_holder_space}/g" <<< "${file}")
            cmd_elements="${cmd_elements} ${element}"
        fi
    done

    echo "${cmd_elements}"
}

function _comp_precheck_file() {
    local cmd_elements=""
    local cmd=${COMP_WORDS[COMP_CWORD-1]}

    case "${cmd}" in
    "all" | "minimal" | "custom" )
        cmd_elements=""
        ;;
    * )
        cmd_elements="all minimal custom"
        ;;
    esac

    echo "${cmd_elements}"
}

function _comp_install_file() {
    local cmd_elements=""
    local cmd=${COMP_WORDS[COMP_CWORD-1]}

    case "${cmd}" in
    "all" )
        cmd_elements=""
        ;;
    * )
        cmd_elements="-f all"
        ;;
    esac

    echo "${cmd_elements}"
}

function _upgrade() {
    local cur_cmd=${COMP_WORDS[COMP_CWORD]}
    local prev_cmd=${COMP_WORDS[COMP_CWORD-1]}
    local prev2_cmd=${COMP_WORDS[COMP_CWORD-2]}

    # 生成占位符，添加随机数后缀
    local place_holder_space="clt_ph_space_${RANDOM}"
    local comp_filename_flag=0

    local upgrade_root_path="/opt/usr/upgrade"
    local cmd_elements=""

    # 每个命令元素，都需要判断“命令行内命令元素数量”，以及"命令元素前后关联性"，
    # 否则用户构造异常输入，因为没有检查，会出现错误的命令联系
    case "${prev_cmd}" in
    # completion of cmd line "upgrade"
    "upgrade" )
        if [ "${COMP_CWORD}" = "2" ] ; then
            cmd_elements="help activate display enter exit finish install precheck query"
        fi
        ;;
    # completion of cmd line "upgrade display"
    "display" )
        if [ "${COMP_CWORD}" = "3" ] ; then
            cmd_elements="version history"
        fi
        ;;
    # completion of cmd line "upgrade finish"
    "finish" )
        if [ "${COMP_CWORD}" = "3" ] ; then
            cmd_elements="auto"
        fi
        ;;
    # completion of cmd line "upgrade enter update-mode/recovery"
    "enter" )
        if [ "${COMP_CWORD}" = "3" ] ; then
            cmd_elements="update-mode recovery"
        fi
        ;;
    # completion of cmd line "upgrade exit"
    "exit" )
        if [ "${COMP_CWORD}" = "3" ] ; then
            cmd_elements="update-mode"
        fi
        ;;
    # completion of cmd line "upgrade precheck"
    "precheck" )
        if [ "${COMP_CWORD}" = "3" ] ; then
            cmd_elements="all minimal custom"
        fi
        ;;
    # completion of cmd line "upgrade enter recovery clear"
    "recovery" )
        if [ "${COMP_CWORD}" = "4" ] ; then
            cmd_elements="clear"
        fi
        ;;
    # completion of cmd line "upgrade precheck custom"
    "custom" )
        if [ "${COMP_CWORD}" = "4" ] && [ "${prev2_cmd}" = "precheck" ] ; then
            comp_filename_flag=1   # 使用了空格占位符
            cmd_elements=$(_get_upgrade_files ${upgrade_root_path} ${place_holder_space})
            # 用户输入'\ '也需要替换为占位符，否则根据输入过滤的时候，会无法匹配
            cur_cmd=$(sed "s/\\ /${place_holder_space}/g" <<< "${cur_cmd}")
        fi
        ;;
    # completion of cmd line "upgrade install"
    "install" )
        if [ "${COMP_CWORD}" = "3" ] ; then
            comp_filename_flag=1   # 使用了空格占位符
            cmd_elements=$(_get_upgrade_files ${upgrade_root_path} ${place_holder_space})
            # 用户输入'\ '也需要替换为占位符，否则根据输入过滤的时候，会无法匹配
            cur_cmd=$(sed "s/\\ /${place_holder_space}/g" <<< "${cur_cmd}")
        fi
        ;;
    # completion of cmd line "upgrade install < filename > -f"
    "-f" )
        if [ "${COMP_CWORD}" = "5" ] ; then
            cmd_elements="1 2 3"
        fi
        ;;
    # completion of cmd line "upgrade install -f { 1 | 2 | 3 }"
    "1" | "2" | "3" )
        if [ "${COMP_CWORD}" = "6" ] && [ "${prev2_cmd}" = "-f" ] ; then
            cmd_elements="all"
        fi
        ;;
    # completion of cmd line "upgrade query"
    "query" )
        if [ "${COMP_CWORD}" = "3" ] ; then
            cmd_elements="precheck status progress"
        fi
        ;;
    # completion of cmd line "upgrade query process"
    "progress" )
        if [ "${COMP_CWORD}" = "4" ] && [ "${prev2_cmd}" = "query" ] ; then
            cmd_elements="precheck install activate"
        fi
        ;;
    # process specifications
    * )
        case "${prev2_cmd}" in
        # completion of cmd line "upgrade install < filename > [ -f { 1 | 2 | 3 } ] [ all ]"
        "install" )
            local spec_file=${COMP_WORDS[3]}
            cmd_elements=$(_comp_install_file "${spec_file}")
            ;;
        * )
            cmd_elements=""
            ;;
        esac

        ;;
    esac

    COMPREPLY=($(compgen -W "${cmd_elements}" -- "$cur_cmd"))

    # 升级命令行支持文件名输入，而文件名支持空格，补全字符串需要处理空格文件名，
    # 使用compgen生成数组时，空格会当成分隔符处理，即使加了斜杠'\ '也会被转义，
    # 针对此场景clt框架提供简单的处理机制，提供'空格占位符'，模块补全字符串中，若包含空格的，将空格替换为'空格占位符'，
    # 生成COMPREPLY数组后，将COMPREPLY数组内的占位符内容替换为'\ '
    # 注意：此方法只能处理文件名的部分场景，有其他特殊字符或场景，需要更细致处理
    if [ "${comp_filename_flag}" = "1" ] ; then
        local i=0
        while [ $i -lt ${#COMPREPLY[@]} ]
        do
            COMPREPLY[$i]=$(sed "s/${place_holder_space}/\\\\ /g" <<< "${COMPREPLY[$i]}")
            let i++
        done
    fi
}
