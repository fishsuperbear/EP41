function _log_convert() {
    local cur_cmd=${COMP_WORDS[COMP_CWORD]}
    local prev_cmd=${COMP_WORDS[COMP_CWORD-1]}
    local prev_two_cmd=${COMP_WORDS[COMP_CWORD-2]}

    local cmd_elements=""
    local comp_filename_flag=0

    case "${prev_cmd}" in
    "log-convert" )
        if [ "${COMP_CWORD}" = "2" ]; then
            comp_filename_flag=1   # 使用了空格占位符
            cmd_elements="datatime manatime continue"
        fi
        ;;
    "datatime" )
        if [ "${COMP_CWORD}" = "3" ]; then
            comp_filename_flag=1   # 使用了空格占位符
            cmd_elements="continue"
        fi
        ;;
    "manatime" )
        if [ "${COMP_CWORD}" = "3" ]; then
            comp_filename_flag=1   # 使用了空格占位符
            cmd_elements="continue"
        fi
        ;;
    "continue" )
        if [ "${COMP_CWORD}" = "3" -o "${COMP_CWORD}" = "4" ]; then
            comp_filename_flag=1   # 使用了空格占位符
        fi
        ;;

    * )
        ;;
    esac

    case "${prev_two_cmd}" in
    "continue" )
        if [ "${COMP_CWORD}" = "4" -o "${COMP_CWORD}" = "5" ]; then
            comp_filename_flag=1   # 使用了空格占位符
        fi
        ;;
    * )
        ;;
    esac

    # 生成占位符，添加随机数后缀
    local place_holder_space="clt_ph_space_${RANDOM}"
    cmd_elements="$cmd_elements $(compgen -o nospace -o default)"
    compgen -c readline
    cur_cmd=$(sed "s/\\ /${place_holder_space}/g" <<< "${cur_cmd}")
    COMPREPLY=($(compgen -o default -W "${cmd_elements}" -- "$cur_cmd"))

    # log-convert命令行支持文件名输入，而文件名支持空格，补全字符串需要处理空格文件名，
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