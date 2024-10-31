function _log_control() {
    local cur_cmd=${COMP_WORDS[COMP_CWORD]}
    local prev_cmd=${COMP_WORDS[COMP_CWORD-1]}
    local prev_two_cmd=${COMP_WORDS[COMP_CWORD-2]}

    local cmd_elements=""

    case "${prev_cmd}" in
    "log-control" )
        if [ "${COMP_CWORD}" = "2" ]; then
            cmd_elements="appid ctxid level"
        fi
        ;;
    "level" )
        if [ "${COMP_CWORD}" = "3" -o "${COMP_CWORD}" = "5" -o "${COMP_CWORD}" = "7" ]; then
            cmd_elements="off fatal error warn info debug verbose"
        fi
        ;;
    * )
        ;;
    esac

    case "${prev_two_cmd}" in
    "appid" )
        if [ "${COMP_CWORD}" = "4" ]; then
            cmd_elements="ctxid level"
        fi
        ;;
    "ctxid" )
        if [ "${COMP_CWORD}" = "4" -o "${COMP_CWORD}" = "6" ]; then
            cmd_elements="level"
        fi
        ;;
    * )
        ;;
    esac

    COMPREPLY=($(compgen -W "${cmd_elements}" -- "$cur_cmd"))
}
