function _em() {
    local cur_cmd=${COMP_WORDS[COMP_CWORD]}
    local prev_cmd=${COMP_WORDS[COMP_CWORD-1]}
    local cmd_elements=""

    case "${prev_cmd}" in
    "em" )
        if [ "${COMP_CWORD}" = "2" ]; then
            cmd_elements="help query restart"
        fi
        ;;
    "query" )
        if [ "${COMP_CWORD}" = "3" ]; then
            cmd_elements="mdcPlatformStatus processStatus"
        fi
        ;;
    * )
        ;;
    esac

    COMPREPLY=($(compgen -W "${cmd_elements}" -- "$cur_cmd"))
}
