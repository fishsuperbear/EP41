function _lpm() {
    local cur_cmd=${COMP_WORDS[COMP_CWORD]}
    local prev_cmd=${COMP_WORDS[COMP_CWORD-1]}

    local cmd_elements=""

    case "${prev_cmd}" in
    "lpm" )
        if [ "${COMP_CWORD}" = "2" ]; then
            cmd_elements="help mock-cfg-proc set-enter-deep-sleep-time"
        fi
        ;;
    "mock-cfg-proc" )
        if [ "${COMP_CWORD}" = "3" ]; then
            cmd_elements="0 1"
        fi
        ;;
    "0" | "1" )
        if [ "${COMP_CWORD}" = "4" ]; then
            cmd_elements="0 1"
        fi
        ;;
    * )
        ;;
    esac

    COMPREPLY=($(compgen -W "${cmd_elements}" -- "$cur_cmd"))
}
