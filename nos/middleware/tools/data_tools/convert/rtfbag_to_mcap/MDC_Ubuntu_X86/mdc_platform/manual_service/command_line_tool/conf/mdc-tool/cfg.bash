function _cfg() {
    local cur_cmd=${COMP_WORDS[COMP_CWORD]}
    local prev_cmd=${COMP_WORDS[COMP_CWORD-1]}

    local cmd_elements=""

    case "${prev_cmd}" in
    "cfg" )
        if [ "${COMP_CWORD}" = "2" ]; then
            cmd_elements="set get del"
        fi
        ;;
    * )
        ;;
    esac

    COMPREPLY=($(compgen -W "${cmd_elements}" -- "$cur_cmd"))
}
