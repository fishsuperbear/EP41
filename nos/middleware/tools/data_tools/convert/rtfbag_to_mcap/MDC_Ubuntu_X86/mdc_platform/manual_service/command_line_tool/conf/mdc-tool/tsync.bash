function _tsync() {
    local cur_cmd=${COMP_WORDS[COMP_CWORD]}
    local prev_cmd=${COMP_WORDS[COMP_CWORD-1]}
    local prev_two_cmd=${COMP_WORDS[COMP_CWORD-2]}

    local cmd_elements=""
   
    case "${prev_cmd}" in
    "tsync" )
        if [ "${COMP_CWORD}" = "2" ] ; then
            cmd_elements="DP MP DP_AND_MP DPS MPS"
        fi
        ;;
    "DP" | "MP" )
        if [ "${COMP_CWORD}" = "3" ] ; then
            cmd_elements="-g -s"
        fi
        ;;
    "DP_AND_MP" | "DPS" | "MPS" )
        if [ "${COMP_CWORD}" = "3" ] ; then
            cmd_elements="-g"
        fi
        ;;

    * )
        case "${prev_two_cmd}" in
        "DP" | "MP" )
            local t_sec=${COMP_WORDS[4]}
            local t_nsec=${COMP_WORDS[5]}
            cmd_elements="${t_sec} ${t_nsec}"
            ;;
        esac

        ;;
    esac

    COMPREPLY=( $( compgen -W "${cmd_elements}" -- "$cur_cmd" ) )
}
