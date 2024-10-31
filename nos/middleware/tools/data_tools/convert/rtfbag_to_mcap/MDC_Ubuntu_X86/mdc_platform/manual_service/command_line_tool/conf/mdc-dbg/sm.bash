function _sm() {
    local cur_cmd=${COMP_WORDS[COMP_CWORD]}
    local prev_cmd=${COMP_WORDS[COMP_CWORD-1]}
    local prev2_cmd=${COMP_WORDS[COMP_CWORD-2]}

    local cmd_elements=""

    case "${prev_cmd}" in
    "sm" )
        if [ "${COMP_CWORD}" = "2" ]; then
            cmd_elements="help get query request reset query_platform_state request_platform_state reset_recovery_count"
        fi
        ;;
    "get" )
        if [ "${COMP_CWORD}" = "3" ]; then
            cmd_elements="functionGroupList functionGroupInfo"
        fi
        ;;
    "request_platform_state" )
        if [ "${COMP_CWORD}" = "3" ]; then
            cmd_elements="Working Upgrade Reset"
        fi
        ;;
    "query" | "request" )
        if [ "${COMP_CWORD}" = "3" ]; then
            cmd_elements="MachineState Access"
        fi
        ;;
    "reset" )
        if [ "${COMP_CWORD}" = "3" ]; then
            cmd_elements="soft hard"
        fi
        ;;
    "soft" | "hard" )
        if [ "${COMP_CWORD}" = "4" ]; then
            cmd_elements="0 1"
        fi
        ;;
    "MachineState" )
        if [ "${COMP_CWORD}" = "4" ] && [ "${prev2_cmd}" = "request" ] ; then
            cmd_elements="Reset Restart Startup Standby Shutdown Update Verify"
        fi
        ;;
    "Access" )
        if [ "${COMP_CWORD}" = "4" ] && [ "${prev2_cmd}" = "request" ] ; then
            cmd_elements="Off Running"
        fi
        ;;
    * )
        ;;
    esac

    COMPREPLY=($(compgen -W "${cmd_elements}" -- "$cur_cmd"))
}
