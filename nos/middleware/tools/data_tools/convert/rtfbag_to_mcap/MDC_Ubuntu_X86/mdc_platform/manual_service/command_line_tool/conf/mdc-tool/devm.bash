function _devm() {
    local cur_cmd=${COMP_WORDS[COMP_CWORD]}
    local prev_cmd=${COMP_WORDS[COMP_CWORD-1]}

    local cmd_elements=""

    case "${prev_cmd}" in
    "devm" )
        if [ "${COMP_CWORD}" = "2" ]; then
            cmd_elements="get-dev-cfg get-dev-info get-dev-state get-dev-statinfo get-boardinfo read-did set-ddr-ecc-enable get-ddr-ecc-enable get-cpu-cfg set-cpu-cfg
                get-compute-power get-ai-group-info  get-qos-config set-qos-config"
        fi
        ;;
    "get-dev-cfg" )
        if [ "${COMP_CWORD}" = "3" ]; then
            cmd_elements="3"
        fi
        ;;
    "get-dev-info" )
        if [ "${COMP_CWORD}" = "3" ]; then
            cmd_elements="Soc Mcu Cpld Lsw0"
        fi
        ;;
    "get-dev-state" )
        if [ "${COMP_CWORD}" = "3" ]; then
            cmd_elements="Soc Mcu Cpld Lsw0"
        fi
        ;;
    "get-dev-statinfo" )
        if [ "${COMP_CWORD}" = "3" ]; then
            cmd_elements="Soc Mcu Lsw0 Storage"
        fi
        ;;
    "get-boardinfo" )
        if [ "${COMP_CWORD}" = "3" ]; then
            cmd_elements="topo"
        fi
        ;;
    "read-did" )
        if [ "${COMP_CWORD}" = "3" ]; then
            cmd_elements="Soc"
        fi
        ;;
    "set-ddr-ecc-enable" )
        if [ "${COMP_CWORD}" = "3" ]; then
            cmd_elements="1(enable) 0(disable)"
        fi
        ;;
    * )
        ;;
    esac

    COMPREPLY=($(compgen -W "${cmd_elements}" -- "$cur_cmd"))
}
