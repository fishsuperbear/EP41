#!/bin/bash
# Self-starting file of EM

function main()
{
    export EM_LOG_TYPE=ARA_LOG
    export SET_MDC_AP_LOG_LEVEL=INFO

    /opt/platform/mdc_platform/bin/execution-manager -D -U mdc -R /opt/platform/mdc_platform/ \
        -S machine/ -F runtime_service/ &

    return 0
}

main "$@"

exit $?
