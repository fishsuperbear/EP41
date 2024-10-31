#!/bin/sh
# fix check error according buildMC v2.3 rule 5.1.4.1
set -e; set +e

runDir=/opt/platform/mdc_platform/bin
confDir=/opt/platform/mdc_platform/conf

function main()
{
        slot_id=$(cat < /proc/cmdline | awk -F 'slotid=' '{print $2}'|cut -d' ' -f1|cut -b 2)
        #mini0
        if [ "${slot_id}" == "" ]; then
                SOMEIP_CONFIG_FILE=${confDir}/vsomeip-host.json SOMEIP_APP_NAME=someipd ${runDir}/someipd &
                echo " host: vsomeipd start! "
        fi
        return 0
}

main "$@"

exit $?
