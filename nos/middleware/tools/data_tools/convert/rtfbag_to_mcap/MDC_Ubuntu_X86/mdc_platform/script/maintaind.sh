#!/bin/sh

set -e; set +e

ROOT_PATH="/opt/platform/mdc_platform"
RUN_DIR=$ROOT_PATH/bin
CONF_DIR=$ROOT_PATH/conf
MSG_PATH=$CONF_DIR/rtftools/msg
DATA_ROOT_DIR="/opt/usr/mnt"
BAG_DIR="${DATA_ROOT_DIR}/mbag"

function create_dir()
{
    mkdir -p "${BAG_DIR}" || { echo "failed to mkdir BAG_DIR";  return 1;}
    chown mdc:mdc "${BAG_DIR}" || { echo "failed to chown ${BAG_DIR}";  return 1;}
    chmod 750 "${BAG_DIR}" || { echo "failed to chmod ${BAG_DIR}";  return 1;}

    chown root:root "${DATA_ROOT_DIR}" || { echo "failed to chown ${DATA_ROOT_DIR}";  return 1;}
    chmod 755 "${DATA_ROOT_DIR}" || { echo "failed to chmod ${DATA_ROOT_DIR}";  return 1;}
    return 0
}

function main()
{
    create_dir

    app_name=maintaind
    runuser -s /bin/sh - mdc -c "${RUN_DIR}/${app_name} -p ${MSG_PATH} &"
    echo "${app_name} start, ret=$?."

    return 0
}

main "$@"
exit $?
