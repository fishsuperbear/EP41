#!/bin/bash

FILE_NAME=$1

CURR_PATH=$(cd "$(dirname ""$0"")"; pwd)
FASTDDS_GEN_TOOLS=${CURR_PATH}/../netaos_thirdparty/x86/fast-dds/bin/fastddsgen
FASTDDS_GEN_IDL_PATH=${CURR_PATH}/../middleware/idl/data_type/
FASTDDS_GEN_GENERATE_PATH=${CURR_PATH}/../middleware/idl/generated/

function SedIDLFile() {
    echo "======Sed idl : ${FILE_NAME} start.======"
    sed -i 's/int8_t/int8/g' $FILE_NAME
    sed -i 's/int16_t/int16/g' $FILE_NAME
    sed -i 's/int32_t/int32/g' $FILE_NAME
    sed -i 's/int64_t/int64/g' $FILE_NAME

    sed -i 's/uint8_t/uint8/g' $FILE_NAME
    sed -i 's/uint16_t/uint16/g' $FILE_NAME
    sed -i 's/uint32_t/uint32/g' $FILE_NAME
    sed -i 's/uint64_t/uint64/g' $FILE_NAME

    sed -i 's/float_t/float/g' $FILE_NAME
    sed -i 's/float32_t/float/g' $FILE_NAME
    sed -i 's/float64_t/double/g' $FILE_NAME

    sed -i 's/bool /boolean /g' $FILE_NAME

    sed -i 's/=.*[0-9]//g' $FILE_NAME
    sed -i 's/=.*false//g' $FILE_NAME
    sed -i 's/=.*true//g' $FILE_NAME

    sed -i 's/public AlgDataBase/CommonHeader/g' $FILE_NAME
    sed -i '/AlgHeader/d' $FILE_NAME

    sed -i 's/std::vector/sequence/g' $FILE_NAME
    sed -i 's/std::string/string/g' $FILE_NAME

    sed -i 's/>>/> >/g' $FILE_NAME
    echo "======Sed idl : ${FILE_NAME} end.======"
}

function GenerateTool() {
    echo "======Generate start.======"
    ${FASTDDS_GEN_TOOLS} -replace -cs -typeobject -I ${FASTDDS_GEN_IDL_PATH} -d ${FASTDDS_GEN_GENERATE_PATH} ${FILE_NAME}
    echo "======Generate end.======"
}

SedIDLFile
GenerateTool
