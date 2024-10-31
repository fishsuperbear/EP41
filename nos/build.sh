#!/bin/bash

set -e

ROOT_PATH=$(cd "$(dirname ""$0"")"; pwd)
TARGET_PLATFORM=x86_2004
BUILD_TYPE=release
BUILD_WITH_IDL=true
BUILD_UT=false
CPPCHECK_LINT_MODULE_DIR=()
CLANG_TIDY_MODULE_DIR=()

CORE_NUM=8

function Usage() {
    echo "-h: show help info"
    echo "-p: [x86] [j5] [orin] specify platform, default is x86."
    echo "-t: [debug] [release] specify build type, default is release"
    echo "-k: [all] [dir] cpplint/cppcheck"
    echo "-y: [all] [dir] clang-tidy"
}

function Build() {
    module=$1

    mkdir -p build/$TARGET_PLATFORM
    mkdir -p output

    # Set library find path for netaos_thirdparty/x86/protobuf/bin/protoc
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${ROOT_PATH}/netaos_thirdparty/x86_2004/protobuf/lib"
    echo "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH}"

    echo "build on [$TARGET_PLATFORM]"
    if [ $TARGET_PLATFORM == "x86_2004" ]; then
        cmake $module -B build/$TARGET_PLATFORM/$module -DCMAKE_INSTALL_PREFIX=./output -DTARGET_PLATFORM="x86_2004" -DBUILD_TYPE="$BUILD_TYPE" -DBUILD_WITH_IDL=$BUILD_WITH_IDL
    elif [ $TARGET_PLATFORM == "j5" ]; then
        cmake $module -B build/$TARGET_PLATFORM/$module -DCMAKE_INSTALL_PREFIX=./output -DTARGET_PLATFORM="j5" -DBUILD_TYPE="$BUILD_TYPE" -DCMAKE_TOOLCHAIN_FILE="${ROOT_PATH}/cmake/j5-toolchain.cmake" -DBUILD_WITH_IDL=$BUILD_WITH_IDL
    elif [ $TARGET_PLATFORM == "mdc" ]; then
        cmake $module -B build/$TARGET_PLATFORM/$module -DCMAKE_INSTALL_PREFIX=./output -DTARGET_PLATFORM="mdc" -DBUILD_TYPE="$BUILD_TYPE" -DCMAKE_TOOLCHAIN_FILE="${ROOT_PATH}/cmake/mdc-toolchain.cmake" -DBUILD_WITH_IDL=$BUILD_WITH_IDL
    elif [ $TARGET_PLATFORM == "mdc-llvm" ]; then
        cmake $module -B build/$TARGET_PLATFORM/$module -DCMAKE_INSTALL_PREFIX=./output -DTARGET_PLATFORM="mdc-llvm" -DBUILD_TYPE="$BUILD_TYPE" -DCMAKE_TOOLCHAIN_FILE="${ROOT_PATH}/cmake/mdc-llvm-toolchain.cmake" -DBUILD_WITH_IDL=$BUILD_WITH_IDL
    elif [ $TARGET_PLATFORM == "orin" ]; then
        cmake $module -B build/$TARGET_PLATFORM/$module -DCMAKE_INSTALL_PREFIX=./output -DTARGET_PLATFORM="orin" -DBUILD_TYPE="$BUILD_TYPE" -DCMAKE_TOOLCHAIN_FILE="${ROOT_PATH}/cmake/orin-toolchain.cmake" -DBUILD_WITH_IDL=$BUILD_WITH_IDL
    elif [ $TARGET_PLATFORM == "ut" ]; then
        cmake $module -B build/$TARGET_PLATFORM/$module -DCMAKE_INSTALL_PREFIX=./output -DTARGET_PLATFORM="orin" -DBUILD_TYPE="$BUILD_TYPE" -DCMAKE_TOOLCHAIN_FILE="${ROOT_PATH}/cmake/orin-toolchain.cmake" -DBUILD_WITH_IDL=$BUILD_WITH_IDL
    else
        echo "TARGET_PLATFORM [$TARGET_PLATFORM] not supported."
        exit -1
    fi

    cd build/$TARGET_PLATFORM/$module
    make -j${CORE_NUM}
    make install
    cd -
}

function Clean() {
    echo "clean all"
    rm build -rf
    rm output -rf
    rm report -rf
    rm middleware/idl/generated -rf
    rm service/idl/generated -rf
    find proto/ -name *.pb.* -exec rm -rf {} \;
}

function UnitTestCreate() {
    if [ $TARGET_PLATFORM == "orin" ]; then
        if [ -d auto_test/nos/cases/ ]; then
            cp -r output/$TARGET_PLATFORM/test/unit_test/* auto_test/nos/cases/
            # 拷贝adflite测试用例
            mkdir -p auto_test/nos/cases/adflite_test/adf_lite_test_common
            cp -r output/$TARGET_PLATFORM/test/emproc_adf_test  auto_test/nos/cases/adflite_test/adf_lite_test_common
        fi
    fi
}

function TestCreate() {
    cd output/$TARGET_PLATFORM/
    currentpath=$(pwd)
    echo "$currentpath"

    mkdir -p scripts

    cd ../../
    cp -r test/sample output/$TARGET_PLATFORM/test/
    if [ "$BUILD_UT" == "true" ]; then
        UnitTestCreate
    fi
}

function PostProcess() {
    cp version.json output/$TARGET_PLATFORM/
    cp  -rfp ./scripts/* ./output/$TARGET_PLATFORM/scripts/

}

function UnitTest {
    if [ -f gtgc.sh ]
    then
        cd "${PROJECT_FOLDER}"
        bash gtgc.sh
    else
        echo -e "\033[1;31m ERROR: Can't found UniTest script exsit.\033[0m"
    fi
}


while getopts "p:t:k:y:n:hcsue" arg; do
    case $arg in
        h)
            Usage
            exit 0
            ;;
        p)
            TARGET_PLATFORM=$OPTARG
            echo "set TARGET_PLATFORM to [$TARGET_PLATFORM]"
            ;;

        t)
            BUILD_TYPE=$OPTARG
            echo "set BUILD_TYPE to [$BUILD_TYPE]"
            ;;
        c)
            Clean
            exit 0
            ;;
        u)
            BUILD_UT=true
            echo "build unit_test and copy to dragon"
            ;;
        s)
            BUILD_WITH_IDL=false
            ;;
        k)
            CPPCHECK_LINT_MODULE_DIR+=("$OPTARG")
            ;;
        y)
            CLANG_TIDY_MODULE_DIR+=("$OPTARG")
            ;;
        n)
            CORE_NUM=$OPTARG
            echo "set CORE_NUM to [$CORE_NUM]"
            ;;
    esac
done

if [ ${#CPPCHECK_LINT_MODULE_DIR[@]} -ne 0 ]; then
    if [ "$CPPCHECK_LINT_MODULE_DIR" == "all" ]; then
        checkdir="middleware service"
    else
        for arg in "${CPPCHECK_LINT_MODULE_DIR[@]}"; do checkdir+="$arg "; done
    fi
    echo "---------------------------------------------------------cppcheck=$checkdir---------------------------------------------------------"
    cppcheck --enable=all  -i middleware/idl  -i middleware/tools -i service/idl -i middleware/core  -i middleware/per/proto $checkdir
    echo "---------------------------------------------------------cpplint=$checkdir----------------------------------------------------------"
    cpplint --filter=-.* --linelength=250   --recursive --exclude=middleware/idl  --exclude=middleware/tools --exclude=service/idl --exclude=middleware/core --exclude=middleware/per/proto $checkdir
    echo "https://google.github.io/styleguide/cppguide.html"
    exit 0
fi
if [ ${#CLANG_TIDY_MODULE_DIR[@]} -ne 0 ]; then
    if [ "$CLANG_TIDY_MODULE_DIR" == "all" ]; then
        checkdir="middleware service"
    else
        for arg in "${CLANG_TIDY_MODULE_DIR[@]}"; do checkdir+="$arg "; done
    fi
    # --fix
    echo "---------------------------------------------------------clang-tidy=$checkdir---------------------------------------------------------"
    CHECKS=-*,performance-*,bugprone-*,portability-*,modernize-*,abseil-*,boost-*,clang-analyzer-*,concurrency-*,cppcoreguidelines-*,google-*,misc-*,readability-*,
    find $checkdir \( -path "middleware/idl" -o -path "middleware/tools" -o -path "service/idl" -o -path "middleware/core" -o -path "middleware/per/proto" \) -prune -o \( -name "*.cpp" -or -name "*.hpp" -or -name "*.h" \) -exec clang-tidy -checks="$CHECKS"  {} \;
    exit 0
fi

$ROOT_PATH/fetch.sh
Build .

Build test
TestCreate
PostProcess

$ROOT_PATH/release.sh $TARGET_PLATFORM
if [ "$TARGET_PLATFORM" == "ut" ]; then
    echo "curr bulid ut"
    Build ut_test
    UnitTest
fi
