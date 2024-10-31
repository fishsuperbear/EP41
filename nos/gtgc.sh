#!/bin/bash


DEV_PW="nvidia"
DEV_IP="root@10.6.75.186"
CURR_PATH=$(readlink -f "$(dirname "$0")")
UT_START_SCRIPT_PATH="${CURR_PATH}/ut_test/script/"
UT_REMOTE_GCDA_PATH_PRE="/opt/usr/ut_test/"
UT_REMOTE_GCDA_PATH_AFTER="/gcda/*"
UT_REMOTE_XML_PATH_AFTER="/*.xml"
UT_LOCAL_GCNO_PATH="${CURR_PATH}/build/ut/ut_test"
utFileName="${CURR_PATH}/output/orin/ut_test"

ut_arr=()

#################################################################
# error_info
# This Function will check the return value,if the value not equl 0
# we will exit the script
#################################################################
function error_info(){
    if [ $? -ne 0 ]
    then
        echo -e "\033[1;31m Error happend and exit.\033[0m"
        exit
    fi
}

# Copy All the ut files to mdc:/opt/ap/app
function copy_utimage_to_dev(){

    sshpass -p ${DEV_PW} scp -r $utFileName ${DEV_IP}:/opt/usr
    error_info
}


#################################################################
# run_deployment
# This Function will create ut/canstack_ut and ut/ethstack_ut floder
# call create_stack_ut function
#ut_test
#    ├── startall.sh
#    └── ut
#        └── module1
#            ├── xxx1_ut
#            │   ├── bin
#            │   └── start.sh
#            └── xxx2_ut 
#                ├── bin
#                └── start.sh
#################################################################
function run_deployment() {
    if [ ! -d "$utFileName" ]
    then
        mkdir -p "$utFileName"/ut
    else
        rm -rf "$utFileName"
        mkdir -p "$utFileName"/ut
    fi

    if [ -d "$utFileName/ut" ]
    then
        utName=$(ls "$CURR_PATH/ut_test/ut")
        a=0
        for i in $utName
        do
            if [ "$i" == "CMakeLists.txt" ]
            then
                echo "ut cmakelist file, jump!!"
            else
                if [ -f "$CURR_PATH/output/orin/bin/$i" ]
                then
                    ut_arr[$a]=$i
                    let a++
                    mkdir -p "$utFileName"/ut/"$i"/bin
                    cp -rf "$CURR_PATH"/output/orin/bin/"$i"  "$utFileName"/ut/"$i"/bin
                    cp -rf "$CURR_PATH"/ut_test/script/start.sh  "$utFileName"/ut/"$i"
                fi
            fi
        done
    else
        echo -e "\033[1;31m ERROR: Can't found $utFileName floder.\033[0m"
        error_info
        exit
    fi

    cp -rf "$UT_START_SCRIPT_PATH"/startall.sh "$utFileName"/
    error_info
    # create_stack_ut
    echo -e "\033[1;32m Create the $utFileName image success and Now Copy files to orin \033[0m"
    echo -e "\033[0;36m Please wait and have a good luck...\033[0m"
    copy_utimage_to_dev
    echo -e "\033[1;32m Copy finished! \033[0m"
}

#################################################################
# run_ut
# This Function will call the remote startall.sh and runing UnitTest
# path_num: The folder Path depth
#################################################################
function run_ut() {
    path_num=$(echo "$UT_LOCAL_GCNO_PATH" | grep -o '/'|wc -l)
    sshpass -p ${DEV_PW} ssh ${DEV_IP} "cd /opt/usr/ut_test; bash startall.sh $path_num"
    error_info
}

#################################################################
# run_report
# This Function will create finally UT report
#################################################################
function run_report() {
    rm -rf report
    mkdir -p report/gtest_report

    # create ut report
    for ((i=0; i<${#ut_arr[*]}; i++))
    do
        path1=${UT_REMOTE_GCDA_PATH_PRE}ut/${ut_arr[i]}${UT_REMOTE_GCDA_PATH_AFTER}
        path2=${UT_REMOTE_GCDA_PATH_PRE}ut/${ut_arr[i]}${UT_REMOTE_XML_PATH_AFTER}
        sshpass -p ${DEV_PW} scp -r ${DEV_IP}:"${path1}" "${UT_LOCAL_GCNO_PATH}"
        sshpass -p ${DEV_PW} scp ${DEV_IP}:"${path2}" "${CURR_PATH}"/report/gtest_report
        xsltproc gtest2html.xslt "${CURR_PATH}"/report/gtest_report/"${ut_arr[i]}".xml > "${CURR_PATH}"/report/gtest_report/"${ut_arr[i]}".html
        error_info
    done

    lcov --capture --directory . --output-file o.info --test-name ut
    lcov --remove o.info '9.3.0/*' '*/proto/*' '*/include/*' '*/generated/*' '*/ut_test/*' --output-file ut.info --test-name ut
    genhtml ut.info --branch-coverage --output-directory report/gcov_report --title "SoC App Coverage"
    error_info
    rm -f ut.info o.info
    # firefox report/gcov_report/index.html report/gtest_report/*.html
    firefox report/gcov_report/index.html
}


# this is the main function
function main(){
    if [ -d "$CURR_PATH/output/orin" ]
    then
        run_deployment
        error_info
        run_ut
        error_info
        run_report
        error_info
    else
        echo -e "\033[1;31m ERROR: output/orin not found \033[0m"
    fi
}

main