#!/bin/bash
if [ "$#" -eq 0 ]; then
  echo "Please specify the directory that needs to be managed"
  exit 1
fi
#conf不存在或者conf存在type字段不存在或者type在预期外 按照(maxFolderNumDefault)和(maxFilesNumDefault,maxFileSizeDefault,maxDaysDefault)管理文件和目录，
#conf存在且type存在按照type判断是否按照(maxFolderNum,maxFolderSize)管理目录或者(maxFilesNum,maxFileSize,maxDays)管理文件,配置参数不存在时用Default值
currentdir=$1
configFileName="config.txt"
maxFilesNumDefault=210
maxFileSizeDefault=$(expr 10000 \* 1024 \* 1024)
maxDaysDefault=365
maxFolderNumDefault=50
maxFolderSizeDefault=$(expr 10000 \* 1024 \* 1024)
typeDefault="none" #"folder" "file" "none"
traverse_directory() {
    local directory="$1"
    file_count=$(ls -l "$1" | grep "^-" | wc -l)
    echo "dir $directory has $file_count files"
    maxFilesNum=$maxFilesNumDefault
    maxFileSize=$maxFileSizeDefault
    maxDays=$maxDaysDefault
    maxFolderNum=$maxFolderNumDefault
    maxFolderSize=$maxFolderSizeDefault
    type=$typeDefault
    # 1 查看当下目录下是否有配置文件
    if test -e "$directory/$configFileName"; then
        source $directory/$configFileName
        if [ "$type" != "folder" ] && [ "$type" != "file" ]; then
            maxFilesNum=$maxFilesNumDefault
            maxFileSize=$maxFileSizeDefault
            maxDays=$maxDaysDefault
            maxFolderNum=$maxFolderNumDefault
            maxFolderSize=$maxFolderSizeDefault
            type=$typeDefault
        else
            maxFilesNum=$((maxFilesNum == -1 ? maxFilesNumDefault : maxFilesNum))
            maxFileSize=$((maxFileSize == -1 ? maxFileSizeDefault : maxFileSize))
            maxDays=$((maxDays == -1 ? maxDaysDefault : maxDays))
            maxFolderNum=$((maxFolderNum == -1 ? maxFolderNumDefault : maxFolderNum))
            maxFolderSize=$((maxFolderSize == -1 ? maxFolderSizeDefault : maxFolderSize))
        fi
        echo "config exist: $directory/$configFileName" 
    else 
        echo "$directory config not exist"
    fi
    work_loop $directory $maxFilesNum $maxFileSize $maxDays $maxFolderNum $maxFolderSize $type
    # for file in "$directory"/*; do
	#     if [[ -d "$file" ]]; then
    #         traverse_directory "$file"
    #     fi
    # done
}
work_loop() {
    echo "directory is $1 maxFiles is $2  maxFileSize is :$3 maxDays is :$4  maxFolderNum is :$5 maxFolderSize is :$6 type is :$7"
    file_list=()
    folder_list=()
    if [ "$7" = "none" ] || [ "$7" = "file" ]; then
        # 2 如果超过文件更新周期， 则删除文件 除去config.txt //maxDays
        if [ "$4" -eq 0 ]; then
            #find $1 -maxdepth 1 -type f  -not -name $configFileName  -print -delete
        else
            count=$4
            ((count--))
            #find $1 -maxdepth 1 -type f  -not -name $configFileName -mtime +$count -print -delete
        fi
    fi
    for file in $(ls --hide=$configFileName -t $1 | tail -n +1 | tac)
    do
        relativefile=$1/$file
        if [ -f "$relativefile" ]; then
            file_list+=("$relativefile")
        elif [ -d "$relativefile" ]; then
            folder_list+=("$relativefile")
        fi
    done
    if [ "$7" = "none" ] || [ "$7" = "file" ]; then
        # 3 如果超过文件个数限制， 则删除文件 除去config.txt //maxFilesNum
        delfilecount=$((${#file_list[@]}-$2))
        if [ $delfilecount -gt 0 ]; then
            echo "need delfilecount:$delfilecount"
            for ((i=0; i<delfilecount; i++))
            do
                delete_file ${file_list[i]}
                unset file_list[i]
            done
            file_list=("${file_list[@]}")
        fi
        # 4 如果超过文件size限制， 则删除文件 除去config.txt //maxFileSize
        for ((i=0; i<${#file_list[@]}; i++))
        do
            size=$(ls -l "$1" | grep "^-" | grep -v "config.txt" | awk '{print $5}' | awk '{sum += $1} END {print sum}')
            echo "current dir $1  size  $size bytes"
            if [ $size -gt $3 ]; then
                delete_file ${file_list[i]}
            else 
                break
            fi
        done
    fi   
    if [ "$7" = "none" ] || [ "$7" = "folder" ]; then
        # 5 如果超过文件夹个数限制， 则删除文件夹 //maxFolderNum
        delfoldercount=$((${#folder_list[@]}-$5))
        if [ $delfoldercount -gt 0 ]; then
            echo "need delfoldercount:$maxFolderNum"
            for ((i=0; i<delfoldercount; i++))
            do
                delete_folder ${folder_list[i]}
                unset folder_list[i]
            done
            folder_list=("${folder_list[@]}")
        fi
        if [ "$7" = "folder" ]; then
            # 6 如果超过目录size限制， 则删除文件目录 //maxFileSize
            for ((i=0; i<${#folder_list[@]}; i++))
            do
                size=$(du -s "$1"| cut -f1)
                echo "current dir $1  size  $size bytes"
                if [ $size -gt $6 ]; then
                    delete_folder ${folder_list[i]}
                else 
                    break
                fi
            done
        fi
    fi
} 
delete_file() {
    echo " delete file is  $1"
    if [ -f "$1" ]; then
        rm "$1"
        echo "file:$1 delete suc"
    else
        echo "file:$1 not exist"
    fi
}
delete_folder() {
    echo " delete folder is  $1"
    if [ -d "$1" ]; then
        rm -rf "$1"
        echo "folder:$1 delete suc"
    else
        echo "folder:$1 not exist"
    fi
}

echo "$currentdir: + $(find $currentdir -mindepth 1 -type d | wc -l)   get_file_count_recursion: $(find $currentdir -mindepth 1 -type f | wc -l)"
traverse_directory $currentdir
echo "$currentdir: - $(find $currentdir -mindepth 1 -type d | wc -l)   get_file_count_recursion: $(find $currentdir -mindepth 1 -type f | wc -l)"
