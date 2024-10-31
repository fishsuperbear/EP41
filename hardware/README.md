

### build

##### 1. build on docker with sh script

> ./build.sh

##### 1. build on board with sh script

> ./build.sh board

##### 3. hardware产出并提交到netaos仓库

> build完成后，hardware所有对外提供的 **头文件**、**库文件**、**main程序** 都会 `install` 到 `out/hardware` 目录下

```shell
root@6.0.6.0-0004-build-linux-sdk:/home/nvidia/wangjinfeng/work/hardware# tree out/hardware/ -L 2
out/hardware/
|-- bin
|   `-- camera_hpp_main
|-- include
|   |-- camera
|   |-- devices
|   |-- halnode
|   |-- hardware.h
|   |-- hw_hal_api.h
|   |-- hw_tag_and_version.h
|   `-- platform
`-- lib
    `-- camera

8 directories, 4 files
```

###### 将hardware产出拷贝到netaos仓库并提交

```shell
cp -r out/hardware ${netaos_path}/
cd ${netaos_path}/
git add hardware
git commit 
git push
```
##### 4.编译注意事项

1. 编译产出为交叉编译产物，只支持 arm 架构，不支持 x86 产出
2. 当前使用整体cmake配置，不支持cd到单个子项目路径下编译。如果想使用命令行执行cmake编译，请按照如下顺序执行
```shell
mkdir out
mkdir out/build out/hardware
cd out/build

cmake -DCMAKE_INSTALL_PREFIX=../hardware ../.. 
# 默认在docker上编译，如果是在板子上编译，请执行 
# cmake -DCMAKE_INSTALL_PREFIX=../hardware -DBUILD_ON_DOCKER=OFF ../.. 

make -j8 
make install

或者可以编译单独子项目,例如
make hw_nvmedia_multiipc_main -j8
```