## 编译
  bash build.sh
  编译完成后，生成~/output/x86_2004/test/auto_test目录
  test
  └──auto_test
      ├── bin
      │   └── auto_test
      ├── Conf
      │   ├── uds
      │   │   ├── uds_request_test.json
      │   │   └── uds_request_test2.json
      │   ├── docan
      │   │   ├── cantp_test.json
      │   │   └── cantp_test2.json
      │   ├── doip
      │   │   ├── doip_socket_test.json
      │   │   └── doip_socket_test2.json
      └── lib
         ├── libjsoncpp.so
         └── libneta_diag_sa.so

## 目录说明
  ~/output/x86_2004/test/auto_test/bin 可执行文件
  ~/output/x86_2004/test/auto_test/Conf 用例配置文件
  ~/output/x86_2004/test/auto_test/lib 依赖库文件，测试人员制作json文件用于测试

## 执行文件
  运行文件 auto_test/bin/auto_test 后，顺序检索Conf目录中的各个json文件，输出执行结果，失败后停止

## 配置文件制作
  Conf/uds 存放所有的与应用层 iso14229 相关用例配置文件
  Conf/doip 存放用于测试 iso13400-2 相关用例配置文件
  Conf/docan 存放用于测试 iso15765-2 相关用例配置文件

