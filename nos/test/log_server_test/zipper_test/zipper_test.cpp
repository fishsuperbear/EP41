#include <thread>
#include <stdio.h>
#include <iostream>
#include <zipper.h>

#include <signal.h>


// case 01 不会删除 被压缩 的文件

// case 02 zipfile.add(filename); 执行前终止程序，只会产生文件，但实际是错误的zip格式

// case 03 zipfile.add(filename); 前 删除 被压缩的文件，会怎么样？ 产生正常格式的zip文件，但是里面没有内容

// test_123.log_bk 可以直接用于压缩


int main(int argc, char * argv[])
{
    std::string file_path_ = {"/home/xiaoyu/work/netaos/nos/output/x86_2004/bin/"};

    std::string filename = {"/home/xiaoyu/work/netaos/nos/output/x86_2004/bin/test_123.log_bk"};


    zipper::Zipper zipfile(file_path_ + "tmp_app.zip");
    std::this_thread::sleep_for(std::chrono::seconds(2));


	zipfile.add(filename);
    std::this_thread::sleep_for(std::chrono::seconds(2));

	zipfile.close();

    return 0;
}