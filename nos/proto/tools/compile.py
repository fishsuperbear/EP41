#!/usr/bin/env python3
import argparse
import datetime
import os
import os.path as osp
import subprocess as sp
import shutil

from multiprocessing import cpu_count

def parse_args():
    p = argparse.ArgumentParser(description='compiling usage and options',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # 位置参数
    p.add_argument('platform', choices=['mdc', 'orin', 'x86', 'all'], help='choose platform')
    # 开关类型
    p.add_argument('--verbose', action='store_true', help='enable verbose mode for cmake')
    p.add_argument('--release', action='store_true', help='set CMAKE_BUILD_TYPE=Release, default Debug')
    p.add_argument('--ut', action='store_true', help='use unit test or not, default false')
    p.add_argument('--clean', action='store_true', help='remove cmake cache first, then build again')
    p.add_argument('--gcc', action='store_true', help='build lib with gcc,default llvm')
    p.add_argument('--make', action='store_true', help='directly execute make, will not compile again')
    # 默认值类型
    p.add_argument('--workspace', default=None, help='root of code repository')
    p.add_argument('-j', default=max(cpu_count() - 2, 1), dest="jobs", type=int, help='make -j')

    return p.parse_args()

# 执行shell命令
def execute_shell(cmd,shell=True,stdout=None):
    result = sp.run(cmd,shell=shell,stdout=stdout)
    return result.stdout

def LOG_INFO(*msg):
    print('\033[36m[INFO]', *msg)

def LOG_ERROR(*msg):
    print('\033[31m[ERROR]', *msg)

def set_env(var, value):
    if var in ['LD_LIBRARY_PATH', 'PATH']:
        if os.environ.get(var) is not None:
            os.environ[var] = value + ":" + os.environ[var]
        else:
            os.environ[var] = value
    else:
        os.environ[var] = value

def cmake_build(workspace, build_directory, cmake_args, jobs, verbose=False):
    """执行cmake 编译过程, 结束后切换 current working to workspace
    workspace: 代码仓根路径
    build_directory: cmake 编译生成目录， 不是编译后的产出目录
    cmake_args: cmake 编译选项设置  -DCMAKE_BUILD_TYPE=Release ..
    jobs: 指定编译使用的cpu数目 make -j8
    verbose: 打开make时的详细信息
    """
    date_s = datetime.datetime.now()
    # cmake ..
    os.makedirs(build_directory, exist_ok=True)
    os.chdir(build_directory)
    if not kwargs['make']:
        args_str = ' '.join([f'{key}={value}' for key, value in cmake_args.items()])
        cmd = 'cmake ' + args_str + ' ..'
        LOG_INFO(cmd)
        execute_shell(cmd)
    # make -j
    cmd = 'make -j{}'.format(jobs)
    if verbose:
        cmd += " VERBOSE=1"
    cmd += "&& make install"
    LOG_INFO(cmd)
    execute_shell(cmd)
    elapsed_time = datetime.datetime.now() - date_s
    LOG_INFO("build finished. elapsed_time:", str(elapsed_time))
    os.chdir(workspace)

def mdc_build(workspace, build_directory, release_directory, **kwargs):
    # download release package
    sp.run('bash tools/downloadPkg.sh', shell=1)

    # 设置环境变量
    if kwargs['gcc']:
        set_env('PATH', '/usr/local/mdc_sdk/dp_gea/mdc_cross_compiler/bin')
        set_env('CC', 'aarch64-target-linux-gnu-gcc')
        set_env('CXX', 'aarch64-target-linux-gnu-g++')
    else:
        set_env('PATH', '/usr/local/mdc_sdk_llvm/dp_gea/mdc_cross_compiler/bin')
        set_env('CC', 'clang')
        set_env('CXX', 'clang++')
        set_env('LD_LIBRARY_PATH', '/usr/local/mdc_sdk_llvm/dp_gea/mdc_cross_compiler/sysroot/usr/lib64')

    # 设置cmake编译选项
    args = dict()
    args['-DCMAKE_INSTALL_PREFIX'] = release_directory + "/mdc/"
    args['-DCMAKE_BUILD_TYPE'] = "Release" if kwargs['release'] else "Debug"
    args['-DPLATFORM_3RD'] = osp.join(workspace, 'third_party/third_party/arm')
    args['-DGLOBALPROTO_SINGLE_MODULE_COMPILE'] = 'ON'
    # args['-DENABLE_UT'] = int(kwargs['ut'])
    # args['-DUSE_GCC'] = "True" if kwargs['gcc'] else "False"
    cmake_build(workspace, build_directory, args, kwargs['jobs'], kwargs['verbose'])

def x86_build(workspace, build_directory, release_directory, **kwargs):
    """x86 编译流程"""
    print("x86 build")
    sp.run('bash tools/downloadPkg.sh', shell=1)
    set_env('PATH', '/usr/bin')
    set_env('CC', '/usr/bin/x86_64-linux-gnu-gcc')
    set_env('CXX', '/usr/bin/x86_64-linux-gnu-g++')

    set_env('LD_LIBRARY_PATH', osp.join(workspace, 'third_party/third_party/x86/protobuf/lib'))
    args = dict()
    args['-DCMAKE_INSTALL_PREFIX'] = release_directory+"/x86/"
    args['-DCMAKE_BUILD_TYPE'] = "Release" if kwargs['release'] else "Debug"
    args['-DPLATFORM_3RD'] = osp.join(workspace, 'third_party/third_party/x86')
    # args['-DUSE_GCC'] = "True" if kwargs['gcc'] else "False"
    # args['-DENABLE_UT'] = int(kwargs['ut'])
    args['-DGLOBALPROTO_SINGLE_MODULE_COMPILE'] = 'ON'
    cmake_build(workspace, build_directory, args, kwargs['jobs'], kwargs['verbose'])

def orin_build(workspace, build_directory, release_directory, **kwargs):
    """orin 编译流程"""
    set_env('PATH', '/usr/local/orin_sdk/aarch64/bin')
    set_env('CC', 'aarch64-linux-gcc')
    set_env('CXX', 'aarch64-linux-g++')
    LOG_INFO("Start update submodule")

    # download release package
    execute_shell('bash tools/downloadPkg.sh')
    # cmake param set
    args = dict()
    args['-DCMAKE_INSTALL_PREFIX'] = release_directory+"/orin/"
    args['-DCMAKE_BUILD_TYPE'] = "Release" if kwargs['release'] else "Debug"
    args['-DPLATFORM_3RD'] = osp.join(workspace, 'third_party/third_party/orin')
    # args['-DENABLE_UT'] = 'FLASE' if not kwargs['ut'] else 'TRUE'
    args['-DGLOBALPROTO_SINGLE_MODULE_COMPILE'] = 'ON'
    cmake_build(workspace, build_directory, args, kwargs['jobs'], kwargs['verbose'])

def copy_header_files_by_path(src_path, dst_path):
    header_files = []
    for root, dirs, files in os.walk(src_path):
        dirs[:] = [d for d in dirs if d not in ['third_party', 'release', 'Debug', 'Release', 'build', 'test']]  # 排除文件夹
        for file in files:
            if file.endswith('.h') or file.endswith('.hpp'):
                header_files.append(os.path.join(root, file))

    for file in header_files:
        file_path_in_dst_folder = os.path.join(dst_path, os.path.relpath(file, start=src_path))
        if not os.path.exists(os.path.dirname(file_path_in_dst_folder)):
            os.makedirs(os.path.dirname(file_path_in_dst_folder))
        shutil.copy(file, file_path_in_dst_folder)

def find_header_files_in_directory(dir_path):
    header_files = []
    for entry in os.scandir(dir_path):
        if entry.is_dir(follow_symlinks=False):
            header_files += find_header_files_in_directory(entry.path)
        elif entry.is_file(follow_symlinks=False):
            if entry.name.endswith('.h') or entry.name.endswith('.hpp'):
                header_files.append(entry.path)
    return header_files

def delete_empty_dir(dir):
    dir_list = []
    for root,dirs,files in os.walk(dir):
        dir_list.append(root)
    # 先生成文件夹的列表，重点是下边
    for root in dir_list[::-1]:
        if not os.listdir(root):
            os.rmdir(root)

def start_copy_head(platform):
    # 开始copy 头文件
    if platform=="x86":
        head_src_path = os.getcwd()
        LOG_INFO(head_src_path)
        head_target_path = release_directory+"/x86/include/"
        LOG_INFO(head_target_path)
        copy_header_files_by_path(head_src_path, head_target_path)
    elif platform=="mdc":
        head_src_path = os.getcwd()
        head_target_path = release_directory+"/mdc/include/"
        copy_header_files_by_path(head_src_path, head_target_path)
    elif platform=="orin":
        head_src_path = os.getcwd()
        head_target_path = release_directory+"/orin/include/"
        LOG_INFO(head_target_path)
        copy_header_files_by_path(head_src_path, head_target_path)

def copy_files_with_extension(source_dir, target_dir, extension):
    # 创建目标目录，如果不存在的话
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    # 遍历源目录中的文件
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.endswith(extension):
                source_file = os.path.join(root, file)
                target_file = os.path.join(target_dir, file)
                # 拷贝文件到目标目录
                shutil.copy2(source_file, target_file)
                print(f"已拷贝文件: {source_file} 到 {target_file}")

def install_sub_repo_lib(workspace, platform, release_directory):
    copy_files_with_extension(workspace+"/third_party/high_performace_compute_library/lib/"+platform, release_directory+"/"+platform+"/lib/", ".so")

def make_package(platform):
    start_copy_head(platform)

def clean(workspace):
    build_list=["Debug", "Release", "release", "build", "output"]
    for dir in build_list:
        path=os.path.join(workspace,dir)
        if os.path.exists(path):
            execute_shell("rm -r {}".format(path))
    # 移除proto生成的.h和.cc文件
    execute_shell("find . -name *.pb.h | grep -v 'third_party/third_party' |xargs rm -rf")
    execute_shell("find . -name *.pb.cc | grep -v 'third_party/third_party' |xargs rm -rf")
    execute_shell("find . -name *.om | grep -v 'third_party/third_party' |xargs rm -rf")
    LOG_INFO("delete cmake build production success.")

def all_build(workspace, build_directory, release_directory, **kwargs):
    sp.run('rm -rf {}'.format(build_directory), shell=1)
    x86_build(workspace, build_directory, release_directory, **kwargs)
    start_copy_head('x86')

    sp.run('rm -rf {}'.format(build_directory), shell=1)
    mdc_build(workspace, build_directory, release_directory, **kwargs)
    start_copy_head('mdc')

    sp.run('rm -rf {}'.format(build_directory), shell=1)
    orin_build(workspace, build_directory, release_directory, **kwargs)
    start_copy_head('orin')

if __name__ == '__main__':
    # 解析编译选项为dict
    kwargs = vars(parse_args())
    LOG_INFO('compile option:', kwargs)
    workspace = kwargs.pop('workspace')
    os.chdir(workspace)
    build_directory = osp.join(workspace, 'build')
    release_directory = osp.join(workspace, 'release')
    platform = kwargs.pop('platform')
    sp.run('git submodule update --init', shell=True)
    # sp.run('git submodule update --remote', shell=True)

    # 清理cmake编译缓存
    if kwargs['clean']:
        clean(workspace)

    # 开始构建工程
    if platform == 'mdc':
        mdc_build(workspace, build_directory, release_directory, **kwargs)
        # 开始copy 头文件
        make_package(platform)
    elif platform == 'x86':
        x86_build(workspace, build_directory, release_directory, **kwargs)
        make_package(platform)
    elif platform == 'orin':
        orin_build(workspace, build_directory, release_directory, **kwargs)
        make_package(platform)
    elif platform == 'all':
        all_build(workspace, build_directory, release_directory, **kwargs)

    install_sub_repo_lib(workspace, platform, release_directory)