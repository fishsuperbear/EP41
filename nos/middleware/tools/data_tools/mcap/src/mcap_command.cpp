/* Copyright (c)  2023 Hozon New Energy Auto Co., Ltd.. All rights reserved.
 *
 * @File: mcap_command.cpp
 * @Date: 2023/08/01
 * @Author: cheng
 * @Desc: --
 */

#include <iostream>
#include <string>
#include <vector>
#include "argvparser.h"
#include "merge.h"
#include "split.h"
#include "filter.h"

using namespace argvparser;
using namespace hozon::netaos::mcap;

enum MethodType {
    merge = 1,
    split = 2,
    filter = 3,
};

enum SplitType {
    extract_attachment = 1,
};

struct McapMergeOption {
    std::vector<std::string> mcap_file_path_vec;
    std::vector<std::string> attachment_file_path_vec;
    std::string output_file_path;
};

struct McapSplitOption {
    std::vector<std::string> mcap_file_path_vec;
    SplitType split_type;
    std::vector<std::string> attachment_file_path_vec;
    std::string output_folder_path;
};

struct McapFilterOption {
    std::string mcap_file_path;
    std::vector<std::string> white_topic_vec;
    std::vector<std::string> black_topic_vec;
    std::string output_folder_path;
};

struct McapOption {
    bool show_help{false};
    MethodType method;
    McapMergeOption mcap_merge_option;
    McapSplitOption mcap_split_option;
    McapFilterOption mcap_filter_option;
};

std::string ConvertVector(std::vector<std::string> &vec) {
    std::stringstream ss;
    for (auto it = vec.begin(); it != vec.end(); it++) {
        if (it != vec.begin()) {
            ss << " ";
        }
        ss << *it;
    }
    return ss.str();
}

void McapTool(McapOption &mcap_option) {
    switch (mcap_option.method) {
        case merge: {
            Merge merge_tool;
            merge_tool.merge_mcap(mcap_option.mcap_merge_option.mcap_file_path_vec,
                                 mcap_option.mcap_merge_option.attachment_file_path_vec,
                                 mcap_option.mcap_merge_option.output_file_path);
            std::cout << "mcap merge finish, result in: "<< std::endl;
            for (std::string path : merge_tool.get_output_file_path_vec()) {
                std::cout << path << std::endl;
            }
            break;
        }
        case split: {
            Split split_tool;
            switch (mcap_option.mcap_split_option.split_type) {
                case extract_attachment: {
                    split_tool.extract_attachments(mcap_option.mcap_split_option.mcap_file_path_vec,
                                                  mcap_option.mcap_split_option.attachment_file_path_vec,
                                                  mcap_option.mcap_split_option.output_folder_path);
                    break;
                }
            }
            if (mcap_option.mcap_split_option.output_folder_path == "") {
                std::cout << "mcap split finish" << std::endl;
            } else {
                std::cout << "mcap split finish, result in: " << mcap_option.mcap_split_option.output_folder_path << std::endl;
            }
            break;
        }
        case filter: {
            if (mcap_option.mcap_filter_option.white_topic_vec.empty() &&
                mcap_option.mcap_filter_option.black_topic_vec.empty()) {
                std::cout << "mcap filter error: -wl or -bl must specify one" << std::endl;
                break;
            }
            Filter filter_tool;
            filter_tool.filter_base_topic_list(mcap_option.mcap_filter_option.mcap_file_path,
                                            mcap_option.mcap_filter_option.white_topic_vec,
                                            mcap_option.mcap_filter_option.black_topic_vec,
                                            mcap_option.mcap_filter_option.output_folder_path);
            std::cout << "mcap filter finish, result in: " << mcap_option.mcap_filter_option.output_folder_path << std::endl;
            break;
        }
    }
}

int main(int argc, char **argv) {
    McapOption mcap_option;
    auto merge_args =
        (command("merge").set(mcap_option.method, MethodType::merge) % "merge files to one mcap file",
            (option("-h", "--help").set(mcap_option.show_help, true)) % "print the help infomation",
            (option("-am", "--add-mcap") &
                values("mcap files", mcap_option.mcap_merge_option.mcap_file_path_vec)) %
                "optional paraments(0~n), merge mcap file list",
            (option("-at", "--add-attachment") &
                values("attachments", mcap_option.mcap_merge_option.attachment_file_path_vec)) %
                "optional paraments(0~n), merge attachment list",
            (required("-o", "--output-dirpath") &
                value("output dirpath", mcap_option.mcap_merge_option.output_file_path)) %
                "required parameters, specific the output file path, "
        );
    auto split_args =
        (command("split").set(mcap_option.method, MethodType::split) % "split mcap file to multiple files",
            (option("-h", "--help").set(mcap_option.show_help, true)) % "print the help infomation",
            (required("-m", "--mcap-file") &
                values("mcap files", mcap_option.mcap_split_option.mcap_file_path_vec)) %
                "required parameters(1~n), mcap file list that will be splited",
            (required("-ea", "--extract-attachment")
                .set(mcap_option.mcap_split_option.split_type, SplitType::extract_attachment)) %
                "required parameters, extrace attachment from mcap to file",
            (option("-st", "--split-topic") &
                values("mcap file", mcap_option.mcap_split_option.attachment_file_path_vec)) %
                "optional parameters(0~n), split attachment list, if this parameter is not specified, all attachments are extracted",
            (option("-o", "--output-dirpath") &
                value("output dirpath", mcap_option.mcap_split_option.output_folder_path)) %
                "optional parameters, specific the output folder path, "
        );
    auto filter_args =
        (command("filter").set(mcap_option.method, MethodType::filter) % "filter topics from mcap files",
            (option("-h", "--help").set(mcap_option.show_help, true)) % "print the help infomation",
            (required("-m", "--mcap-file") &
                values("mcap file", mcap_option.mcap_filter_option.mcap_file_path)) %
                "required parameters, mcap file that will be filtered",
            (option("-wl", "--white-list") &
                values("white list", mcap_option.mcap_filter_option.white_topic_vec)) %
                "optional paraments(0~n), topic white list",
            (option("-bl", "--black-list") &
                values("black list", mcap_option.mcap_filter_option.black_topic_vec)) %
                "optional paraments(0~n), topic black list",
            (required("-o", "--output-dirpath") &
                value("output dirpath", mcap_option.mcap_filter_option.output_folder_path)) %
                "required parameters, specific the output folder path"
        );
    auto mcap_tool_args = (merge_args | split_args | filter_args);
    auto cmd = parse(argc, argv, mcap_tool_args);
    if (!cmd || mcap_option.show_help) {
        switch (mcap_option.method) {
            case merge: {
                std::cout << make_man_page(merge_args, argv[0]) << '\n';
                return 0;
            }
            case split: {
                std::cout << make_man_page(split_args, argv[0]) << '\n';
                return 0;
            }
            case filter: {
                std::cout << make_man_page(filter_args, argv[0]) << '\n';
                return 0;
            }
            default: {
                std::cout << make_man_page(mcap_tool_args, argv[0]) << std::endl;
                return 0;
            }
        }
    } else {
        McapTool(mcap_option);
    }
}