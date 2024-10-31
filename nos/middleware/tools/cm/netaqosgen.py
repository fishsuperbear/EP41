#!/usr/bin/python
import os
import json
import sys
import getopt
import copy

Version_ID = "V1.0"

_json_file = "default_qos.json"

domain_args = []
topic_args = {
    "domain" : 0,
    "writer" : [],
    "reader" : []
}

participant_info = {
    "participant" : []
}

domain_info = {
    "name": "participant",
    "domain" : 0,
    "transport" : "",
    "discover": "",
    "datawriter" : [],
    "datareader" : []
}

transport_info = {
    "use_builtin_transports": False,
    "udp": {
        "enable": False,
        "network": "127.0.0.1",
        "send_socket_buffer_size": 2097152,
        "listen_socket_buffer_size": 2097152
    },
    "shm": {
        "enable": True,
        "segment_size": 20971520,
        "max_message_size": 10485760
    }
}

discover_info = {
    "typelookup_client" : True,
    "typelookup_server" : True,
    "leaseDuration": 5,
    "leaseDuration_announce_period": 1,
    "initial_announce_count": 100,
    "initial_announce_period": 100
}

reader_info = {
    "topic" : "DefaultTopic",
    "reliability" : "BEST_EFFORT",
    "durability" : "TRANSIENT_LOCAL_DURABILITY_QOS",
    "endpoint" : {
        "history_memory_policy" : "PREALLOCATED_WITH_REALLOC_MEMORY_MODE"
    },
    "history" : {
        "kind" : "KEEP_LAST",
        "depth" : 5
    },
    "data_sharing" : "AUTO"
}

writer_info = {
    "topic" : "DefaultTopic",
    "reliability" : "RELIABLE",
    "durability" : "TRANSIENT_LOCAL_DURABILITY_QOS",
    "endpoint" : {
        "history_memory_policy" : "PREALLOCATED_WITH_REALLOC_MEMORY_MODE"
    },
    "history" : {
        "kind" : "KEEP_LAST",
        "depth" : 5
    },
    "data_sharing" : "AUTO"
}

def PrintJsonToConsole(data):
    json_str = json.dumps(data, indent=4, ensure_ascii=False, sort_keys=False)
    print(json_str)

def WriteJsonToFile(file, json_data):
    with open(file, 'w') as write_f:
        json.dump(json_data, write_f, indent=4, ensure_ascii=False, sort_keys=False)
        write_f.close()

def ReadJsonFile(file_path):
    with open(file_path, 'r') as read_f:
        readtext = json.load(read_f)
        read_f.close()
        return readtext

def GetReaderInfo(topic_list):
    value = []
    for topic in topic_list:
        reader_ = copy.deepcopy(reader_info)
        reader_["topic"] = topic
        value.append(copy.deepcopy(reader_))
    return value

def GetWriterInfo(topic_list):
    value = []
    for topic in topic_list:
        writer_ = copy.deepcopy(writer_info)
        writer_["topic"] = topic
        value.append(copy.deepcopy(writer_))
    return value

def GetDomainInfo(domain_list):
    value = []
    for domain in domain_list:
        domain_ = copy.deepcopy(domain_info)
        domain_["domain"] = int(domain["domain"])
        domain_["transport"] = copy.deepcopy(transport_info)
        domain_["discover"] = copy.deepcopy(discover_info)
        domain_["datawriter"] = GetWriterInfo(domain["writer"])
        domain_["datareader"] = GetReaderInfo(domain["reader"])
        value.append(copy.deepcopy(domain_))
    return value

def GetParticipantInfo(domain_list):
    value = copy.deepcopy(participant_info)
    value["participant"] = GetDomainInfo(domain_list)
    return value

def GenerateJsonQosFile(file):
    participant_ = GetParticipantInfo(domain_args)
    WriteJsonToFile(file, participant_)

def UpdateWriterReaderInfo(domain, domain_arg):
    domain_tmp = copy.deepcopy(domain)
    for reader in domain["datareader"]:
        if reader["topic"] in domain_arg["reader"]:
            domain_arg["reader"].remove(reader["topic"])
    for writer in domain["datawriter"]:
        if writer["topic"] in domain_arg["writer"]:
            domain_arg["writer"].remove(writer["topic"])
    domain_tmp["datareader"].extend(GetReaderInfo(domain_arg["reader"]))
    domain_tmp["datawriter"].extend(GetWriterInfo(domain_arg["writer"]))
    # domain_args.remove(domain_arg)
    return domain_tmp

def UpdateDomainInfo(read_text):
    domain_list = []
    domain_arg_tmp = copy.deepcopy(domain_args)
    for domain in read_text["participant"]:
        domain_tmp = copy.deepcopy(domain)
        for domain_arg in domain_arg_tmp:
            if domain["domain"] == int(domain_arg["domain"]):
                domain_tmp = UpdateWriterReaderInfo(domain, domain_arg)
                domain_tmp["transport"] = copy.deepcopy(transport_info)
                domain_tmp["discover"] = copy.deepcopy(discover_info)
                domain_arg_tmp.remove(domain_arg)
        domain_list.append(domain_tmp)
    if domain_arg_tmp:
        domain_list.append(GetDomainInfo(domain_arg_tmp))
    return domain_list

def UpdateParticipantInfo(read_text):
    partic_tmp = copy.deepcopy(participant_info)
    partic_tmp["participant"].extend(UpdateDomainInfo(read_text))
    return partic_tmp

def UpdateJsonQosFile(file):
    read_text = ReadJsonFile(file)
    update_json_data = UpdateParticipantInfo(read_text)
    WriteJsonToFile(file, update_json_data)

def PrintTopicInfo():
    print(" ")
    print("[Modify domain topic info :]")
    for domain in domain_args:
        print("\nDomain ID : ", end="")
        print(domain["domain"], end="")
        print("\nWriter topic : ", end="")
        for writer in domain["writer"]:
            print(" ", writer, end="")
        print("\nReader topic : ", end="")
        for reader in domain["reader"]:
            print(" ", reader, end="")
    print("\n")

def find_default_json(root_folder):
    found_files = []
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.endswith("qos.json"):
                file_path = os.path.join(root, file)
                found_files.append(file_path)
                print(f"Found and read {file_path}")
    return found_files

def is_path_file_or_directory(path):
    if os.path.exists(path):
        if os.path.isfile(path):
            print(f"{path} is a file.")
            return 0
        elif os.path.isdir(path):
            print(f"{path} is a directory.")
            return 1
        else:
            print(f"{path} is neither a file nor a directory.")
            return -1
    else :
        return -1

def Start():
    print("Process qos json file Start.")
    PrintTopicInfo()
    ret = is_path_file_or_directory(_json_file)
    if ret == 0:
        print("Update qos file : ", _json_file)
        UpdateJsonQosFile(_json_file)
    elif ret == 1:
        json_files = find_default_json(_json_file)
        for file in json_files:
            UpdateJsonQosFile(file)
    else:
        print("Generate qos file : ", _json_file)
        GenerateJsonQosFile(_json_file)
    print("Process qos json file Sucess.")


def Usage():
    print("Tool Usage : ")
    print("    -d, --domain")
    print("        Domain id")
    print("    -w, --writer")
    print("        writer topic name")
    print("    -r, --reader")
    print("        reader topic name")
    print("    -o, --output")
    print("        output file name")
    print("    -v, --version")
    print("        Show version info")
    print("    -h, --help")
    print("        Show usage")
    print("Example :")
    print("      ")
    print("python3 netaqosgen.py -d 1 -w avm0 -r avm1")
    print("      ")

def DisplayVersion():
    print("Version ID: ", Version_ID)

def ParseOpts():
    count = -1

    if len(sys.argv) < 2 :
        print("Number of input parameters is error.")
        Usage()
        sys.exit(0)

    try:
        opts,args = getopt.getopt(sys.argv[1:],
                "hvd:w:r:o:",["help","version","domain=","writer=","reader=","output=",])
    except getopt.GetoptError:
        print("Error: input parameters is error.")
        Usage()
        sys.exit(2)

    for opts,arg in opts:
        if opts == "-h" or opts == "--help":
            Usage()
        elif opts == "-v" or opts == "--version":
            DisplayVersion()
        elif opts == "-d" or opts == "--domain":
            count = count + 1
            domain_args.append(copy.deepcopy(topic_args))
            domain_args[count]["domain"] = arg
        elif opts == "-w" or opts == "--writer":
            domain_args[count]["writer"].append(arg)
        elif opts == "-r" or opts == "--reader":
            domain_args[count]["reader"].append(arg)
        elif opts == "-o" or opts == "--output":
            global _json_file
            _json_file = arg
    #PrintJsonToConsole(domain_args)

if __name__ == '__main__':
    ParseOpts()
    Start()