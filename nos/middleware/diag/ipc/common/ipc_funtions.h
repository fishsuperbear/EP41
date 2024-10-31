#pragma once

#include <vector>
#include <string>
#include <cstdint>
#include <algorithm>
#include <cstring>
#include <sys/stat.h>
#include <dirent.h>
#include <iostream>

std::vector<uint8_t> VoidPointerToVector(void* data, int size);

void* VectorToVoidPointer(const std::vector<uint8_t>& vecData);

std::vector<uint8_t> CharPointerToVector(const char* data, size_t dataSize);

bool IPCPathRemove(const std::string &pathName);

bool doesFileExistWithPrefix(const std::string& directoryPath, const std::string& prefix);