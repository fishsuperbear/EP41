

/*
 * Copyright (c) Hozon Auto Co., Ltd. 2023-2025. All rights reserved.
 * Module: per
 * Description: 配置通用函数
 * Created on: Feb 7, 2023
 *
 */
#ifndef SERVICE_CONFIG_SERVER_INCLUDE_CFG_UTILS_H_
#define SERVICE_CONFIG_SERVER_INCLUDE_CFG_UTILS_H_
#include <dirent.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <ctime>
#include <iostream>
#include <iterator>
#include <map>
#include <regex>
#include <shared_mutex>
#include <sstream>
#include <string>
#include <typeinfo>
#include <vector>

#include "include/cfg_data_def.h"
#include "include/cfg_logger.h"
#include "phm/include/phm_client.h"
namespace hozon {
namespace netaos {
namespace cfg {
struct Callback {
    std::function<void(const std::string&, const std::string&, const bool&)> MonBoolcb;
    std::function<void(const std::string&, const std::string&, const uint8_t&)> MonUint8cb;
    std::function<void(const std::string&, const std::string&, const int32_t&)> MonInt32cb;
    std::function<void(const std::string&, const std::string&, const float&)> MonFloatcb;
    std::function<void(const std::string&, const std::string&, const double&)> MonDoublecb;
    std::function<void(const std::string&, const std::string&, const int64_t&)> MonLongcb;
    std::function<void(const std::string&, const std::string&, const std::string&)> MonStringcb;
    std::function<void(const std::string&, const std::string&, const std::vector<bool>&)> MonVecBoolcb;
    std::function<void(const std::string&, const std::string&, const std::vector<uint8_t>&)> MonVecUint8cb;
    std::function<void(const std::string&, const std::string&, const std::vector<int32_t>&)> MonVecInt32cb;
    std::function<void(const std::string&, const std::string&, const std::vector<float>&)> MonVecFloatcb;
    std::function<void(const std::string&, const std::string&, const std::vector<double>&)> MonVecDoublecb;
    std::function<void(const std::string&, const std::string&, const std::vector<int64_t>&)> MonVecLongcb;
    std::function<void(const std::string&, const std::string&, const std::vector<std::string>&)> MonVecStringcb;
    std::function<CfgResultCode(const bool&)> ResBoolcb;
    std::function<CfgResultCode(const uint8_t&)> ResUint8cb;
    std::function<CfgResultCode(const int32_t&)> ResInt32cb;
    std::function<CfgResultCode(const float&)> ResFloatcb;
    std::function<CfgResultCode(const double&)> ResDoublecb;
    std::function<CfgResultCode(const int64_t&)> ResLongcb;
    std::function<CfgResultCode(const std::string&)> ResStringcb;
    std::function<CfgResultCode(const std::vector<bool>&)> ResVecBoolcb;
    std::function<CfgResultCode(const std::vector<uint8_t>&)> ResVecUint8cb;
    std::function<CfgResultCode(const std::vector<int32_t>&)> ResVecInt32cb;
    std::function<CfgResultCode(const std::vector<float>&)> ResVecFloatcb;
    std::function<CfgResultCode(const std::vector<double>&)> ResVecDoublecb;
    std::function<CfgResultCode(const std::vector<int64_t>&)> ResVecLongcb;
    std::function<CfgResultCode(const std::vector<std::string>&)> ResVecStringcb;
};
class CfgUtils {
 public:
    template <typename T>
    static uint8_t GetDataType(const T& datavalue) {
        if (typeid(T) == typeid(bool)) {
            return CFG_DATA_BOOL;
        } else if (typeid(T) == typeid(double)) {
            return CFG_DATA_DOUBLE;
        } else if (typeid(T) == typeid(float)) {
            return CFG_DATA_FLOAT;
        } else if (typeid(T) == typeid(int32_t)) {
            return CFG_DATA_INT32;
        } else if (typeid(T) == typeid(uint8_t)) {
            return CFG_DATA_UINT8;
        } else if (typeid(T) == typeid(int64_t)) {
            return CFG_DATA_LONG;
        } else if (typeid(T) == typeid(std::string)) {
            return CFG_DATA_STRING;
        } else if (typeid(T) == typeid(std::vector<bool>)) {
            return CFG_DATA_VECTOR_BOOL;
        } else if (typeid(T) == typeid(std::vector<uint8_t>)) {
            return CFG_DATA_VECTOR_UINT8;
        } else if (typeid(T) == typeid(std::vector<int32_t>)) {
            return CFG_DATA_VECTOR_INT32;
        } else if (typeid(T) == typeid(std::vector<float>)) {
            return CFG_DATA_VECTOR_FLOAT;
        } else if (typeid(T) == typeid(std::vector<double>)) {
            return CFG_DATA_VECTOR_DOUBLE;
        } else if (typeid(T) == typeid(std::vector<int64_t>)) {
            return CFG_DATA_VECTOR_LONG;
        } else if (typeid(T) == typeid(std::vector<std::string>)) {
            return CFG_DATA_VECTOR_STRING;
        } else {
            return CFG_DATA_OTHER;
        }
    }
    static std::string GetDataTypeStr(const uint8_t& datatype) {
        std::string res = "unknown";
        switch (datatype) {
            case CFG_DATA_BOOL:
                res = "bool";
                break;
            case CFG_DATA_DOUBLE:
                res = "double";
                break;
            case CFG_DATA_FLOAT:
                res = "float";
                break;
            case CFG_DATA_INT32:
                res = "int32_t";
                break;
            case CFG_DATA_UINT8:
                res = "uint8_t";
                break;
            case CFG_DATA_LONG:
                res = "long";
                break;
            case CFG_DATA_STRING:
                res = "string";
                break;
            case CFG_DATA_VECTOR_BOOL:
                res = "vector<bool>";
                break;
            case CFG_DATA_VECTOR_UINT8:
                res = "vector<uint8_t>";
                break;
            case CFG_DATA_VECTOR_INT32:
                res = "vector<int32_t>";
                break;
            case CFG_DATA_VECTOR_FLOAT:
                res = "vector<float>";
                break;
            case CFG_DATA_VECTOR_DOUBLE:
                res = "vector<double>";
                break;
            case CFG_DATA_VECTOR_LONG:
                res = "vector<long>";
                break;
            case CFG_DATA_VECTOR_STRING:
                res = "vector<string>";
                break;
            default:
                break;
        }
        return res;
    }
    template <typename T = bool>
    static void SetCallback(const std::function<void(const std::string&, const std::string&, const bool&)> func, Callback& callback) {
        callback.MonBoolcb = func;
    }
    template <typename T = double>
    static void SetCallback(const std::function<void(const std::string&, const std::string&, const double&)> func, Callback& callback) {
        callback.MonDoublecb = func;
    }
    template <typename T = float>
    static void SetCallback(const std::function<void(const std::string&, const std::string&, const float&)> func, Callback& callback) {
        callback.MonFloatcb = func;
    }
    template <typename T = int32_t>
    static void SetCallback(const std::function<void(const std::string&, const std::string&, const int32_t&)> func, Callback& callback) {
        callback.MonInt32cb = func;
    }
    template <typename T = uint8_t>
    static void SetCallback(const std::function<void(const std::string&, const std::string&, const uint8_t&)> func, Callback& callback) {
        callback.MonUint8cb = func;
    }
    template <typename T = int64_t>
    static void SetCallback(const std::function<void(const std::string&, const std::string&, const int64_t&)> func, Callback& callback) {
        callback.MonLongcb = func;
    }
    template <typename T = std::string>
    static void SetCallback(const std::function<void(const std::string&, const std::string&, const std::string&)> func, Callback& callback) {
        callback.MonStringcb = func;
    }
    template <typename T = std::vector<bool>>
    static void SetCallback(const std::function<void(const std::string&, const std::string&, const std::vector<bool>&)> func, Callback& callback) {
        callback.MonVecBoolcb = func;
    }
    template <typename T = std::vector<double>>
    static void SetCallback(const std::function<void(const std::string&, const std::string&, const std::vector<double>&)> func, Callback& callback) {
        callback.MonVecDoublecb = func;
    }
    template <typename T = std::vector<float>>
    static void SetCallback(const std::function<void(const std::string&, const std::string&, const std::vector<float>&)> func, Callback& callback) {
        callback.MonVecFloatcb = func;
    }
    template <typename T = std::vector<int32_t>>
    static void SetCallback(const std::function<void(const std::string&, const std::string&, const std::vector<int32_t>&)> func, Callback& callback) {
        callback.MonVecInt32cb = func;
    }
    template <typename T = std::vector<uint8_t>>
    static void SetCallback(const std::function<void(const std::string&, const std::string&, const std::vector<uint8_t>&)> func, Callback& callback) {
        callback.MonVecUint8cb = func;
    }
    template <typename T = std::vector<int64_t>>
    static void SetCallback(const std::function<void(const std::string&, const std::string&, const std::vector<int64_t>&)> func, Callback& callback) {
        callback.MonVecLongcb = func;
    }
    template <typename T = std::vector<std::string>>
    static void SetCallback(const std::function<void(const std::string&, const std::string&, const std::vector<std::string>&)> func, Callback& callback) {
        callback.MonVecStringcb = func;
    }

    template <typename T = bool>
    static void SetCallback(const std::function<CfgResultCode(const bool&)> func, Callback& callback) {
        callback.ResBoolcb = func;
    }
    template <typename T = double>
    static void SetCallback(const std::function<CfgResultCode(const double&)> func, Callback& callback) {
        callback.ResDoublecb = func;
    }
    template <typename T = float>
    static void SetCallback(const std::function<CfgResultCode(const float&)> func, Callback& callback) {
        callback.ResFloatcb = func;
    }
    template <typename T = int32_t>
    static void SetCallback(const std::function<CfgResultCode(const int32_t&)> func, Callback& callback) {
        callback.ResInt32cb = func;
    }
    template <typename T = uint8_t>
    static void SetCallback(const std::function<CfgResultCode(const uint8_t&)> func, Callback& callback) {
        callback.ResUint8cb = func;
    }
    template <typename T = int64_t>
    static void SetCallback(const std::function<CfgResultCode(const int64_t&)> func, Callback& callback) {
        callback.ResLongcb = func;
    }
    template <typename T = std::string>
    static void SetCallback(const std::function<CfgResultCode(const std::string&)> func, Callback& callback) {
        callback.ResStringcb = func;
    }
    template <typename T = std::vector<bool>>
    static void SetCallback(const std::function<CfgResultCode(const std::vector<bool>&)> func, Callback& callback) {
        callback.ResVecBoolcb = func;
    }
    template <typename T = std::vector<double>>
    static void SetCallback(const std::function<CfgResultCode(const std::vector<double>&)> func, Callback& callback) {
        callback.ResVecDoublecb = func;
    }
    template <typename T = std::vector<float>>
    static void SetCallback(const std::function<CfgResultCode(const std::vector<float>&)> func, Callback& callback) {
        callback.ResVecFloatcb = func;
    }
    template <typename T = std::vector<int32_t>>
    static void SetCallback(const std::function<CfgResultCode(const std::vector<int32_t>&)> func, Callback& callback) {
        callback.ResVecInt32cb = func;
    }
    template <typename T = std::vector<uint8_t>>
    static void SetCallback(const std::function<CfgResultCode(const std::vector<uint8_t>&)> func, Callback& callback) {
        callback.ResVecUint8cb = func;
    }
    template <typename T = std::vector<int64_t>>
    static void SetCallback(const std::function<CfgResultCode(const std::vector<int64_t>&)> func, Callback& callback) {
        callback.ResVecLongcb = func;
    }
    template <typename T = std::vector<std::string>>
    static void SetCallback(const std::function<CfgResultCode(const std::vector<std::string>&)> func, Callback& callback) {
        callback.ResVecStringcb = func;
    }
    static uint8_t GetCallback(const uint8_t& datatype, const std::string& clientname, const std::string& key, const std::vector<uint8_t>& value, const Callback callback) {
        uint8_t res = 255;
        switch (datatype) {
            case CFG_DATA_BOOL: {
                bool val;
                BytesToNum(value, val);
                if (callback.MonBoolcb != nullptr) {
                    callback.MonBoolcb(clientname, key, val);
                } else if (callback.ResBoolcb != nullptr) {
                    res = callback.ResBoolcb(val);
                }
                break;
            }
            case CFG_DATA_DOUBLE: {
                double val;
                BytesToNum(value, val);
                if (callback.MonDoublecb != nullptr) {
                    callback.MonDoublecb(clientname, key, val);
                } else if (callback.ResDoublecb != nullptr) {
                    res = callback.ResDoublecb(val);
                }
                break;
            }
            case CFG_DATA_FLOAT: {
                float val;
                BytesToNum(value, val);
                if (callback.MonFloatcb != nullptr) {
                    callback.MonFloatcb(clientname, key, val);
                } else if (callback.ResFloatcb != nullptr) {
                    res = callback.ResFloatcb(val);
                }
                break;
            }
            case CFG_DATA_INT32: {
                int32_t val;
                BytesToNum(value, val);
                if (callback.MonInt32cb != nullptr) {
                    callback.MonInt32cb(clientname, key, val);
                } else if (callback.ResInt32cb != nullptr) {
                    res = callback.ResInt32cb(val);
                }
                break;
            }
            case CFG_DATA_UINT8: {
                uint8_t val;
                BytesToNum(value, val);
                if (callback.MonUint8cb != nullptr) {
                    callback.MonUint8cb(clientname, key, val);
                } else if (callback.ResUint8cb != nullptr) {
                    res = callback.ResUint8cb(val);
                }
                break;
            }
            case CFG_DATA_LONG: {
                int64_t val;
                BytesToNum(value, val);
                if (callback.MonLongcb != nullptr) {
                    callback.MonLongcb(clientname, key, val);
                } else if (callback.ResLongcb != nullptr) {
                    res = callback.ResLongcb(val);
                }
                break;
            }
            case CFG_DATA_STRING: {
                std::string val;
                BytesToNum(value, val);
                if (callback.MonStringcb != nullptr) {
                    callback.MonStringcb(clientname, key, val);
                } else if (callback.ResStringcb != nullptr) {
                    res = callback.ResStringcb(val);
                }
                break;
            }
            case CFG_DATA_VECTOR_BOOL: {
                std::vector<bool> vec;
                BytesToVec<bool>(value, vec);
                if (callback.MonVecBoolcb != nullptr) {
                    callback.MonVecBoolcb(clientname, key, vec);
                } else if (callback.ResVecBoolcb != nullptr) {
                    res = callback.ResVecBoolcb(vec);
                }
                break;
            }
            case CFG_DATA_VECTOR_UINT8: {
                std::vector<uint8_t> vec;
                BytesToVec<uint8_t>(value, vec);
                if (callback.MonVecUint8cb != nullptr) {
                    callback.MonVecUint8cb(clientname, key, vec);
                } else if (callback.ResVecUint8cb != nullptr) {
                    res = callback.ResVecUint8cb(vec);
                }
                break;
            }
            case CFG_DATA_VECTOR_INT32: {
                std::vector<int32_t> vec;
                BytesToVec<int32_t>(value, vec);
                if (callback.MonVecInt32cb != nullptr) {
                    callback.MonVecInt32cb(clientname, key, vec);
                } else if (callback.ResVecInt32cb != nullptr) {
                    res = callback.ResVecInt32cb(vec);
                }
                break;
            }
            case CFG_DATA_VECTOR_FLOAT: {
                std::vector<float> vec;
                BytesToVec<float>(value, vec);
                if (callback.MonVecFloatcb != nullptr) {
                    callback.MonVecFloatcb(clientname, key, vec);
                } else if (callback.ResVecFloatcb != nullptr) {
                    res = callback.ResVecFloatcb(vec);
                }
                break;
            }
            case CFG_DATA_VECTOR_DOUBLE: {
                std::vector<double> vec;
                BytesToVec<double>(value, vec);
                if (callback.MonVecDoublecb != nullptr) {
                    callback.MonVecDoublecb(clientname, key, vec);
                } else if (callback.ResVecDoublecb != nullptr) {
                    res = callback.ResVecDoublecb(vec);
                }
                break;
            }
            case CFG_DATA_VECTOR_LONG: {
                std::vector<int64_t> vec;
                BytesToVec<int64_t>(value, vec);
                if (callback.MonVecLongcb != nullptr) {
                    callback.MonVecLongcb(clientname, key, vec);
                } else if (callback.ResVecLongcb != nullptr) {
                    res = callback.ResVecLongcb(vec);
                }
                break;
            }
            case CFG_DATA_VECTOR_STRING: {
                std::vector<std::string> vec;
                BytesToVec<std::string>(value, vec);
                if (callback.MonVecStringcb != nullptr) {
                    callback.MonVecStringcb(clientname, key, vec);
                } else if (callback.ResVecStringcb != nullptr) {
                    res = callback.ResVecStringcb(vec);
                }
                break;
            }
            default:
                break;
        }
        return res;
    }
    static std::string VecToString(const std::vector<uint8_t>& value, const uint8_t& datatype) {
        std::string str;
        switch (datatype) {
            case CFG_DATA_BOOL: {
                bool val;
                BytesToNum(value, val);
                str = NumToString(val);
                break;
            }
            case CFG_DATA_DOUBLE: {
                double val;
                BytesToNum(value, val);
                str = NumToString(val);
                break;
            }
            case CFG_DATA_FLOAT: {
                float val;
                BytesToNum(value, val);
                str = NumToString(val);
                break;
            }
            case CFG_DATA_INT32: {
                int32_t val;
                BytesToNum(value, val);
                str = NumToString(val);
                break;
            }
            case CFG_DATA_UINT8: {
                uint8_t val;
                BytesToNum(value, val);
                str = NumToString(val);
                break;
            }
            case CFG_DATA_LONG: {
                int64_t val;
                BytesToNum(value, val);
                str = NumToString(val);
                break;
            }
            case CFG_DATA_STRING: {
                BytesToNum(value, str);
                break;
            }
            case CFG_DATA_VECTOR_BOOL: {
                std::vector<bool> vec;
                BytesToVec<bool>(value, vec);
                str = ChopLineStringEx<bool>(vec);
                break;
            }
            case CFG_DATA_VECTOR_UINT8: {
                std::vector<uint16_t> vec;
                for (size_t size = 0; size < value.size(); size++) {
                    vec.push_back(value[size]);
                }
                str = ChopLineStringEx<uint16_t>(vec);
                break;
            }
            case CFG_DATA_VECTOR_INT32: {
                std::vector<int32_t> vec;
                BytesToVec<int32_t>(value, vec);
                str = ChopLineStringEx<int32_t>(vec);
                break;
            }
            case CFG_DATA_VECTOR_FLOAT: {
                std::vector<float> vec;
                BytesToVec<float>(value, vec);
                str = ChopLineStringEx<float>(vec);
                break;
            }
            case CFG_DATA_VECTOR_DOUBLE: {
                std::vector<double> vec;
                BytesToVec<double>(value, vec);
                str = ChopLineStringEx<double>(vec);
                break;
            }
            case CFG_DATA_VECTOR_LONG: {
                std::vector<int64_t> vec;
                BytesToVec<int64_t>(value, vec);
                str = ChopLineStringEx<int64_t>(vec);
                break;
            }
            case CFG_DATA_VECTOR_STRING: {
                std::vector<uint16_t> vec;
                for (size_t size = 0; size < value.size(); size++) {
                    vec.push_back(value[size]);
                }
                str = ChopLineStringEx<uint16_t>(vec);
                break;
            }
            default:
                break;
        }
        return str;
    }

    static std::vector<uint8_t> stringToVec(const std::string value, const uint8_t& datatype) {
        std::vector<uint8_t> vec;
        switch (datatype) {
            case CFG_DATA_BOOL: {
                bool val;
                stringToNum<bool>(value, val);
                vec = NumToBytes(val);
                break;
            }
            case CFG_DATA_DOUBLE: {
                double val;
                stringToNum<double>(value, val);
                vec = NumToBytes(val);
                break;
            }
            case CFG_DATA_FLOAT: {
                float val;
                stringToNum<float>(value, val);
                vec = NumToBytes(val);
                break;
            }
            case CFG_DATA_INT32: {
                int32_t val;
                stringToNum<int32_t>(value, val);
                vec = NumToBytes(val);
                break;
            }
            case CFG_DATA_UINT8: {
                uint8_t val;
                stringToNum<uint8_t>(value, val);
                vec = NumToBytes(val);
                break;
            }
            case CFG_DATA_LONG: {
                int64_t val;
                stringToNum<int64_t>(value, val);
                vec = NumToBytes(val);
                break;
            }
            case CFG_DATA_STRING: {
                vec = NumToBytes(value);
                break;
            }
            case CFG_DATA_VECTOR_BOOL: {
                std::vector<bool> val;
                ChopStringLineEx<bool>(value, val);
                VecToBytes<bool>(val, vec);
                break;
            }
            case CFG_DATA_VECTOR_UINT8: {
                std::vector<uint16_t> val;
                ChopStringLineEx<uint16_t>(value, val);
                for (size_t size = 0; size < val.size(); size++) {
                    vec.push_back(val[size]);
                }
                break;
            }
            case CFG_DATA_VECTOR_INT32: {
                std::vector<int32_t> val;
                ChopStringLineEx<int32_t>(value, val);
                VecToBytes<int32_t>(val, vec);
                break;
            }
            case CFG_DATA_VECTOR_FLOAT: {
                std::vector<float> val;
                ChopStringLineEx<float>(value, val);
                VecToBytes<float>(val, vec);
                break;
            }
            case CFG_DATA_VECTOR_DOUBLE: {
                std::vector<double> val;
                ChopStringLineEx<double>(value, val);
                VecToBytes<double>(val, vec);
                break;
            }
            case CFG_DATA_VECTOR_LONG: {
                std::vector<int64_t> val;
                ChopStringLineEx<int64_t>(value, val);
                VecToBytes<int64_t>(val, vec);
                break;
            }
            case CFG_DATA_VECTOR_STRING: {
                std::vector<uint16_t> val;
                ChopStringLineEx<uint16_t>(value, val);
                for (size_t size = 0; size < val.size(); size++) {
                    vec.push_back(val[size]);
                }
                break;
            }
            default:
                break;
        }
        return vec;
    }

    template <typename T>
    static void ChopStringLineEx(std::string line, std::vector<T>& subvec) {
        std::stringstream linestream(line);
        std::string sub;
        while (linestream >> sub) {
            T val;
            stringToNum<T>(sub, val);
            subvec.push_back(val);
        }
    }
    template <typename T>
    static std::string ChopLineStringEx(const std::vector<T>& t) {
        std::stringstream ss;
        copy(t.begin(), t.end(), std::ostream_iterator<T>(ss, " "));
        return ss.str();
    }
    template <typename T>
    static std::string NumToString(const T t) {
        std::ostringstream os;
        os << t;
        return os.str();
    }
    template <typename T = uint8_t>
    static std::string NumToString(const uint8_t t) {
        return std::to_string(t);
    }
    template <typename T = uint8_t>
    static void stringToNum(const std::string str, uint8_t& num) {
        num = std::stoi(str);
    }
    template <typename T>
    static void stringToNum(const std::string str, T& num) {
        std::istringstream iss(str);
        // num = iss.get();
        iss >> std::noskipws >> num;
    }
    template <typename T>
    static void BytesToVec(const std::vector<uint8_t>& invalue, std::vector<T>& outvalue) {
        for (size_t i = 0; i < invalue.size(); i += sizeof(T)) {
            T val;
            // memcpy(&val, &invalue[i], sizeof(T));
            val = *reinterpret_cast<const T*>(&invalue[i]);
            outvalue.push_back(val);
        }
    }
    template <typename T = std::string>
    static void BytesToVec(const std::vector<uint8_t>& invalue, std::vector<std::string>& outvalue) {
        size_t size = 0;
        while (size < invalue.size()) {
            size_t nsize = *reinterpret_cast<const uint16_t*>(&invalue[size]);
            size += sizeof(uint16_t);
            std::string itemvalue(invalue.begin() + size, invalue.begin() + size + nsize);
            outvalue.push_back(itemvalue);
            size += nsize;
        }
    }

    template <typename T>
    static void VecToBytes(const std::vector<T>& invalue, std::vector<uint8_t>& outvalue) {
        for (size_t i = 0; i < invalue.size(); i++) {
            std::vector<uint8_t> vecvalue = NumToBytes<T>(invalue[i]);
            outvalue.insert(outvalue.end(), vecvalue.begin(), vecvalue.end());
        }
        CONFIG_LOG_INFO << "invaluesize  " << invalue.size() << " outvaluesize  " << outvalue.size();
    }
    template <typename T = std::string>
    static void VecToBytes(const std::vector<std::string>& invalue, std::vector<uint8_t>& outvalue) {
        for (size_t i = 0; i < invalue.size(); i++) {
            std::vector<uint8_t> vecvalue = NumToBytes<std::string>(invalue[i]);
            std::vector<uint8_t> lenvalue(sizeof(uint16_t));
            *reinterpret_cast<uint16_t*>(lenvalue.data()) = vecvalue.size();
            outvalue.insert(outvalue.end(), lenvalue.begin(), lenvalue.end());
            outvalue.insert(outvalue.end(), vecvalue.begin(), vecvalue.end());
        }
        CONFIG_LOG_INFO << "invaluesize  " << invalue.size() << " outvaluesize  " << outvalue.size();
    }
    template <typename T>
    static std::vector<uint8_t> NumToBytes(const T t) {
        std::vector<uint8_t> bytes(sizeof(T));
        *reinterpret_cast<T*>(bytes.data()) = t;
        return bytes;
    }

    template <typename T = std::string>
    static std::vector<uint8_t> NumToBytes(const std::string t) {
        std::vector<uint8_t> bytes(t.size());
        ::memcpy(bytes.data(), t.data(), t.size());
        return bytes;
    }
    template <typename T>
    static void BytesToNum(std::vector<uint8_t> t, T& value) {
        value = *reinterpret_cast<const T*>(t.data());
    }
    template <typename T = std::string>
    static void BytesToNum(std::vector<uint8_t> t, std::string& value) {
        value.assign(t.begin(), t.end());
    }
    static std::string GetUTCTime() {
        struct timeval nowTus;
        gettimeofday(&nowTus, NULL);
        struct tm nowTm2;
        localtime_r(&nowTus.tv_sec, &nowTm2);
        char strT[64];
        snprintf(strT, sizeof(strT), "%04d/%02d/%02d %02d:%02d:%02d:%03ld", nowTm2.tm_year + 1900, nowTm2.tm_mon + 1, nowTm2.tm_mday, nowTm2.tm_hour, nowTm2.tm_min, nowTm2.tm_sec,
                 nowTus.tv_usec / 1000);
        return strT;
    }

    static void StringToVec(std::string strstr, std::vector<uint8_t>& dstvec) {
        std::regex re(" ");
        std::sregex_token_iterator first{strstr.begin(), strstr.end(), re, -1}, last;
        std::vector<std::string> vec = {first, last};
        for (std::string& item : vec) {
            dstvec.push_back(static_cast<uint8_t>(std::strtoul(item.c_str(), 0, 16)));
        }
    }

    static std::string GetProcName() {
        char szProcessName[1024];
        std::string procname;
        ssize_t count = readlink("/proc/self/exe", szProcessName, sizeof(szProcessName));
        if (count != -1) {
            szProcessName[count] = '\0';
            std::string domain(szProcessName);
            string::size_type pos = domain.find_last_of("/");
            if (pos == domain.npos) {
                procname = domain;
            } else {
                procname = domain.substr(pos + 1, domain.size());
            }
            CONFIG_LOG_INFO << "Process name: " << procname.c_str();
        } else {
            CONFIG_LOG_INFO << "Failed to get process name.";
        }
        pid_t p1 = getpid();
        return procname + "/" + std::to_string(p1);
    }

    static std::string GetProcid(const std::string clientname) {
        std::string procid;
        // string::size_type pos = clientname.find_last_of("/");
        // if (pos == clientname.npos) {
        //     procid = clientname;
        // } else {
        //     procid = clientname.substr(0, pos);
        // }
        std::size_t firstPos = clientname.find("/");
        if (firstPos != std::string::npos) {
            std::size_t secondPos = clientname.find("/", firstPos + 1);
            if (secondPos != std::string::npos) {
                procid = clientname.substr(0, secondPos);
            } else {
                procid = clientname;
            }
        } else {
            procid = clientname;
        }
        return procid;
    }
};

}  // namespace cfg
}  // namespace netaos
}  // namespace hozon
#endif  // SERVICE_CONFIG_SERVER_INCLUDE_CFG_UTILS_H_
