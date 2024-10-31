/**
 * * --------------------------------------------------------------------
 * * |                                                                  |
 * * |     _         _    _ _______ ____         _____ ____  __  __     |
 * * |    (_)   /\  | |  | |__   __/ __ \       / ____/ __ \|  \/  |    |
 * * |     _   /  \ | |  | |  | | | |  | |     | |   | |  | | \  / |    |
 * * |    | | / /\ \| |  | |  | | | |  | |     | |   | |  | | |\/| |    |
 * * |    | |/ ____ \ |__| |  | | | |__| |  _  | |___| |__| | |  | |    |
 * * |    |_/_/    \_\____/   |_|  \____/  (_)  \_____\____/|_|  |_|    |
 * * |                                                                  |
 * * --------------------------------------------------------------------
 *
 *  * Copyright @ 2020 iAuto (Shanghai) Co., Ltd.
 *  * All Rights Reserved.
 *  *
 *  * Redistribution and use in source and binary forms, with or without
 *  * modification, are NOT permitted except as agreed by
 *  * iAuto (Shanghai) Co., Ltd.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS,
 *  * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *
 *
 * @file NCJson.h
 * @brief
 * @date 2020-05-08
 *
 */

#ifndef MANIFEST_JSON_PARSE
#define MANIFEST_JSON_PARSE

#include <osal/ncore/NCString.h>

#include <memory>
#include <string>
#include <vector>

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif

OSAL_BEGIN_NAMESPACE

/*
 * ReadJsonType
 * READ_JSON_TYPE_RAW:Read data directly from json file
 * READ_JSON_TYPE_KEY_VALUE:Read all data to memory.Deleted keywords:key, value
 * in the memory data.
 */
enum ReadJsonType : UINT32 { READ_JSON_TYPE_RAW = 0, READ_JSON_TYPE_KEY_VALUE };

/*
 * JsonFileType:the type of the json file
 */
enum JsonFileType : UINT32 {
    JSON_FILE_TYPE_PROCESS = 0,  // for process_manifest.json
    JSON_FILE_TYPE_PERSISTENCY,  // for persistency_manifest.json
    JSON_FILE_TYPE_E2ESM,        // for e2e_statemachines.json
    JSON_FILE_TYPE_E2EDATAID,    // for e2e_dataid_mapping.json
    JSON_FILE_TYPE_MAX
};

/*
 * JsonNodeType:type of data
 */
enum JsonNodeType : UINT32 {
    JSON_NODE_TYPE_NULL = 0U,
    JSON_NODE_TYPE_BOOL,
    JSON_NODE_TYPE_INT,
    JSON_NODE_TYPE_UINT,
    JSON_NODE_TYPE_DOUBLE,
    JSON_NODE_TYPE_FLOAT,
    JSON_NODE_TYPE_STRING,
    JSON_NODE_TYPE_OBJECT,
    JSON_NODE_TYPE_ARRAY,
    JSON_NODE_TYPE_NONE,
};

class JsonObject;

class NCJsonObject {
   public:
    ~NCJsonObject();

    /*
     * Will Delete
     *
     * [IN]ReadJsonType type:the method of read json file
     * [IN]JsonType jsontype:the type of jsonfilepath
     * [IN]const std::string &filepath:path of json file.if JsonType is
     * JSON_TYPE_PROCESS, filepath can be ignored.
     * [OUT]NCJsonObject:ues to get json data
     */
    static NCJsonObject LoadJsonObject( ReadJsonType type, const std::string& filepath );

    /**
     * @brief load machine manifest
     *
     * @param filename file name
     * @return NCJsonObject
     */
    static NCJsonObject LoadMachineJsonObject( const NCString& filename );

    /**
     * @brief load self applicantion manifest
     *
     * @param filename file name
     * @return NCJsonObject
     */
    static NCJsonObject LoadSelfJsonObject( const NCString& filename );

    /**
     * @brief load other applicantion manifest
     *
     * @param cluster   cluster name
     * @param module    process name
     * @param filename  file name
     * @return NCJsonObject
     */
    static NCJsonObject LoadProcessJsonObject( const NCString& cluster, const NCString& module, const NCString& filename );

    static NCJsonObject LoadJsonFromData( const CHAR* data, const UINT32 size);

    NCJsonObject( std::shared_ptr<JsonObject> other );
    NCJsonObject( JsonObject* other );
    NCJsonObject( const NCJsonObject& other );
    NCJsonObject();
    NCJsonObject( DOUBLE value );
    NCJsonObject( FLOAT value );
    NCJsonObject( NC_BOOL value );
    NCJsonObject( INT64 value );
    NCJsonObject( UINT64 value );
    NCJsonObject( const char* value );

    NCJsonObject operator[]( const std::string& key );
    NCJsonObject operator[]( const UINT32 uindex );
    NCJsonObject& operator=( const NCJsonObject& other );

    VOID append( INT64 value );
    VOID append( UINT64 value );
    VOID append( DOUBLE value );
    VOID append( FLOAT value );
    VOID append( std::string value );
    VOID append( NCJsonObject& other );
    VOID        dump();
    std::string toString();

    UINT32               getListSize();
    JsonNodeType getNodeType();
    const CHAR*  getKey();
    NC_BOOL isKeyExist( const std::string& key );
    NC_BOOL isKeyExist();
    NC_BOOL isValueExist();
    NC_BOOL isValid();

    NC_BOOL getInt8( INT8& num );
    NC_BOOL getInt16( INT16& num );
    NC_BOOL getInt32( INT32& num );
    NC_BOOL getInt64( INT64& num );
    NC_BOOL getUInt8( UINT8& num );
    NC_BOOL getUInt16( UINT16& num );
    NC_BOOL getUInt32( UINT32& num );
    NC_BOOL getUInt64( UINT64& num );
    NC_BOOL getFloat( FLOAT& num );
    NC_BOOL getDouble( DOUBLE& num );
    NC_BOOL getBool( NC_BOOL& bvalue );
    NC_BOOL getString( CHAR** str );

    INT8        asInt8();
    INT16       asInt16();
    INT32       asInt32();
    INT64       asInt64();
    UINT8       asUInt8();
    UINT16      asUInt16();
    UINT32      asUInt32();
    UINT64      asUInt64();
    FLOAT       asFloat();
    DOUBLE      asDouble();
    NC_BOOL     asBool();
    const CHAR* asString();

   private:
    NCJsonObject load( const ReadJsonType type, const std::string& filepath );
    std::shared_ptr<JsonObject> m_obj;
};
OSAL_END_NAMESPACE
#endif  // MANIFEST_JSON_PARSE

/* EOF */