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
 * @file NCIniOperator.h
 * @brief
 * @date 2020-05-08
 *
 */

#ifndef NCINIOPERATOR_H
#define NCINIOPERATOR_H

#include "osal/ncore/NCRefBase.h"
#include "osal/ncore/NCString.h"

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif

OSAL_BEGIN_NAMESPACE
// external class declaration
class NCIniLineList;

// class declaration
class NCIniOperator;
class NCIniKey;

/**
 * @brief The class of key in .ini file
 *
 * @class NCIniKey
 *
 */
class __attribute__( ( visibility( "default" ) ) ) NCIniKey {
   public:
    /**
     * @brief Default constructor
     */
    NCIniKey();

    /**
     * @brief Constructor
     *
     * @param[in] keyname     The name of the key
     * @param[in] keyvalue    The value of the key
     */
    NCIniKey( const NCString &keyname, const NCString &keyvalue );

    /**
     * @brief \overload
     *
     * @param[in] keyname     The name of the key
     * @param[in] keyvalue    The value of the key
     */
    NCIniKey( const NCString &keyname, const INT32 &keyvalue );

    /**
     * @brief \overload
     *
     * @param[in] keyname      The name of the key
     * @param[in] keyvalue    The value of the key
     */
    NCIniKey( const NCString &keyname, const NC_BOOL &keyvalue );

    /**
     * @brief \overload
     *
     * @param[in] keyname      The name of the key
     * @param[in] keyvalue    The value of the key
     */
    NCIniKey( const NCString &keyname, const FLOAT &keyvalue );

    /**
     * @brief \overload
     *
     * @param[in] keyname      The name of the key
     * @param[in] keyvalue    The value of the key
     */
    NCIniKey( const NCString &keyname, const DOUBLE &keyvalue );

    /**
     * Destructor
     */
    ~NCIniKey();

    /**
     * @brief Return the name of the key
     *
     * @return the name of the key
     */
    NCString name() const;

    /**
     * @brief Return the value of the key
     *
     * @return the name of the key
     */
    NCString value() const;

    /**
     * @brief Set the keyname of the key
     *
     * @param[in] keyname    The name of the key
     * @return If the function call succeeds, it returns NC_TRUE.
               Otherwise, it returns NC_FALSE.
     */
    NC_BOOL setName( const NCString &keyname );

    /**
     * @brief Set the value of the key
     *
     * @param[in] keyvalue    The value of the key
     * @return If the function call succeeds, it returns NC_TRUE.
               Otherwise, it returns NC_FALSE.
     */
    NC_BOOL setValue( const NCString &keyvalue );

    /**
     * @brief \overload
     *
     * @param[in] keyvalue    The value of the key
     * @return If the function call succeeds, it returns NC_TRUE.
               Otherwise, it returns NC_FALSE.
     */
    NC_BOOL setValue( const INT32 &keyvalue );

    /**
     * @brief \overload
     *
     * @param[in] keyvalue    The value of the key
     * @return If the function call succeeds, it returns NC_TRUE.
               Otherwise, it returns NC_FALSE.
     */
    NC_BOOL setValue( const NC_BOOL &keyvalue );

    /**
     * @brief \overload
     *
     * @param[in] keyvalue    The value of the key
     * @return If the function call succeeds, it returns NC_TRUE.
               Otherwise, it returns NC_FALSE.
     */
    NC_BOOL setValue( const FLOAT &keyvalue );

    /**
     * @brief \overload
     *
     * @param[in] keyvalue    The value of the key
     * @return If the function call succeeds, it returns NC_TRUE.
               Otherwise, it returns NC_FALSE.
     */
    NC_BOOL setValue( const DOUBLE &keyvalue );

   private:
    NCString m_name;   // The name of the key
    NCString m_value;  // The value of the key
};

typedef VOID ( *NC_IniArrayWriter )( const NCString &var, VOID *container, UINT32 len );

/**
 * @brief
 *
 * @class NCIniArrayValue
 */
class __attribute__( ( visibility( "default" ) ) ) NCIniArrayValue {
   public:
    /**
     * @brief Construct a new NCIniArrayValue
     */
    NCIniArrayValue();

    /**
     * @brief Construct a new NCIniArrayValue
     *
     * @param value [IN] data
     * @param split [IN] data separator
     */
    NCIniArrayValue( const NCString &value, const CHAR split = ',' );

    /**
     * @brief Destroy the NCIniArrayValue object
     */
    ~NCIniArrayValue();

    /**
     * @brief set data(include separator)
     *
     * @param value [IN] data of value
     * @param split [IN] data separator
     * @return VOID
     */
    VOID set( const NCString &value, const CHAR split = ',' );

    /**
     * @brief get length of data
     *
     * @return INT32 length of data
     */
    INT32 length() const;

    /**
     * @brief Get the Integer from data
     *
     * @param keyindex [IN] index of data
     * @param ret      [OUT] integer type data
     * @return NC_BOOL True:sucess False:otherwise
     */
    NC_BOOL getInteger( UINT32 keyindex, INT32 &ret ) const;

    /**
     * @brief Get the Hex Integer form data
     *
     * @param keyindex [IN] index of data
     * @param ret      [OUT] hex integer type data
     * @return NC_BOOL True:sucess False:otherwise
     */
    NC_BOOL getHexInteger( UINT32 keyindex, INT32 &ret ) const;

    /**
     * @brief Get the String from data
     *
     * @param keyindex [IN] index of data
     * @param ret      [OUT] string type data
     * @return NC_BOOL True:sucess False:otherwise
     */
    NC_BOOL getString( UINT32 keyindex, NCString &ret ) const;

    /**
     * @brief Get the Boolean from data
     *
     * @param keyindex [IN] index of data
     * @param ret      [OUT] boolean type data
     * @return NC_BOOL True:sucess False:otherwise
     */
    NC_BOOL getBoolean( UINT32 keyindex, NC_BOOL &ret ) const;

    /**
     * @brief Get the Float from data
     *
     * @param keyindex [IN] index of data
     * @param ret      [OUT] float type data
     * @return NC_BOOL True:sucess False:otherwise
     */
    NC_BOOL getFloat( UINT32 keyindex, FLOAT &ret ) const;

    /**
     * @brief Get the Double from data
     *
     * @param keyindex [IN] index of data
     * @param ret      [OUT] double type data
     * @return NC_BOOL True:sucess False:otherwise
     */
    NC_BOOL getDouble( UINT32 keyindex, DOUBLE &ret ) const;

    /**
     * @brief Get the String data array from data
     *
     * @param array [OUT] container for data
     * @param len   [IN] number of data to get
     * @return NC_BOOL True:sucess False:otherwise
     */
    NC_BOOL getString( NCString *const array, UINT32 len ) const;

    /**
     * @brief Get the Integer data array from data
     *
     * @param array [OUT] container for data
     * @param len   [IN] number of data to get
     * @return NC_BOOL True:sucess False:otherwise
     */
    NC_BOOL getInteger( INT32 *const array, UINT32 len ) const;

    /**
     * @brief Get the Hex Integer data array from data
     *
     * @param array [OUT] container for data
     * @param len   [IN] number of data to get
     * @return NC_BOOL True:sucess False:otherwise
     */
    NC_BOOL getHexInteger( INT32 *const array, UINT32 len ) const;

    /**
     * @brief Get the Boolean data array from data
     *
     * @param array [OUT] container for data
     * @param len   [IN] number of data to get
     * @return NC_BOOL True:sucess False:otherwise
     */
    NC_BOOL getBoolean( NC_BOOL *const array, UINT32 len ) const;

    /**
     * @brief Get the Float data array from data
     *
     * @param array [OUT] container for data
     * @param len   [IN] number of data to get
     * @return NC_BOOL True:sucess False:otherwise
     */
    NC_BOOL getFloat( FLOAT *const array, UINT32 len ) const;

    /**
     * @brief Get the Double data array from data
     *
     * @param array [OUT] container for data
     * @param len   [IN] number of data to get
     * @return NC_BOOL True:sucess False:otherwise
     */
    NC_BOOL getDouble( DOUBLE *const array, UINT32 len ) const;

   private:
    VOID    count();
    NC_BOOL getArray( NC_IniArrayWriter writer, VOID *const container, UINT32 len ) const;

   private:
    UINT32   m_len;
    NCString m_value;
    CHAR     m_split;
};

/**
 * @brief The key iterator of a specified section
 *
 * @class NCIniIterator
 *
 * @section examples How to use Iterator
 *        \code
            NCIniFile cFile;
            ...
            // open the file
            ...
            NCIniIterator& iter = cFile.iterator(SecName);
            for(; !iter.end(); ++iter)
            {
                // do some thing, for example:
                NCIniKey key;
                iter.current(key);    // get current key
                iter.insert(key);     // insert a key
            }
            ...
        \endcode
 *
 */
class __attribute__( ( visibility( "default" ) ) ) NCIniIterator {
   public:
    /**
     * @brief Constructor
     */
    NCIniIterator();

    /**
     * @brief Copy constructor
     */
    NCIniIterator( const NCIniIterator & );

    /**
     * @brief Operator =
     */
    NCIniIterator &operator=( const NCIniIterator &it );

    /**
     * @brief Constructor
     */
    NCIniIterator( ncsp<NCIniLineList>::sp list, const NCString &section );

    /**
     * @brief Constructor
     */
    NCIniIterator( ncsp<NCIniLineList>::sp list, const NCString &section, NC_BOOL readOnly );

    /**
     * @brief Set to the first key
     */
    VOID reset();

    /**
     * @brief Get the key on current position
     *
     * @param[out] key    The reference to a NCIniKey instance
     * @return If the function call succeeds, it returns NC_TRUE.
               Otherwise, it returns NC_FALSE.
     */
    NC_BOOL current( NCIniKey &key ) const;

    /**
     * @brief Get the line string on current position
     *
     * @param[out] line    The string of the line
     * @return If the function call succeeds, it returns NC_TRUE.
               Otherwise, it returns NC_FALSE.
     */
    NC_BOOL current( NCString &line ) const;

    /**
     * @brief insert a key after current position
     *
     * @param[in] key    The reference to a NCIniKey instance
     * @return If the function call succeeds, it returns NC_TRUE.
               Otherwise, it returns NC_FALSE.
     */
    NC_BOOL insert( NCIniKey &key );

    /**
     * @brief Overload the left increment operator
     */
    NCIniIterator &operator++();

    /**
     * @brief Overload the right increment operator
     */
    const NCIniIterator operator++( INT32 param );

    /**
     * @brief Is the end NCIniKey?
     */
    NC_BOOL end() const;

    /**
     * @brief Has the next NCIniKey?
     */
    NC_BOOL hasNext() const;

   private:
    ncsp<NCIniLineList>::sp m_lineList;  ///< The list of file line.
    INT32                   m_first;     // the position of the first key in the section
    INT32                   m_current;   // the current key's position
    NC_BOOL                 m_readOnly;  // read only
};

/**
 * @brief The operator of a INI line list
 *
 * @class NCIniOperator
 *
 */
class __attribute__( ( visibility( "default" ) ) ) NCIniOperator {
   public:
    /**
     * @brief Default constructor
     */
    NCIniOperator();

    /**
     * @brief Destructor
     */
    virtual ~NCIniOperator();

    /**
     * @brief Retrieve an integer from a key in the specified section
     *
     * @param[in] section    The name of the section containing the key name.
     * @param[in] key        The name of the key whose value is to be retrieved.
     * @param[out] ret       The integer equivalent of the string following the
     key name.\n
                             <b>You should set it to default value before calling
     this function.</b>
     * @return If the function call succeeds, it returns NC_TRUE. Otherwise, it
     returns NC_FALSE.
     *
     * @note If the key or the section does not exist, NC_FALSE will be returned,
     and ret is not changed.\n
     *       <b>The value of the key in .ini file should be between -2147483648
     and 2147483647.</b> Otherwise, the result may be wrong.
     */
    NC_BOOL getInteger( const NCString &section, const NCString &key, INT32 &ret ) const;

    /**
     * @brief Retrieve an hexadecimal integer from a key in the specified section
     *
     * @param[in] section    The name of the section containing the key name.
     * @param[in] key        The name of the key whose value is to be retrieved.
     * @param[out] ret       The integer equivalent of the string following the
     key name.\n
                             <b>You should set it to default value before calling
     this function.</b>
     * @return If the function call succeeds, it returns NC_TRUE. Otherwise, it
     returns NC_FALSE.
     *
     * @note If the key or the section does not exist, NC_FALSE will be returned,
     and ret is not changed.\n
     *       <b>The value of the key in .ini file should not exceed
     0xFFFFFFFF.</b> Otherwise, the result may be wrong.
     */
    NC_BOOL getHexInteger( const NCString &section, const NCString &key, INT32 &ret ) const;

    /**
     * @brief Retrieve a string from a key in the specified section
     *
     * @param[in] section    The name of the section containing the key name.
     * @param[in] key        The name of the key whose value is to be retrieved.
     * @param[out] ret       The string following the key name.\n
                             <b>You should set it to default value before calling
     this function.</b>
     * @return If the function call succeeds, it returns NC_TRUE. Otherwise, it
     returns NC_FALSE.
     *
     * @note If the key or the section does not exist or the key value is void,
     NC_FALSE will be returned, and ret is not changed.\n
             <b>All space charactors before and after the string will be
     trimed.</b>
     */
    NC_BOOL getString( const NCString &section, const NCString &key, NCString &ret ) const;

    /**
     * @brief Retrieve a boolean from a key in the specified section
     *
     * @param[in] section    The name of the section containing the key name.
     * @param[in] key        The name of the key whose value is to be retrieved.
     * @param[out] ret       The boolean equivalent of the string following the
     key name.\n
                             <b>You should set it to default value before calling
     this function.</b>
     * @return If the function call succeeds, it returns NC_TRUE. Otherwise, it
     returns NC_FALSE.
     *
     * @note If the key or the section does not exist, NC_FALSE will be returned,
     and ret is not changed.\n
     *       <b>Any value, that is not equal to zero, of the key in .ini file will
     be seen as TRUE.</b>
     */
    NC_BOOL getBoolean( const NCString &section, const NCString &key, NC_BOOL &ret ) const;

    /**
     * @brief Retrieve a float from a key in the specified section
     *
     * @param[in] section    The name of the section containing the key name.
     * @param[in] key        The name of the key whose value is to be retrieved.
     * @param[out] ret       The float equivalent of the string following the key
     name.\n
                             <b>You should set it to default value before calling
     this function.</b>
     * @return If the function call succeeds, it returns NC_TRUE. Otherwise, it
     returns NC_FALSE.
     *
     * @note If the key or the section does not exist, NC_FALSE will be returned,
     and ret is not changed.\n
     *       <b>The value of the key in .ini file should be between
     -3.402823466e+38F and 3.402823466e+38F.</b> Otherwise, the result may be
     wrong.
     */
    NC_BOOL getFloat( const NCString &section, const NCString &key, FLOAT &ret ) const;

    /**
     * @brief Retrieve a double float from a key in the specified section
     *
     * @param[in] section    The name of the section containing the key name.
     * @param[in] key        The name of the key whose value is to be retrieved.
     * @param[out] ret       The double float equivalent of the string following
     the key name.\n
                             <b>You should set it to default value before calling
     this function.</b>
     * @return If the function call succeeds, it returns NC_TRUE. Otherwise, it
     returns NC_FALSE.
     *
     * @note If the key or the section does not exist, NC_FALSE will be returned,
     and ret is not changed. \n
     *       <b>The value of the key in .ini file should be between
     -1.7976931348623158e+308 and 1.7976931348623158e+308.</b> Otherwise, the
     result may be wrong.
     */
    NC_BOOL getDouble( const NCString &section, const NCString &key, DOUBLE &ret ) const;

    /**
     * @brief Set an integer to a key in the specified section
     *
     * @param[in] section    The name of the section containing the key name.
     * @param[in] key        The name of the key whose value is to be set.
     * @param[in] value      The integer to set.
     * @return If the function call succeeds, it returns NC_TRUE. Otherwise, it
     * returns NC_FALSE.
     */
    NC_BOOL setInteger( const NCString &section, const NCString &key, const INT32 &value );

    /**
     * @brief Set an hexadecimal integer to a key in the specified section
     *
     * @param[in] section    The name of the section containing the key name.
     * @param[in] key        The name of the key whose value is to be set.
     * @param[in] value      The integer to set.
     * @return If the function call succeeds, it returns NC_TRUE. Otherwise, it
     * returns NC_FALSE.
     */
    NC_BOOL setHexInteger( const NCString &section, const NCString &key, const INT32 &value );

    /**
     * @brief Set a string to a key in the specified section
     *
     * @param[in] section    The name of the section containing the key name.
     * @param[in] key        The name of the key whose value is to be set.
     * @param[in] value      The string to set.
     * @return If the function call succeeds, it returns NC_TRUE. Otherwise, it
     * returns NC_FALSE.
     */
    NC_BOOL setString( const NCString &section, const NCString &key, const NCString &value );

    /**
     * @brief Set a boolean to a key in the specified section
     *
     * @param[in] section    The name of the section containing the key name.
     * @param[in] key        The name of the key whose value is to be set.
     * @param[in] value      The boolean value to set.
     * @return If the function call succeeds, it returns NC_TRUE. Otherwise, it
     * returns NC_FALSE.
     */
    NC_BOOL setBoolean( const NCString &section, const NCString &key, const NC_BOOL &value );

    /**
     * @brief Set a float to a key in the specified section
     *
     * @param[in] section    The name of the section containing the key name.
     * @param[in] key        The name of the key whose value is to be set.
     * @param[in] value      The float to set.
     * @return If the function call succeeds, it returns NC_TRUE. Otherwise, it
     * returns NC_FALSE.
     */
    NC_BOOL setFloat( const NCString &section, const NCString &key, const FLOAT &value );

    /**
     * @brief Set a double to a key in the specified section
     *
     * @param[in] section    The name of the section containing the key name.
     * @param[in] key        The name of the key whose value is to be set.
     * @param[in] value      The double float to set.
     * @return If the function call succeeds, it returns NC_TRUE. Otherwise, it
     * returns NC_FALSE.
     */
    NC_BOOL setDouble( const NCString &section, const NCString &key, const DOUBLE &value );

    /**
     * @brief Delete a key in the specified section
     *
     * @param[in] section    The name of the section containing the key name.
     * @param[in] key        The name of the key to be deleted.
     * @return If the function call succeeds, it returns NC_TRUE. Otherwise, it
     * returns NC_FALSE.
     *
     * @remarks If the the key does not exist, return NC_TRUE;
     */
    NC_BOOL delKey( const NCString &section, const NCString &key );

    /**
     * @brief Delete a section
     *
     * @param[in] section    The name of the section to be deleted.
     * @return If the function call succeeds, it returns NC_TRUE. Otherwise, it
     * returns NC_FALSE.
     *
     * @remarks If the the section does not exist, return NC_TRUE;
     */
    NC_BOOL delSection( const NCString &section );

    /**
     * @brief Check whether a section or key exists
     *
     * @param[in] section    The name of the section to check.
     * @param[in] key        The name of the key to to check.
     * @return If the section or the key exists, it returns NC_TRUE. Otherwise, it
     returns NC_FALSE.
     *
     * @note If you want to check a section,please set the <i>key</i> to empty
     string.
            And NC_TRUE will be returned even though there's no key in the
     section.\n
            If you want to check a key,the section must be specified.
            And NC_TRUE will be returned even though the key has no value, just
     like "key=".
     */
    NC_BOOL exist( const NCString &section, const NCString &key ) const;

    /**
     * @brief Get the key iterator of a specified section
     *
     * @param[in] section    The name of the section
     * @return The reference to the key iterator of the section
     */
    NCIniIterator iterator( const NCString &section ) const;

   public:
    /**
     * @brief Make the section line
     *
     * @param[in] name     The name of the section.
     * @param[out] line    The line string.
     * @return If the function succeeds, it returns NC_TRUE. Otherwise, it returns
     * NC_FALSE.
     */
    static NC_BOOL getSecLine( const NCString &name, NCString &line );

    /**
     * @brief Make the key line
     *
     * @param[in] name     The name of the key.
     * @param[in] value    The value of the key.
     * @param[out] line    The line string.
     * @return If the function succeeds, it returns NC_TRUE. Otherwise, it returns
     * NC_FALSE.
     */
    static NC_BOOL getKeyLine( const NCString &name, const NCString &value, NCString &line );

    VOID setLineList( ncsp<NCIniLineList>::sp list );

   private:
    ncsp<NCIniLineList>::sp m_lineList;  // The list of file line.

    NCIniOperator( const NCIniOperator & );
    NCIniOperator &operator=( const NCIniOperator & );
};
OSAL_END_NAMESPACE
#endif /* NCINIOPERATOR_H */
/* EOF */
