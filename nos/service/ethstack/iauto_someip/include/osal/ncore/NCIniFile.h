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
 * @file NCIniFile.h
 * @brief
 * @date 2020-05-08
 *
 */

#ifndef NCINIFILE_H
#define NCINIFILE_H

#include "osal/ncore/NCFile.h"
#include "osal/ncore/NCIniOperator.h"
#include "osal/ncore/NCString.h"

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation (use a .cpp suffix)
#endif

OSAL_BEGIN_NAMESPACE
// external class declaration
class NCIniLineList;
class NCIniKey;
class NCIniIterator;
class NCIniOperator;
class NCFile;

// class declaration
class NCIniFile;

/**
 * @brief Ini file's open mode
 */
enum NCIniOpenMode {
    NCINI_OM_R,   ///< open in read only mode
    NCINI_OM_RW,  ///< open in read and write mode
};

/**
 * @class NCIniFile
 * @brief The class to manipulate initialization files(file's suffix is .ini)
 *
 * @section examples Examples
 *    This section shows some sample codes. For how to use NCIniFile, see the
 * page \ref Notice.
 *        \code
 *            NCString fileName = NCTEXT("\\Test\\Test.ini");
 *            NCString section = NCTEXT("TestSection");
 *            NCString key = NCTEXT("TestKey");
 *
 *            NCIniFile cIniFile;
 *        \endcode
 * @par open the ini file
 *        Before manipulating the file, you should open it first.
 *        \code
 *            if (!cIniFile.open(fileName)) {
 *                // failed, do some things
 *            }
 *            else {
 *                // successful, then you can manipulate the file
 *            }
 *        \endcode
 * @par Get the value of a key
 *        After opening the file successfully, you can get the value of a key.
 *        \code
 *            INT nTestInt = 1;    // this is the default value of the key
 *            if (cIniFile.getInteger(section,key,nTestInt)) {
 *                // successful, do some things
 *            }
 *            else {
 *                // failed, do some things
 *            }
 *        \endcode
 * @par Set the value of a key
 *        After opening the file successfully, you can set the value of a key.
 *        \code
 *            nTestInt = 2;
 *            if (cIniFile.setInteger(section,key,nTestInt)) {
 *                // successful, do some things
 *            }
 *            else {
 *                // failed, do some things
 *            }
 *        \endcode
 * @par Delete a key
 *        After opening the file successfully, you can delete of a key.
 *        \code
 *            if (cIniFile.delKey(section,key)) {
 *                // successful, do some things
 *            }
 *            else {
 *                // failed, do some things
 *            }
 *        \endcode
 * @par Delete a section
 *        After opening the file successfully, you can delete of a section.
 *        \code
 *            if (cIniFile.delSection(section)) {
 *                // successful, do some things
 *            }
 *            else {
 *                // failed, do some things
 *            }
 *        \endcode
 * @par flush the setting to file
 *        If you want to save the change to file immediately, you should do
 * this.
 *        Otherwise, the change will not be saved to file until you close the
 * file.
 *        \code
 *            if (cIniFile.flush()) {
 *                // successful, do some things
 *            }
 *            else {
 *                // failed, do some things
 *            }
 *        \endcode
 * @par close the file
 *        After doing some manipulation, you should close the file.
 *        \code
 *            if (cIniFile.close()) {
 *                // successful, do some things
 *            }
 *            else {
 *                // failed, do some things
 *            }
 *        \endcode
 */
/**
 * @brief
 *
 * @class NCIniFile
 */
class __attribute__( ( visibility( "default" ) ) ) NCIniFile {
   public:
    /**
     * @brief Default constructor
     */
    NCIniFile();

    /**
     * @brief Destructor
     */
    virtual ~NCIniFile();

    /**
     * @brief open the initialization file
     *
     * This function associate this class with the initialization file.
     * If the initialization file does not exist, this function will create it.
     *
     * @param[in] fileName    The <b>absolute</b> path of file.
     * @param[in] codePage    The file's encoding as code page.
     * @return If the function call succeeds, it returns NC_TRUE. Otherwise, it
     returns NC_FALSE.
     *
     * @note If the file has specified the encoding in the head of the file, then
     use the file's encoding.\n
             And the parameter <i>codePage</i> will become invalid.
     */
    NC_BOOL openFile( const NCString &fileName, const UINT32 &codePage = NC_CP_UTF8 );

    /**
     * @brief \overload
     *
     * This function associate this class with the initialization file.
     *
     * @param[in] fileName    The <b>absolute</b> path of file.
     * @param[in] mode        open mode. See \r NCIniOpenMode.
     * @param[in] codePage    The file's encoding as code page.
     * @return If the function call succeeds, it returns NC_TRUE. Otherwise, it
     returns NC_FALSE.
     *
     * @note If the file has specified the encoding in the head of the file, then
     use the file's encoding.\n
             And the parameter <i>codePage</i> will become invalid.
     */
    NC_BOOL openFile( const NCString &fileName, const NCIniOpenMode &mode,
                      const UINT32 &codePage = NC_CP_UTF8 );

    /**
     * @brief close the initialization file
     *
     * @param[in] VOID
     * @return If the function call succeeds, it returns NC_TRUE. Otherwise, it
     * returns NC_FALSE.
     *
     * @remarks This function will cause the change be flushed to file.
     */
    NC_BOOL closeFile();

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
     * @brief flush the setting to file
     *
     * @param[in] VOID
     * @return If the function call succeeds, it returns NC_TRUE. Otherwise, it
     returns NC_FALSE.
     *
     * @note If you don't call this function, the change of setting will not be
     saved to file immediately.\n
            So if you want to do that, you should call this function.
            Otherwise, the change will not be saved to file until you close the
     file.
     */
    NC_BOOL flush();

    /**
     * @brief Get the key iterator of a specified section
     *
     * @param[in] section    The name of the section
     * @return The reference to the key iterator of the section
     */
    NCIniIterator iterator( const NCString &section ) const;

   private:
    NCString m_fileName;               ///< The initialization file's full path.
    NCFile   m_file;                   ///< The pointer to an instance of the class that implements
                                       /// NCFile.
    NC_BOOL                 m_opened;  ///< Whether the file is opened.
    NC_BOOL                 m_readOnly;  ///< Whether the file is read-only.
    NCIniOperator           m_iniOpe;    ///< The operator of ini line.
    ncsp<NCIniLineList>::sp m_lineList;  ///< The list of file line.

    NCIniFile( const NCIniFile & );
    NCIniFile &operator=( const NCIniFile & );
};
OSAL_END_NAMESPACE
#endif /* NCINIFILE_H */
/* EOF */
