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
 * @file NCCheckFile.h
 * @brief
 * @date 2020-05-08
 *
 */

#ifndef NCCHECKFILE_H
#define NCCHECKFILE_H

#include "osal/ncore/NCFile.h"
#include "osal/ncore/NCString.h"

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation (use a .cpp suffix)
#endif

OSAL_BEGIN_NAMESPACE

/**
 * @brief Save and load file with power off protection
 *
 * This file would protect you from power off.\n
 * If there is a power off during you saving data to file,
 * you would get the last right version of this data next time you load it.
 *
 * @par Example
 * @anchor CKFEX
 * The following code is an example of
 * how to use this class to load and save data.
        \code
        NCCheckFile cCheckFile("user/A/aa.cfg");

        // Allocate a memory to store the data.
        UINT8 *pData = new UINT8[1024];

        // Load data from the file.
        if(!cCheckFile.loadData(pData, 1024)){
            // Load failed.
        }

        // Save data to the file.
        if(!cCheckFile.saveData(pData, 1024)){
            // Save failed.
        }

        delete[] pData;
        \endcode
 *
 */
class __attribute__( ( visibility( "default" ) ) ) NCCheckFile {
   public:
    /**
     * @brief Construct a new NCCheckFile object
     *
     * @param strFileName the string of file name
     */
    NCCheckFile( const NCString &strFileName );

    /**
     * @brief Destroy the NCCheckFile object
     *
     */
    virtual ~NCCheckFile();

    // See Base class for more information.

    /**
     * @brief read data from file
     *
     * @param pData the buffer
     * @param dwSize the size of buffer
     * @return NC_BOOL true: sucess false: otherwise
     */
    virtual NC_BOOL loadData( UINT8 *const pData, UINT32 dwSize );

    /**
     * @brief save the to file
     *
     * @param pData the data
     * @param dwSize the size of data
     * @return NC_BOOL true: sucess false: otherwise
     */
    virtual NC_BOOL saveData( const UINT8 *const pData, UINT32 dwSize );

    /**
     * @brief make directory
     *
     * @return NC_BOOL true: sucess false: otherwise
     */
    virtual NC_BOOL makeDir();

    /**
     * @brief Get the Data Size object
     *
     * @return UINT32 the data size
     */
    virtual UINT32 getDataSize();

   protected:
    /**
     * @brief Write data to temp area
     *
     * @param[in] pData  The data's pointer
     * @param[in] dwSize The data's size
     * @param[out] the the temp data's offset from file head
     * @return Whether the function call is successful.
     * @retval NC_TRUE Function call succeeds.
     * @retval NC_FALSE Function call fails
     * @pre The file must have been opened yet.
     */
    NC_BOOL writeTempData( const UINT8 *const pData, UINT32 dwSize, UINT32 &nOffset );

    /**
     * @brief Copy the data from the temp data area
     *
     * @param[in] dwSize   The data's size
     * @param[in] nOffset The temp data's offset from file head
     * @return Whether the function call is successful.
     * @retval NC_TRUE Function call succeeds.
     * @retval NC_FALSE Function call fails
     * @pre The file must have been opened yet.
     */
    NC_BOOL updateData( UINT32 dwSize, UINT32 nOffset );

    /**
     * @brief Move data from temp area
     *
     * @param[in] pFile A pointer that points to a file derived from NCFile.
     * @param[in] dwDataSize The data's size to move
     * @param[in] nOffset The temp data's offset
                    from current file pointer position
     * @return Whether the function call is successful.
     * @retval NC_TRUE Function call succeeds.
     * @retval NC_FALSE Function call fails.
     * @note The data's size should not be more than DATA_COPY_SIZE.
     */
    NC_BOOL moveFileData( NCFile *pFile, UINT32 dwDataSize, INT32 nOffset ) const;

    /**
     * @brief Get data from the file
     *
     * @param[out] pBuffer The pointer of the buffer to hold the data
     * @param[in]  dwBufferSize The size of the buffer
     * @return Whether the function call is successful.
     * @retval NC_TRUE Function call succeeds.
     * @retval NC_FALSE Function call fails.
     */
    NC_BOOL getData( UINT8 *const pBuffer, UINT32 dwBufferSize );

    /**
     * @brief Create a file header
     *
     * This function saves the new file header and
     *  empty data with the specified size to the file.
     *
     * @param[in] dwSize Size of the data
     * @return Whether the function call is successful.
     * @retval NC_TRUE Function call succeeds.
     * @retval NC_FALSE Function call fails
     * @pre The file must have been opened yet.
     */
    NC_BOOL createFileHeader( UINT32 dwSize );

   private:
    NCFile   m_file;
    NCString m_fileName;
};
OSAL_END_NAMESPACE
#endif /* NCCHECKFILE_H */
/* EOF */
