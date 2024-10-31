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
 * @file NCFilePubDef.h
 * @brief
 * @date 2020-05-08
 *
 */

#ifndef INCLUDE_NCORE_NCFILEPUBDEF_H_
#define INCLUDE_NCORE_NCFILEPUBDEF_H_

#include "osal/ncore/NCTypesDefine.h"

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif

OSAL_BEGIN_NAMESPACE

#define PATH_SEPARATOR "/"

/**
 * @brief Definition of error code
 */
enum NCFileError : UINT32 {
    // /< no error
    NC_FER_NoError = 0U,
    // /< parameter is invalid
    NC_FER_InvalidParam,
    // /< path name is invalid
    NC_FER_InvalidPathName,
    // /< file is not opened
    NC_FER_FileNotOpen,
    // /< no access right or permission denied
    NC_FER_NoAccess,
    // /< the file or directory does not exist
    NC_FER_NotExists,
    // /< the file or directory has already existed
    NC_FER_AlreadyExists,
    // /< the path is a directory
    NC_FER_IsDirectory,
    // /< the path is not a directory
    NC_FER_IsNotDirectory,
    // /< the disk is full, and there's no remaining space
    NC_FER_NoSpace,
    // /< there's no more file
    NC_FER_NoMoreFile,
    // /< the directory is not empty
    NC_FER_NotEmpty,
    // /< the file or directory is being used by other process
    NC_FER_IsUsed,
    // /< there's no more memory
    NC_FER_NoMemory,
    // /< link is eloop, too many symbols link, include cyclic symbols link
    NC_FER_Eloop,
    // /< other unknown error
    NC_FER_Unknown
};

/**
 * @brief Definition of file's open mode
 *
 * You can use one or a combination of these modes.
 * If you use a combination, they should be combined with bitwise-<i>or</i>(|).
 */
enum NCFileOpenMode : UINT32 {
    // /< not open.
    NC_FOM_NotOpen = 0x0000U,
    // /< can be used alone.
    NC_FOM_ReadOnly = 0x0001U,
    // /< can be used alone.
    NC_FOM_WriteOnly = 0x0002U,
    // NC_FOM_ReadWrite    = NC_FOM_ReadOnly | NC_FOM_WriteOnly,
    NC_FOM_ReadWrite = 0x0003U,
    // /< can be used alone.
    NC_FOM_Append = 0x0004U,
    // /< can not be used alone, should have WriteOnly mode,
    // /< but not Append mode.
    NC_FOM_Truncate = 0x0008U,
    // /< can not be used alone, should have ReadOnly,
    // /< WriteOnly or Append mode.
    // NC_FOM_Text       = 0x0010,
    // /< can not be used alone, should have ReadOnly,
    // /< WriteOnly or Append mode.
    NC_FOM_NoBuffer = 0x0010U
};

/**
 * @brief Type definition of file's user
 */
enum NCFileUser : UINT32 {
    NC_FOW_Owner = 0x0U,  // /< owner of the file
    NC_FOW_Group,         // /< user's group of the file
    NC_FOW_Other          // /< other user of the file
};

/**
 * @brief Definition of file's permission
 */
enum NCFilePermission : UINT32 {
    // /< the owner can read the file
    NC_FPM_OwnerRead = 0x0400U,
    // /< the owner can write the file
    NC_FPM_OwnerWrite = 0x0200U,
    // /< the owner can execute the file
    NC_FPM_OwnerExe = 0x0100U,

    // /< the member in owner's group can read the file
    NC_FPM_GroupRead = 0x0040U,
    // /< the member in owner's group can write the file
    NC_FPM_GroupWrite = 0x0020U,
    // /< the member in owner's group can execute the file
    NC_FPM_GroupExe = 0x0010U,

    // /< other user can read the file
    NC_FPM_OtherRead = 0x0004U,
    // /< other user can write the file
    NC_FPM_OtherWrite = 0x0002U,
    // /< other user can execute the file
    NC_FPM_OtherExe = 0x0001U,

    // /< permission mask
    NC_FPM_Mask = 0xFFFFU,
};

/**
 * @brief Definition of file's type
 */
enum NCFileType : UINT32 {
    // /< Normal file
    NC_FTP_File = 0x010000U,
    // /< Directory
    NC_FTP_Directory = 0x020000U,
    // /< Link file
    NC_FTP_Link = 0x040000U,
    // /< Device
    NC_FTP_Device = 0x080000U,
    // /< Pipe file
    NC_FTP_Pipe = 0x100000U,
    // /< Socket
    NC_FTP_Socket = 0x200000U,
    // /< Type mask
    NC_FTP_Mask = 0xFF0000U
};

/**
 * @brief Type definition of file's attribute
 */
enum NCFileAttribute : UINT32 {
    // /< The file is a normal file
    NC_FAB_Normal = 0x01000000U,
    // /< The file is a hidden file
    NC_FAB_Hidden = 0x02000000U,
    // /< The file is a system file
    NC_FAB_System = 0x04000000U,
    // /< The file does not exist
    NC_FAB_NotExist = 0x08000000U,
    // /< Attribute mask
    NC_FAB_Mask = 0xFF000000U
};

/**
 * @brief Definition of filemap's prot
 */
enum NCFileMapProt : UINT32 {
    // /< Pages may be read.
    NC_FM_PROT_READ = 0x010000U,
    // /< Pages may be write.
    NC_FM_PROT_WRITE = 0x020000U,
    // /< Pages may be executed.
    NC_FM_PROT_EXEC = 0x040000U,
    // /< Pages may not be accessed.
    NC_FM_PROT_NONE = 0x080000U,
    // /< file map prot mask.
    NC_FM_PROT_Mask = 0xFF0000U
};

/**
 * @brief Definition of filemap's flags
 */
enum NCFileMapFlags : UINT32 {
    // /< Create a private copy-on-write mapping,
    // /< write operation is not visible to other process,
    // /< also content it write will not write to the disk
    NC_FM_FLAGS_PRIVATE = 0x010000U,
    // /< Share this mapping, visible to other process
    NC_FM_FLAGS_SHARED = 0x020000U,
    // /< file map flag mask.
    NC_FM_FLAGS_MASK = 0xFF0000U
};

/**
 * @brief Definition of file's flag
 */
enum NCFileFlag : UINT32 {
    // /< the owner can read the file
    NC_FFG_PermOwnerRead = NC_FPM_OwnerRead,
    // /< the owner can write the file
    NC_FFG_PermOwnerWrite = NC_FPM_OwnerWrite,
    // /< the owner can execute the file
    NC_FFG_PermOwnerExe = NC_FPM_OwnerExe,
    // /< the member in owner's group can read the file
    NC_FFG_PermGroupRead = NC_FPM_GroupRead,
    // /< the member in owner's group can write the file
    NC_FFG_PermGroupWrite = NC_FPM_GroupWrite,
    // /< the member in owner's group can execute the file
    NC_FFG_PermGroupExe = NC_FPM_GroupExe,
    // /< other user can read the file
    NC_FFG_PermOtherRead = NC_FPM_OtherRead,
    // /< other user can write the file
    NC_FFG_PermOtherWrite = NC_FPM_OtherWrite,
    // /< other user can execute the file
    NC_FFG_PermOtherExe = NC_FPM_OtherExe,

    // /< Normal file
    NC_FFG_TypeFile = NC_FTP_File,
    // /< Directory
    NC_FFG_TypeDirectory = NC_FTP_Directory,
    // /< Link file
    NC_FFG_TypeLink = NC_FTP_Link,
    // /< Device
    NC_FFG_TypeDevice = NC_FTP_Device,
    // /< Pipe file
    NC_FFG_TypePipe = NC_FTP_Pipe,
    // /< Socket
    NC_FFG_TypeSocket = NC_FTP_Socket,

    // /< The file is a normal file
    NC_FFG_AttrNormal = NC_FAB_Normal,
    // /< The file is a hidden file
    NC_FFG_AttrHidden = NC_FAB_Hidden,
    // /< The file is a system file
    NC_FFG_AttrSystem = NC_FAB_System,
    // /< The file does not exist
    NC_FFG_AttrNotExist = NC_FAB_NotExist,

    // /< permission mask
    NC_FFG_PermsMask = NC_FPM_Mask,
    // /< Type mask
    NC_FFG_TypesMask = NC_FTP_Mask,
    // /< Attribute mask
    NC_FFG_AttrsMask = NC_FAB_Mask,

    // /< All info mask
    // NC_FFG_AllInfo = (NC_FFG_PermsMask | NC_FFG_TypesMask | NC_FFG_AttrsMask)
    NC_FFG_AllInfo = 0xFFFFFFFFU
};

/**
 * @brief Type definition of file's time
 */
enum NCFileTime : UINT32 {
    // /< File's creation time
    NC_FTM_Creation = 0U,
    // /< File's last-write time
    NC_FTM_LastWrite,
    // /< File's last-access time
    NC_FTM_LastAccess
};

/**
 * @brief Definition of file's filter
 */
enum NCFileFilter : UINT32 {
    // /< no filter
    NC_FFT_NoFilter = 0x0000U,

    // /< the files should be directories
    NC_FFT_Dirs = 0x0001U,
    // /< the files should be normal files
    NC_FFT_Files = 0x0002U,
    // /< the files should be drives
    NC_FFT_Drives = 0x0004U,
    // /< there should not be symbol links
    NC_FFT_NoSymLinks = 0x0008U,
    // /< all entries
    // NC_FFT_AllEntries  = (NC_FFT_Dirs| NC_FFT_Files
    //                         | NC_FFT_Drives),
    NC_FFT_AllEntries = 0x0007U,
    // /< file type mask
    NC_FFT_TypeMask = 0x000FU,

    // /< the files should be readable
    NC_FFT_Readable = 0x0010U,
    // /< the files should be writable
    NC_FFT_Writable = 0x0020U,
    // /< the files should be executable
    NC_FFT_Executable = 0x0040U,
    // /< file's permission mask
    NC_FFT_PermMask = 0x0070U,

    // /< the files should be modified
    NC_FFT_Modified = 0x0080U,
    // /< the files should not be hidden files
    NC_FFT_NoHidden = 0x0100U,
    // /< the files should not be system files
    NC_FFT_NoSystem = 0x0200U,
    // /< file's access mask
    NC_FFT_AccessMask = 0x03F0U,

    // /< all directories
    NC_FFT_AllDirs = 0x0400U,
    // /< the file name is case-sensitive
    NC_FFT_CaseSensitive = 0x0800U,
    // /< there should not be . and ..
    NC_FFT_NoDotAndDotDot = 0x1000U
};

/**
 * @brief Definition of file's sort flag
 */
enum NCFileSortFlag : UINT32 {
    // /< no sort
    NC_FSF_NoSort = 0x0000U,
    // /< sort by name
    NC_FSF_ByName = 0x0001U,
    // /< sort by time
    NC_FSF_ByTime = 0x0002U,
    // /< sort by size
    NC_FSF_BySize = 0x0004U,

    // /< the directories should be at first
    NC_FSF_DirFirst = 0x0010U,
    // /< the directories should be at last
    NC_FSF_DirLast = 0x0020U
};

/**
 * @brief Definition of file's seek mode
 */
enum NCFileSeekMode : UINT32 {
    // /< the starting point is 0 (zero) or the beginning of the file.
    NC_FSM_Begin = 0U,
    // /< the starting point is the current position of the file pointer.
    NC_FSM_Current,
    // /< the starting point is the current end-of-file position.
    NC_FSM_End
};

/**
 * @brief Definition of filesystem's type
 */
enum NCFileSystemType : UINT32 {
    // /< the normal filesystem such as /dev/hd1, /dev/hd0t177.
    NC_FST_Normal = 0U,
    // /< the RAM filesystem such as /dev/shmem, /tmp.
    NC_FST_RAM
};

OSAL_END_NAMESPACE
#endif  // INCLUDE_NCORE_NCFILEPUBDEF_H_
/* EOF */
