#include "sensor/nvs_adapter/nvs_utility.h"
#include "sensor/nvs_adapter/nvs_logger.h"

namespace hozon {
namespace netaos {
namespace nv { 

const char* NvsUtility::GetKeyName(int key) {
    struct KeyName {
        int key;
        const char* name;
    };
    struct KeyName key_name_mapping[] = {
        { NvSciBufGeneralAttrKey_RequiredPerm, "NvSciBufGeneralAttrKey_RequiredPerm"},
        { NvSciBufGeneralAttrKey_Types, "NvSciBufGeneralAttrKey_Types"},
        { NvSciBufGeneralAttrKey_NeedCpuAccess, "NvSciBufGeneralAttrKey_NeedCpuAccess"},
        { NvSciBufGeneralAttrKey_EnableCpuCache, "NvSciBufGeneralAttrKey_EnableCpuCache"},
        { NvSciBufImageAttrKey_TopPadding, "NvSciBufImageAttrKey_TopPadding"},
        { NvSciBufImageAttrKey_BottomPadding, "NvSciBufImageAttrKey_BottomPadding"},
        { NvSciBufImageAttrKey_LeftPadding, "NvSciBufImageAttrKey_LeftPadding"},
        { NvSciBufImageAttrKey_RightPadding, "NvSciBufImageAttrKey_RightPadding"},
        { NvSciBufImageAttrKey_Layout, "NvSciBufImageAttrKey_Layout"},
        { NvSciBufImageAttrKey_PlaneCount, "NvSciBufImageAttrKey_PlaneCount"},
        { NvSciBufImageAttrKey_PlaneColorFormat, "NvSciBufImageAttrKey_PlaneColorFormat"},
        { NvSciBufImageAttrKey_PlaneColorStd, "NvSciBufImageAttrKey_PlaneColorStd"},
        { NvSciBufImageAttrKey_PlaneBaseAddrAlign, "NvSciBufImageAttrKey_PlaneBaseAddrAlign"},
        { NvSciBufImageAttrKey_PlaneWidth, "NvSciBufImageAttrKey_PlaneWidth"},
        { NvSciBufImageAttrKey_PlaneHeight, "NvSciBufImageAttrKey_PlaneHeight"},
        { NvSciBufImageAttrKey_VprFlag, "NvSciBufImageAttrKey_VprFlag"},
        { NvSciBufImageAttrKey_ScanType, "NvSciBufImageAttrKey_ScanType"},
        { NvSciBufImageAttrKey_Size, "NvSciBufImageAttrKey_Size"}
    };

    for (size_t i = 0; i < sizeof(key_name_mapping) / sizeof(struct KeyName); ++i) {
        if (key_name_mapping[i].key == key) {
            return key_name_mapping[i].name;
        }
    }

    return "unknown key";
}

void NvsUtility::PrintBufAttrs(NvSciBufAttrList buf_attrs) {
    if (!buf_attrs) {
        return;
    }

    size_t slot_count = NvSciBufAttrListGetSlotCount(buf_attrs);
    for (size_t slot_index = 0; slot_index < slot_count; ++slot_index) {
        NvSciBufAttrKeyValuePair keyVals[] = {
            { NvSciBufGeneralAttrKey_RequiredPerm, NULL, 0 },
            { NvSciBufGeneralAttrKey_Types, NULL, 0 },
            { NvSciBufGeneralAttrKey_NeedCpuAccess, NULL, 0 },
            { NvSciBufGeneralAttrKey_EnableCpuCache, NULL, 0 },
            { NvSciBufImageAttrKey_TopPadding, NULL, 0 },
            { NvSciBufImageAttrKey_BottomPadding, NULL, 0 },
            { NvSciBufImageAttrKey_LeftPadding, NULL, 0 },
            { NvSciBufImageAttrKey_RightPadding, NULL, 0 },
            { NvSciBufImageAttrKey_Layout, NULL, 0 },
            { NvSciBufImageAttrKey_PlaneCount, NULL, 0 },
            { NvSciBufImageAttrKey_PlaneColorFormat, NULL, 0 },
            { NvSciBufImageAttrKey_PlaneColorStd, NULL, 0 },
            { NvSciBufImageAttrKey_PlaneBaseAddrAlign, NULL, 0 },
            { NvSciBufImageAttrKey_PlaneWidth, NULL, 0 },
            { NvSciBufImageAttrKey_PlaneHeight, NULL, 0 },
            { NvSciBufImageAttrKey_VprFlag, NULL, 0 },
            { NvSciBufImageAttrKey_ScanType, NULL, 0 },
            { NvSciBufImageAttrKey_Size, NULL, 0 },
            { NvSciBufImageAttrKey_PlaneAlignedSize, NULL, 0 },
            { NvSciBufImageAttrKey_Alignment, NULL, 0 },
        };
        
        for (size_t i = 0; i < sizeof(keyVals) / sizeof(NvSciBufAttrKeyValuePair); ++i) {
            // NvSciError err = NvSciBufAttrListGetAttrs(buf_attrs, &keyVals[i], 1);
            NvSciError err = NvSciBufAttrListSlotGetAttrs(buf_attrs, slot_index, &keyVals[i], 1);
            if (NvSciError_Success != err) {
                NVS_LOG_INFO << "Failed to obtain buffer attribute: " <<  GetKeyName(keyVals[i].key) << " from slot " << static_cast<int>(slot_index) << ", err: " << err;
                continue;
            }
        }

        for (size_t i = 0; i < sizeof(keyVals) / sizeof(NvSciBufAttrKeyValuePair); ++i) {
            if (keyVals[i].value && (keyVals[i].len > 0)) {
                printf("slot: %d, key: %s(%d), value_size: %d, value_ptr: 0x%016lx", static_cast<int>(slot_index), GetKeyName(keyVals[i].key),  keyVals[i].key, static_cast<int>(keyVals[i].len), reinterpret_cast<uint64_t>(keyVals[i].value));

                printf(", value: 0x");
                for (size_t j = 0; j < keyVals[i].len; ++j) {
                    printf("%02x ", *((char*)(keyVals[i].value) + j));
                }
                printf("\n");
            }

        }
    }
}

}
}
}