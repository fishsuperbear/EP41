/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description: extwdg
*/
#ifndef EXTWDG_QUEST_ANSWER_H_
#define EXTWDG_QUEST_ANSWER_H_

#include <cstdint>
#include <map>
#include "extwdg_logger.h"
namespace hozon {
namespace netaos {
namespace extwdg {

static const std::map<std::uint32_t, std::uint32_t> qa_table = 
{
    {0x0,0xFF0FF000},
    {0x1,0xB040BF4F},
    {0x2,0xE919E616},
    {0x3,0xA656A959},
    {0x4,0x75857A8A},
    {0x5,0x3ACA35C5},
    {0x6,0x63936C9C},
    {0x7,0x2CDC23D3},
    {0x8,0xD222DD2D},
    {0x9,0x9D6D9262},
    {0xA,0xC434CB3B},
    {0xB,0x8B7B8474},
    {0xC,0x58A857A7},
    {0xD,0x17E718E8},
    {0xE,0x4EBE41B1},
    {0xF,0x01F10EFE}
};

class QuestAndAnswer
{
public:
    QuestAndAnswer() {}
    ~QuestAndAnswer() {}
    uint32_t Quest(uint32_t quest);
    
};

}  // namespace extwdg
}  // namespace netaos
}  // namespace hozon

#endif // EXTWDG_QUEST_ANSWER_H_