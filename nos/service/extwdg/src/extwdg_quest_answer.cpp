/*
 * Copyright (c) Hozon Technologies Co., Ltd. 2023-2023. All rights reserved.
 * Description: main function definition
 */

#include "extwdg_quest_answer.h"

namespace hozon {
namespace netaos {
namespace extwdg {

uint32_t
QuestAndAnswer::Quest(uint32_t quest)
{
    EW_INFO << "QuestAndAnswer::Quest enter!";
    uint32_t answer;
    auto it = qa_table.find(quest);
    if(it != qa_table.end()) {
        answer = it->second;
        EW_INFO << "question is " << quest << "Answer is :"<< answer;
    } 
    else {
        answer = 0xFFFFFFFF;
    }
    return answer;
}

}  // namespace extwdg
}  // namespace netaos
}  // namespace hozon