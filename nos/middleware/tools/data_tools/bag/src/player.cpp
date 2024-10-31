#include "player.h"
#include "impl/player_impl.h"

namespace hozon {
namespace netaos {
namespace bag {

Player::Player() {
    player_impl_ = std::make_unique<PlayerImpl>();
};

Player::~Player(){
    if (player_impl_) {
        player_impl_ = nullptr;
    }
};

void Player::Start(const PlayerOptions& playerOptions) {
    player_impl_->Start(playerOptions);
};

void Player::Stop() {
    player_impl_->Stop();
};

void Player::Pause() {
    player_impl_->Pause();
};

void Player::Resume() {
    player_impl_->Resume();
};

void Player::FastForward(int interval_tiem) {
    player_impl_->FastForward(interval_tiem);
};

void Player::Rewind(int interval_tiem) {
    player_impl_->Rewind(interval_tiem);
};

}  // namespace bag
}  //namespace netaos
}  //namespace hozon
