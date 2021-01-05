//
// Created by batuhanfaik on 04/01/2021.
//

#include <iostream>
#include "player.h"

using namespace std;

Player::Player() {
    full_name = "NIL";
    team = "NIL";
    rebound = -1;
    assist = -1;
    point = -1;
    color = 0;
    parent = nullptr;
    right_child = nullptr;
    left_child = nullptr;
}

Player::Player(string full_name, string team, int rebound, int assist, int point) {
    this->full_name = full_name;
    this->team = team;
    this->rebound = rebound;
    this->assist = assist;
    this->point = point;
    // Initialize a red node with nil children and null pointers
    this->color = 1;
    this->parent = nullptr;
    this->left_child = nullptr;
    this->right_child = nullptr;
}


Player* Player::get_null_player() {
    auto* tmp = new Player;
    return tmp;
}

void Player::nodify_player(Player* player,Player* null_player) {
    player->left_child = null_player;
    player->right_child = null_player;
}

void Player::print_player() {
    cout << full_name << " " << team << " " << rebound << " " << assist << " " << point << endl;
}