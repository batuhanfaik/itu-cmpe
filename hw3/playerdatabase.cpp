//
// Created by batuhanfaik on 04/01/2021.
//

#include "playerdatabase.h"
#include "player.h"

using namespace std;

PlayerDatabase::PlayerDatabase() {
    Player* null_player = Player::get_null_player();
    this->root = null_player;
    this->nil = null_player;
    this->season = nullptr;
}

string PlayerDatabase::get_season() {
    return season;
}

void PlayerDatabase::set_season(string season) {
    this->season = season;
}

void PlayerDatabase::add_player(Player* player_in) {
    // Turn the player into a tree node
    Player::nodify_player(player_in, nil);
    // Find where to insert the new player using binary search tree insertion (iterative method)
    Player* parent_player = nullptr;
    Player* current_player = root;
    while (current_player != nil){
        parent_player = current_player;
        if (player_in->full_name < current_player->full_name){
            // Player fits in the subtree of the left child
            current_player = current_player->left_child;
        } else {
            // Player fits in the subtree of the right child
            current_player = current_player->right_child;
        }
    }
    // Insert the new player
    player_in->parent = parent_player;
    if (!parent_player) {
        // If the tree (db) was empty new player is the root
        player_in->color = 0;    // Root of RB tree is always black
        root = player_in;
        return;
    } else if (player_in->full_name < parent_player->full_name) {
        parent_player->left_child = player_in;
    } else {
        parent_player->right_child = player_in;
    }
    // If the height is less than two the tree is complete, otherwise reset the red-black properties
    if (player_in->parent->parent){
        // Fix (reset) the database
        fix_database(player_in);
    } else {
        // Height is two
        return;
    }

}

void PlayerDatabase::fix_database(Player *) {
    // TODO: Implement fix rb tree

}
