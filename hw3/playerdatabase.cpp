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
    this->first_season = nullptr;
    this->current_season = nullptr;
}

string PlayerDatabase::get_season() {
    return current_season;
}

void PlayerDatabase::set_first_season(string season) {
    this->first_season = season;
}

void PlayerDatabase::set_season(string season) {
    this->current_season = season;
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

void PlayerDatabase::rotate_left(Player* current){
    Player* child = current->right_child;
    current->right_child = child->left_child;
    // If the condition is R - L, set the "child" as the median
    if (child->left_child != nil){
        child->left_child->parent = current;
    }
    // Cascade the parent down
    child->parent = current->parent;
    if (current->parent == nullptr){    // If the current is root, swap nodes
        root = child;
    } else if (current == current->parent->left_child) {    // If the left child is to be changed
        current->parent->left_child = child;
    } else {    // If the right child is to be changed
        current->parent->right_child = child;
    }
    // Swap nodes toward left
    child->left_child = current;
    current->parent = child;
}

void PlayerDatabase::rotate_right(Player* current){
    Player* child = current->left_child;
    current->left_child = child->right_child;
    // If the condition is L - R, set the "child" as the median
    if (child->right_child != nil){
        child->right_child->parent = current;
    }
    // Cascade the parent down
    child->parent = current->parent;
    if (current->parent == nullptr){    // If the current is root, swap nodes
        root = child;
    } else if (current == current->parent->right_child) {    // If the right child is to be changed
        current->parent->right_child = child;
    } else {    // If the left child is to be changed
        current->parent->left_child = child;
    }
    // Swap nodes toward right
    child->right_child = current;
    current->parent = child;
}

void PlayerDatabase::fix_database(Player* player_in) {
    Player* temp_player;

    while (player_in->parent->color == 1 && player_in != root){
        if (player_in->parent == player_in->parent->parent->right_child) {
            temp_player = player_in->parent->parent->left_child;    // Uncle
            if (temp_player->color == 1) {
                // Case 3.1
                temp_player->color = 0;
                player_in->parent->color = 0;
                player_in->parent->parent->color = 1;
                player_in = player_in->parent->parent;
            } else {
                if (player_in == player_in->parent->left_child){
                    // Case 3.2.2
                    player_in = player_in->parent;
                    rotate_right(player_in);
                }
                // Case 3.2.1
                player_in->parent->color = 0;
                player_in->parent->parent->color = 1;
                rotate_left(player_in->parent->parent);
            }
        } else {
            temp_player = player_in->parent->parent->right_child;    // Uncle
            if (temp_player->color == 1){
                // Case 3.2.2
                player_in = player_in->parent;
                rotate_left(player_in);
            }
            // Case 3.2.1
            player_in->parent->color = 0;
            player_in->parent->parent->color = 1;
            rotate_right(player_in->parent->parent);
        }
    }
    // In case the root is replaced, make sure it is black
    root->color = 0;
}

PlayerDatabase::~PlayerDatabase() {
    // TODO: Implement destructor
}
