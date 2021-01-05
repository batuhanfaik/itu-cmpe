//
// Created by batuhanfaik on 04/01/2021.
//

#include <iostream>
#include "playerdatabase.h"
#include "player.h"

using namespace std;

PlayerDatabase::PlayerDatabase() {
    Player* null_player = Player::get_null_player();
    this->root = null_player;
    this->nil = null_player;
    this->first_season = "";
    this->current_season = "";
    this->max_rebound = 0;
    this->max_rebound_name = "";
    this->max_assist = 0;
    this->max_assist_name = "";
    this->max_point = 0;
    this->max_point_name = "";
}

string PlayerDatabase::get_season() {
    return current_season;
}

void PlayerDatabase::set_season(string season) {
    string prev_season = current_season;
    this->current_season = season;
    if (first_season.empty()){    // If the first season is not set, set it
        first_season = season;
    } else {    // Print the season statistics and the database
        print_season_bests(prev_season);
        if (prev_season == first_season)    // Print the db for the first season
            print_database();
    }
}

void PlayerDatabase::update_season_bests(Player* player_in) {
    // Update season statistics
    if (player_in->rebound > max_rebound) {
        max_rebound = player_in->rebound;
        max_rebound_name = player_in->full_name;
    }
    if (player_in->assist > max_assist) {
        max_assist = player_in->assist;
        max_assist_name = player_in->full_name;
    }
    if (player_in->point > max_point) {
        max_point = player_in->point;
        max_point_name = player_in->full_name;
    }
}

void PlayerDatabase::add_player(Player* player_in) {
    // Update season statistics
    update_season_bests(player_in);
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
    } else if (player_in->full_name < parent_player->full_name) {    // Should be the left child
        parent_player->left_child = player_in;
    } else {    // Should be the right child
        parent_player->right_child = player_in;
    }
    // If the height of the node is less than or equal to two the tree is complete,
    // otherwise reset the red-black properties
    if (player_in->parent->parent){
        // Fix (reset) the database
        fix_database(player_in);
    } else {
        // Height is more then two
        return;
    }

}

void PlayerDatabase::update_player(Player* player_in) {
    // Turn the player into a tree node
    Player::nodify_player(player_in, nil);
    // Get the player to update
    Player* player_update;
    player_update = search_player(root, player_in->full_name);
    if (player_update == nil) {    // The player isn't in the database
        cout << "Player " << player_in->full_name << " can't be found!" << endl;
    } else {    // Update the player
        player_update->team = player_in->team;
        player_update->rebound += player_in->rebound;
        player_update->assist += player_in->assist;
        player_update->point += player_in->point;
    }
    // Update season statistics
    update_season_bests(player_update);
    delete (player_in);
}

bool PlayerDatabase::player_exists(Player* player_in) {
    if (search_player(root, player_in->full_name) == nil) {
        return false;
    } else {
        return true;
    }
}

Player* PlayerDatabase::search_player(Player* player_in, const string& player_name){
    if (player_name == player_in->full_name || player_in == nil){    // Found the player
        return player_in;
    } else if (player_name < player_in->full_name) {    // Go to left child
        return search_player(player_in->left_child, player_name);
    } else {    // Go to right child
        return search_player(player_in->right_child, player_name);
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
    /* While parent of new node is red (other conditions are necessary to stay within the tree)
    * Violation of 4th property needs to be resolved
    * The 4th property and the algorithm implemented below can be found on this course's textbook
    * Thomas H. Cormen, Introduction to Algorithms, 2001 in Chapter 13 */
    while (player_in->parent && player_in != root && player_in->parent->color == 1){
        Player* uncle = player_in->parent->parent->right_child;    // Set the uncle
        // Check new node's uncle to resolve 4th property
        if (player_in->parent == uncle) {    // If uncle is parent
            uncle = uncle->parent->left_child;    // Swap the uncle with its sibling
            if (uncle->color == 1) {    // If parent and the uncle are red
                uncle->color = 0;    // Uncle becomes black
                player_in->parent->color = 0;    // Parent becomes black
                player_in->parent->parent->color = 1;    // Grandparent becomes red
                player_in = player_in->parent->parent;    // Update new node (player)
            } else {    // Parent is red and uncle is black
                // If R - L condition is satisfied
                if (player_in == player_in->parent->left_child) {
                    player_in = player_in->parent;    // Update new node (player)
                    rotate_right(player_in);    // Perform right rotation
                }
                // Condition is simplified to R - R
                player_in->parent->color = 0;    // Parent (or previous grandparent) becomes black
                player_in->parent->parent->color = 1;    // Previous grandparent (new sibling) becomes red
                rotate_left(player_in->parent->parent);    // Perform left rotation
            }
        } else {    // If left child of grandparent is the parent, everything should be mirrored
            if (uncle->color == 1){
                uncle->color = 0;
                player_in->parent->color = 0;
                player_in->parent->parent->color = 1;
                player_in = player_in->parent->parent;
            } else {
                if (player_in == player_in->parent->right_child) {
                    player_in = player_in->parent;
                    rotate_left(player_in);
                }
                player_in->parent->color = 0;
                player_in->parent->parent->color = 1;
                rotate_right(player_in->parent->parent);
            }
        }
    }
    // In case the root is replaced, make sure it is black
    root->color = 0;
}

void PlayerDatabase::preorder_print(Player* player, string indentation){
    // Postorder = L R N
    if (player != nil){
        string color;
        if (player->color){
            color = "(RED) ";
        } else {
            color = "(BLACK) ";
        }
        cout << indentation << color << player->full_name << endl;
        indentation += "-";
        preorder_print(player->left_child, indentation);
        preorder_print(player->right_child, indentation);
    }
}

void PlayerDatabase::print_database() {
    preorder_print(root, "");
}

void PlayerDatabase::print_season_bests(const string& season) {
    cout << "End of the " << season << " Season" << endl;
    cout << "Max Points: " << max_point << " - Player Name: " << max_point_name << endl
         << "Max Assists: " << max_assist << " - Player Name: " << max_assist_name << endl
         << "Max Rebounds: " << max_rebound << " - Player Name: " << max_rebound_name << endl;
}

void PlayerDatabase::postorder_delete(Player* player_in) {
    // Postorder = L R N
    if (player_in != nil){
        postorder_delete(player_in->left_child);
        postorder_delete(player_in->right_child);
        delete (player_in);
    }
}

PlayerDatabase::~PlayerDatabase() {
    // Clean your own memory
    postorder_delete(root);
}
