//
// Created by batuhanfaik on 04/01/2021.
//
#include <string>
#include "player.h"

#ifndef HW3_PLAYERDATABASE_H
#define HW3_PLAYERDATABASE_H

using namespace std;

class PlayerDatabase {
    Player* root;
    Player* nil;
    string first_season;
    string current_season;
    int max_rebound;
    string max_rebound_name;
    int max_assist;
    string max_assist_name;
    int max_point;
    string max_point_name;

    // Methods
    void update_season_bests(Player*);
    Player* search_player(Player*, const string&);
    void rotate_left(Player*);
    void rotate_right(Player*);
    void fix_database(Player*);
    void preorder_print(Player*, string);
    void print_season_bests(const string&);
    void postorder_delete(Player*);
public:
    PlayerDatabase();
    string get_season();
    void set_season(string);
    void add_player(Player*);
    void update_player(Player*);
    bool player_exists(Player*);
    void print_database();
    ~PlayerDatabase();
};

#endif //HW3_PLAYERDATABASE_H
