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
    string season;
public:
    PlayerDatabase();
    string get_season();
    void set_season(string);
    void add_player(Player*);
    void fix_database(Player*)
};

#endif //HW3_PLAYERDATABASE_H
