//
// Created by batuhanfaik on 04/01/2021.
//
#include <string>

#ifndef HW3_PLAYER_H
#define HW3_PLAYER_H

using namespace std;

class Player {
    // Friend the database so private attr. can be accessed
    friend class PlayerDatabase;
    // Data to be stored
    string full_name;
    string team;
    int rebound;
    int assist;
    int point;
    // RB Tree properties
    int color;    // Black = 0, Red = 1
    Player *parent;
    Player *left_child;
    Player *right_child;

    // Methods
    Player();
    static void nodify_player(Player*, Player*);
public:
    Player(string, string, int, int, int);
    static Player* get_null_player();
    // No getter, setter required, db is friended
};


#endif //HW3_PLAYER_H
