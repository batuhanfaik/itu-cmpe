/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * /
* @ Filename: main
* @ Date: 04-Jan-2021
* @ AUTHOR: batuhanfaik
* @ Copyright (C) 2020 Batuhan Faik Derinbay
* @ Project: hw3
* @ Description: Read all players and create tree (driver code)
* @ Compiling: g++ -o 150180705.out main.cpp
* @ Running: ./150180705.out <csv_file> (E.g., ./150180705.out sample.csv)
* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
#include <iostream>
#include <fstream>
#include <string>

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
    void print_player();
    // No getter, setter required, db is friended
};

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
    void postorder_delete(Player*);
public:
    PlayerDatabase();
    string get_season();
    void set_season(string);
    void add_player(Player*);
    void update_player(Player*);
    bool player_exists(Player*);
    void print_season_bests(const string&);
    void print_database();
    ~PlayerDatabase();
};

int main(int argc, char** argv) {
    string filename = "sample.csv";
    if (argc < 2){
        cout << "An CSV filename needs to be passed in.\nThis run will use \"sample.csv\"" << endl;
    } else if (argc > 2){
        cout << "More than one parameters are passed in.\nThis run will use \"sample.csv\"" << endl;
    } else {
        filename = argv[1];
    }

    ifstream file;
    file.open(filename);

    if (!file) {
        cerr << "File cannot be opened!";
        exit(1);
    }

    auto* player_db = new PlayerDatabase();
    string line, season, full_name, team;
    int rebound, assist, point;

    getline(file, line); // Read the header line

    // Read until the end of the file
    while (!file.eof()){
        // Read values of the Players
        getline(file, season, ',');     // Read season
        // Season empty checks ensures the program will work correctly
        // even if the CSV file has a newline at the end
        if (season.empty()) {
            season = player_db->get_season();    // CSV format is not correct
        } else {
            if (player_db->get_season() != season)    // Change the season
                player_db->set_season(season);
        }
        getline(file, full_name, ',');    // Read full name
        getline(file, team, ',');    // Read team
        getline(file, line, ',');    // Read rebound
        rebound = stoi(line);
        getline(file, line, ',');    // Read assist
        assist = stoi(line);
        getline(file, line, '\n');    // Read point
        point = stoi(line);
        // Create player
        auto* player = new Player(full_name, team, rebound, assist, point);
        if (!player_db->player_exists(player)){    // If the player doesn't exist
            // Add player to db (tree)
            player_db->add_player(player);
        } else {    // If player already exists in the db
            // Update the existing player
            player_db->update_player(player);
        }
    }

    // Make sure the last season is printed
    if (player_db->get_season() == season) {    // Change the season
        player_db->print_season_bests(season);
    }

    // Delete database
    delete(player_db);
    // Close the file stream
    file.close();

    return 0;
}

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
