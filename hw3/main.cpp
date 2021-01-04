/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * /
* @ Filename: main
* @ Date: 04-Jan-2021
* @ AUTHOR: batuhanfaik
* @ Copyright (C) 2020 Batuhan Faik Derinbay
* @ Project: hw3
* @ Description: Read all players and create tree (driver code)
* @ Compiling: g++ -o 150180705.out main.cpp ----
* @ Running: ./150180705.out <csv_file> (E.g., ./150180705.out sample.csv)
* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
#include <iostream>
#include <fstream>
#include <string>

#include "player.h"
#include "playerdatabase.h"

using namespace std;

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
        if (player_db->get_season() != season) {    // Change the season
            if (player_db->get_season().empty())    // If the first season is not set, set it
                player_db->set_first_season(season);
            player_db->set_season(season);
        }
        getline(file, full_name, ',');    // Read full name
        getline(file, team, ',');    // Read team
        getline(file, line, ',');    // Read rebound
        rebound = stoi(line);
        getline(file, line, ',');    // Read assist
        assist = stoi(line);
        getline(file, line, ',');    // Read point
        point = stoi(line);
        getline(file, line, '\n');    // Read the last new line character
        // Create player
        auto* player = new Player(full_name, team, rebound, assist, point);
        // Add it to db (tree)
        player_db->add_player(player);
    }

    delete(player_db);
    file.close();

    return 0;
}
