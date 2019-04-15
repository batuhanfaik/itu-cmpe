//
// Created by;
// Batuhan Faik Derinbay 
// 150180705
// on 4/15/19.
//
#include "OperatorGrid.h"
#include <iostream>

using namespace std;

//Default constructor of the OperatorGrid
OperatorGrid::OperatorGrid(int rows, int cols) {
    grid_rows = rows;
    grid_cols = cols;

    //Create a grid 2d array
    grid = new char*[grid_rows];
    for (int i = 0; i < grid_rows; ++i) {
        grid[i] = new char[grid_cols];
    }

    //Initialize the elements of the grid
    for (int i = 0; i < grid_rows; ++i) {
        for (int j = 0; j < grid_cols; ++j) {
            grid[i][j] = '\0';
        }
    }
}

//Destructor of the OperatorGrid
OperatorGrid::~OperatorGrid() {
    //Delete the grid 2d array
    for (int i = 0; i < grid_rows; ++i) {
        delete[] grid[i];
    }
    delete[] grid;
    cout << "DESTRUCTOR: GIVE BACK[" << grid_rows << "," << grid_cols << "] chars." << endl;
    //Delete the operators array
    delete[] operators;
    cout << "DESTRUCTOR: GIVE BACK[" << num_operators << "] Operators." << endl;
}

//Place an operator type ArithmeticOperator in the 2d grid
bool OperatorGrid::place_operator(ArithmeticOperator *current_operator) {
    //IMPROVEMENT: Use switch-case
    if (current_operator->get_sign() == '+'){       //Case sign = '+'
        //Initialize errors and indexing variables
        int is_empty = 1;
        int checked_all = 0;
        int border_error = 0;

        //BORDER ERROR Checking in four directions
        if (current_operator->get_y() - current_operator->get_size() - 1 < 0){        //Direction: North
            border_error = 1;
        } else if (current_operator->get_size() + current_operator->get_x() > grid_cols){       //Direction: East
            border_error = 1;
        } else if (current_operator->get_size() + current_operator->get_y() > grid_rows) {       //Direction: South
            border_error = 1;
        } else if (current_operator->get_x() - current_operator->get_size() - 1 < 0) {       //Direction: West
            border_error = 1;
        }

        //CONFLICT ERROR Checking in four directions
        while (is_empty == 1 && checked_all == 0){
            for (int i = 0; i < current_operator->get_size(); ++i) {
                if (grid[current_operator->get_y()-i-1][current_operator->get_x()-1] != '\0'){      //Direction: North
                    is_empty = 0;
                } else if (grid[current_operator->get_y()-1][current_operator->get_x()+i-1] != '\0'){       //Direction: East
                    is_empty = 0;
                } else if (grid[current_operator->get_y()+i-1][current_operator->get_x()-1] != '\0'){       //Direction: South
                    is_empty = 0;
                } else if (grid[current_operator->get_y()-1][current_operator->get_x()-i-1] != '\0'){       //Direction: West
                    is_empty = 0;
                }
                if (is_empty == 0){      //Stop the loop if there is a conflict (IMPROVEMENT: Use a while loop)
                    break;
                }
            }
            if (is_empty == 1){ //If all the locations were empty until this point, then we can quit the loop
                checked_all = 1;
            }
        }
        if (border_error == 1){ //If there is a border error, print out the statement
            cout << "BORDER ERROR: Operator sign with size" << current_operator->get_size() << "can not be placed on ("
            << current_operator->get_x() << "," << current_operator->get_y() << ")." << endl;
        }
        // If there is a conflict error, print out the statement
        if (is_empty == 0){ //Meaning that at least one location that we want to place the operator on top is occupied
            cout << "CONFLICT ERROR: Operator sign with size" << current_operator->get_size() << "can not be placed on ("
            << current_operator->get_x() << "," << current_operator->get_y() << ")." << endl;
        }
        //If there are not any errors, place the operator
        if (border_error == 0 && is_empty == 1){
            for (int i = 0; i < current_operator->get_size(); ++i) {
                grid[current_operator->get_y() - i - 1][current_operator->get_x() - 1] = '+';     //Direction: North
                grid[current_operator->get_y() - 1][current_operator->get_x() + i - 1] = '+';     //Direction: East
                grid[current_operator->get_y() + i - 1][current_operator->get_x() - 1] = '+';     //Direction: South
                grid[current_operator->get_y() - 1][current_operator->get_x() - i - 1] = '+';     //Direction: West
            }
            operators[num_operators] = current_operator;    //Append the successful placement in to the operators array
            num_operators++;
            return true; //Successfully placed
        } else {
            return false; //Placement wasn't successful
        }
    } else if (current_operator->get_sign() == '-'){        //Case sign = '-'
        //Initialize errors and indexing variables
        int is_empty = 1;
        int checked_all = 0;
        int border_error = 0;

        //BORDER ERROR Checking in two directions
        if (current_operator->get_size() + current_operator->get_x() > grid_cols){       //Direction: East
            border_error = 1;
        } else if (current_operator->get_x() - current_operator->get_size() - 1 < 0) {       //Direction: West
            border_error = 1;
        }

        //CONFLICT ERROR Checking in two directions
        while (is_empty == 1 && checked_all == 0){
            for (int i = 0; i < current_operator->get_size(); ++i) {
                if (grid[current_operator->get_y()-1][current_operator->get_x()+i-1] != '\0'){       //Direction: East
                    is_empty = 0;
                } else if (grid[current_operator->get_y()-1][current_operator->get_x()-i-1] != '\0'){       //Direction: West
                    is_empty = 0;
                }
                if (is_empty == 0){      //Stop the loop if there is a conflict (IMPROVEMENT: Use a while loop)
                    break;
                }
            }
            if (is_empty == 1){ //If all the locations were empty until this point, then we can quit the loop
                checked_all = 1;
            }
        }
        if (border_error == 1){ //If there is a border error, print out the statement
            cout << "BORDER ERROR: Operator sign with size" << current_operator->get_size() << "can not be placed on ("
            << current_operator->get_x() << "," << current_operator->get_y() << ")." << endl;
        }
        // If there is a conflict error, print out the statement
        if (is_empty == 0){ //Meaning that at least one location that we want to place the operator on top is occupied
            cout << "CONFLICT ERROR: Operator sign with size" << current_operator->get_size() << "can not be placed on ("
            << current_operator->get_x() << "," << current_operator->get_y() << ")." << endl;
        }
        //If there are not any errors, place the operator
        if (border_error == 0 && is_empty == 1){
            for (int i = 0; i < current_operator->get_size(); ++i) {
                grid[current_operator->get_y() - 1][current_operator->get_x() + i - 1] = '-';     //Direction: East
                grid[current_operator->get_y() - 1][current_operator->get_x() - i - 1] = '-';     //Direction: West
            }
            operators[num_operators] = current_operator;    //Append the successful placement in to the operators array
            num_operators++;
            return true; //Successfully placed
        } else {
            return false; //Placement wasn't successful
        }
    } else if (current_operator->get_sign() == 'x') {        //Case sign = 'x'
        //Initialize errors and indexing variables
        int is_empty = 1;
        int checked_all = 0;
        int border_error = 0;

        //BORDER ERROR Checking in four directions (IMPROVEMENT: Can be done by checking two opposing diagonals)
        if ((current_operator->get_y() - current_operator->get_size() - 1 < 0) &&
        (current_operator->get_x() + current_operator->get_size() > grid_cols)){        //Direction: Northeast
            border_error = 1;
        } else if ((current_operator->get_size() + current_operator->get_y() > grid_rows) &&
        (current_operator->get_x() + current_operator->get_size() > grid_cols)) {       //Direction: Southeast
            border_error = 1;
        } else if ((current_operator->get_size() + current_operator->get_y() > grid_rows) &&
        (current_operator->get_x() - current_operator->get_size() - 1 < 0)) {       //Direction: Southwest
            border_error = 1;
        } else if ((current_operator->get_y() - current_operator->get_size() - 1 < 0) &&
        (current_operator->get_x() - current_operator->get_size() - 1 < 0)) {       //Direction: Northwest
            border_error = 1;
        }

        //CONFLICT ERROR Checking in four directions
        while (is_empty == 1 && checked_all == 0){
            for (int i = 0; i < current_operator->get_size(); ++i) {
                if (grid[current_operator->get_y()-i-1][current_operator->get_x()+i-1] != '\0'){      //Direction: Northeast
                    is_empty = 0;
                } else if (grid[current_operator->get_y()+i-1][current_operator->get_x()+i-1] != '\0'){       //Direction: Southeast
                    is_empty = 0;
                } else if (grid[current_operator->get_y()+i-1][current_operator->get_x()-i-1] != '\0'){       //Direction: Southwest
                    is_empty = 0;
                } else if (grid[current_operator->get_y()-i-1][current_operator->get_x()-i-1] != '\0'){       //Direction: Northwest
                    is_empty = 0;
                }
                if (is_empty == 0){      //Stop the loop if there is a conflict (IMPROVEMENT: Use a while loop)
                    break;
                }
            }
            if (is_empty == 1){ //If all the locations were empty until this point, then we can quit the loop
                checked_all = 1;
            }
        }
        if (border_error == 1){ //If there is a border error, print out the statement
            cout << "BORDER ERROR: Operator sign with size" << current_operator->get_size() << "can not be placed on ("
            << current_operator->get_x() << "," << current_operator->get_y() << ")." << endl;
        }
        // If there is a conflict error, print out the statement
        if (is_empty == 0){ //Meaning that at least one location that we want to place the operator on top is occupied
            cout << "CONFLICT ERROR: Operator sign with size" << current_operator->get_size() << "can not be placed on ("
            << current_operator->get_x() << "," << current_operator->get_y() << ")." << endl;
        }
        //If there are not any errors, place the operator
        if (border_error == 0 && is_empty == 1){
            for (int i = 0; i < current_operator->get_size(); ++i) {
                grid[current_operator->get_y() - i - 1][current_operator->get_x() + i - 1] = 'x';     //Direction: Northeast
                grid[current_operator->get_y() + i - 1][current_operator->get_x() + i - 1] = 'x';     //Direction: Southeast
                grid[current_operator->get_y() + i - 1][current_operator->get_x() - i - 1] = 'x';     //Direction: Southwest
                grid[current_operator->get_y() - i - 1][current_operator->get_x() - i - 1] = 'x';     //Direction: Northwest
            }
            operators[num_operators] = current_operator;    //Append the successful placement in to the operators array
            num_operators++;
            return true; //Successfully placed
        } else {
            return false; //Placement wasn't successful
        }
    } else if (current_operator->get_sign() == '/'){        //Case sign = '/'
        //Initialize errors and indexing variables
        int is_empty = 1;
        int checked_all = 0;
        int border_error = 0;

        //BORDER ERROR Checking in two directions
        if ((current_operator->get_y() - current_operator->get_size() - 1 < 0) &&
        (current_operator->get_x() + current_operator->get_size() > grid_cols)){        //Direction: Northeast
            border_error = 1;
        } else if ((current_operator->get_size() + current_operator->get_y() > grid_rows) &&
        (current_operator->get_x() - current_operator->get_size() - 1 < 0)) {       //Direction: Southwest
            border_error = 1;
        }
        //CONFLICT ERROR Checking in four directions
        while (is_empty == 1 && checked_all == 0){
            for (int i = 0; i < current_operator->get_size(); ++i) {
                if (grid[current_operator->get_y()-i-1][current_operator->get_x()+i-1] != '\0'){      //Direction: Northeast
                    is_empty = 0;
                } else if (grid[current_operator->get_y()+i-1][current_operator->get_x()-i-1] != '\0'){       //Direction: Southwest
                    is_empty = 0;
                }
                if (is_empty == 0){      //Stop the loop if there is a conflict (IMPROVEMENT: Use a while loop)
                    break;
                }
            }
            if (is_empty == 1){ //If all the locations were empty until this point, then we can quit the loop
                checked_all = 1;
            }
        }
        if (border_error == 1){ //If there is a border error, print out the statement
            cout << "BORDER ERROR: Operator sign with size" << current_operator->get_size() << "can not be placed on ("
            << current_operator->get_x() << "," << current_operator->get_y() << ")." << endl;
        }
        // If there is a conflict error, print out the statement
        if (is_empty == 0){ //Meaning that at least one location that we want to place the operator on top is occupied
            cout << "CONFLICT ERROR: Operator sign with size" << current_operator->get_size() << "can not be placed on ("
            << current_operator->get_x() << "," << current_operator->get_y() << ")." << endl;
        }
        //If there are not any errors, place the operator
        if (border_error == 0 && is_empty == 1){
            for (int i = 0; i < current_operator->get_size(); ++i) {
                grid[current_operator->get_y() - i - 1][current_operator->get_x() + i - 1] = '/';     //Direction: Northeast
                grid[current_operator->get_y() + i - 1][current_operator->get_x() - i - 1] = '/';     //Direction: Southwest
            }
            operators[num_operators] = current_operator;    //Append the successful placement in to the operators array
            num_operators++;
            return true; //Successfully placed
        } else {
            return false; //Placement wasn't successful
        }    }
}

//Move the placed operator
bool OperatorGrid::move_operator(int x, int y, char direction, int move_by) {
    int mid_x = x;
    int mid_y = y;
    char current_operator_sign = grid[y-1][x-1];
    if (current_operator_sign == '+'){
        //Initialize variables and indexes
        int mid_found = 0;
        int current_operator_size = 0;

        while (mid_found != 1){        //Search until the midpoint is found
            if (grid[mid_y - 2][mid_x - 1] == '+' && grid[mid_y - 1][mid_x] == '+' &&       //If mid is surrounded by four other equal signs
                    grid[mid_y][mid_x - 1] == '+' && grid[mid_y - 1][mid_x - 2] == '+'){        //We found the midpoint
                mid_found = 1;
            } else if (grid[mid_y - 2][mid_x - 1] == '+'){      //If the mid is to the north
                mid_y--;
            } else if (grid[mid_y - 1][mid_x] == '+') {      //If the mid is to the east
                mid_x++;
            } else if (grid[mid_y][mid_x - 1] == '+') {      //If the mid is to the south
                mid_y++;
            } else if (grid[mid_y - 1][mid_x - 2] == '+') {      //If the mid is to the west
                mid_x--;
            }
        }

        //Find the size iteration through all directions
        int dir_north = 0;
        int dir_east = 0;
        int dir_south = 0;
        int dir_west = 0;

        while (grid[mid_y - dir_north - 1][mid_x - 1] == '+'){  //Find the amount of matching sign to the north
            dir_north++;
        }
        while (grid[mid_y - 1][mid_x + dir_east - 1] == '+'){  //Find the amount of matching sign to the east
            dir_east++;
        }
        while (grid[mid_y + dir_south - 1][mid_x - 1] == '+'){  //Find the amount of matching sign to the south
            dir_south++;
        }
        while (grid[mid_y - 1][mid_x - dir_west - 1] == '+') {  //Find the amount of matching sign to the west
            dir_west++;
        }

        //Minimum amount of the directions is the likeliest value to be the size
        int tmp_min1, tmp_min2;
        if (dir_north > dir_east){
            tmp_min1 = dir_east;
        } else {
            tmp_min1 = dir_north;
        }
        if (dir_south > dir_west){
            tmp_min2 = dir_west;
        } else {
            tmp_min2 = dir_south;
        }
        if (tmp_min1 > tmp_min2){ //Assign the smallest integer
            current_operator_size = tmp_min2;
        } else {
            current_operator_size = tmp_min1;
        }

        //Clear the cells containing the current operator
        for (int i = -current_operator_size; i < current_operator_size + 1; ++i) {
            grid[mid_y + i - 1][mid_x - 1] = '\0'; //From north to south
            grid[mid_y - 1][mid_x + i - 1] = '\0'; //From west to east
        }

        //Check if movable towards the input direction
        //If so move, if not replace the already cleared operator
        if (direction == 'U'){      //Move up
            int new_mid_x = mid_x;
            int new_mid_y = mid_y - move_by;
            ArithmeticOperator tmp_operator = ArithmeticOperator(new_mid_x,new_mid_y,current_operator_size,current_operator_sign);
            if (place_operator(&tmp_operator)){
                place_operator(&tmp_operator);
                cout << "SUCCESS: sign moved from (" << mid_x << "," << mid_y << ") to ("
                << new_mid_x << "," << new_mid_y << ")." << endl;
                return true;
            } else {
                tmp_operator = ArithmeticOperator(mid_x,mid_y,current_operator_size,current_operator_sign);
                place_operator(&tmp_operator);
                return false;
            }
        } else if (direction == 'R'){       //Move right
            int new_mid_x = mid_x + move_by;
            int new_mid_y = mid_y;
            ArithmeticOperator tmp_operator = ArithmeticOperator(new_mid_x,new_mid_y,current_operator_size,current_operator_sign);
            if (place_operator(&tmp_operator)){
                place_operator(&tmp_operator);
                cout << "SUCCESS: sign moved from (" << mid_x << "," << mid_y << ") to ("
                << new_mid_x << "," << new_mid_y << ")." << endl;
                return true;
            } else {
                tmp_operator = ArithmeticOperator(mid_x,mid_y,current_operator_size,current_operator_sign);
                place_operator(&tmp_operator);
                return false;
            }
        } else if (direction == 'D'){       //Move down
            int new_mid_x = mid_x;
            int new_mid_y = mid_y + move_by;
            ArithmeticOperator tmp_operator = ArithmeticOperator(new_mid_x,new_mid_y,current_operator_size,current_operator_sign);
            if (place_operator(&tmp_operator)){
                place_operator(&tmp_operator);
                cout << "SUCCESS: sign moved from (" << mid_x << "," << mid_y << ") to ("
                << new_mid_x << "," << new_mid_y << ")." << endl;
                return true;
            } else {
                tmp_operator = ArithmeticOperator(mid_x,mid_y,current_operator_size,current_operator_sign);
                place_operator(&tmp_operator);
                return false;
            }
        } else if (direction == 'L'){       //Move left
            int new_mid_x = mid_x - move_by;
            int new_mid_y = mid_y;
            ArithmeticOperator tmp_operator = ArithmeticOperator(new_mid_x,new_mid_y,current_operator_size,current_operator_sign);
            if (place_operator(&tmp_operator)){
                place_operator(&tmp_operator);
                cout << "SUCCESS: sign moved from (" << mid_x << "," << mid_y << ") to ("
                << new_mid_x << "," << new_mid_y << ")." << endl;
                return true;
            } else {
                tmp_operator = ArithmeticOperator(mid_x,mid_y,current_operator_size,current_operator_sign);
                place_operator(&tmp_operator);
                return false;
            }
        }
    } else if (current_operator_sign == '-') {
        //Initialize variables and indexes
        int mid_found = 0;
        int current_operator_size = 0;

        while (mid_found != 1) {        //Search until the midpoint is found
            if (grid[mid_y - 1][mid_x] == '-' && grid[mid_y - 1][mid_x - 2] == '-') {      //If mid is surrounded by two other equal signs we found the midpoint
                mid_found = 1;
            } else if (grid[mid_y - 1][mid_x] == '-') {      //If the mid is to the east
                mid_x++;
            } else if (grid[mid_y - 1][mid_x - 2] == '-') {      //If the mid is to the west
                mid_x--;
            }
        }

        //Find the size iteration through all directions
        int dir_east = 0;
        int dir_west = 0;

        while (grid[mid_y - 1][mid_x + dir_east - 1] == '-') {  //Find the amount of matching sign to the east
            dir_east++;
        }
        while (grid[mid_y - 1][mid_x - dir_west - 1] == '-') {  //Find the amount of matching sign to the west
            dir_west++;
        }

        //Minimum amount of the directions is the likeliest value to be the size
        if (dir_east > dir_west) { //Assign the smallest integer
            current_operator_size = dir_west;
        } else {
            current_operator_size = dir_east;
        }

        //Clear the cells containing the current operator
        for (int i = -current_operator_size; i < current_operator_size + 1; ++i) {
            grid[mid_y - 1][mid_x + i - 1] = '\0'; //From west to east
        }

        //Check if movable towards the input direction
        //If so move, if not replace the already cleared operator
        if (direction == 'U') {      //Move up
            int new_mid_x = mid_x;
            int new_mid_y = mid_y - move_by;
            ArithmeticOperator tmp_operator = ArithmeticOperator(new_mid_x, new_mid_y, current_operator_size,
                                                                 current_operator_sign);
            if (place_operator(&tmp_operator)) {
                place_operator(&tmp_operator);
                cout << "SUCCESS: sign moved from (" << mid_x << "," << mid_y << ") to ("
                     << new_mid_x << "," << new_mid_y << ")." << endl;
                return true;
            } else {
                tmp_operator = ArithmeticOperator(mid_x, mid_y, current_operator_size, current_operator_sign);
                place_operator(&tmp_operator);
                return false;
            }
        } else if (direction == 'R') {       //Move right
            int new_mid_x = mid_x + move_by;
            int new_mid_y = mid_y;
            ArithmeticOperator tmp_operator = ArithmeticOperator(new_mid_x, new_mid_y, current_operator_size,
                                                                 current_operator_sign);
            if (place_operator(&tmp_operator)) {
                place_operator(&tmp_operator);
                cout << "SUCCESS: sign moved from (" << mid_x << "," << mid_y << ") to ("
                     << new_mid_x << "," << new_mid_y << ")." << endl;
                return true;
            } else {
                tmp_operator = ArithmeticOperator(mid_x, mid_y, current_operator_size, current_operator_sign);
                place_operator(&tmp_operator);
                return false;
            }
        } else if (direction == 'D') {       //Move down
            int new_mid_x = mid_x;
            int new_mid_y = mid_y + move_by;
            ArithmeticOperator tmp_operator = ArithmeticOperator(new_mid_x, new_mid_y, current_operator_size,
                                                                 current_operator_sign);
            if (place_operator(&tmp_operator)) {
                place_operator(&tmp_operator);
                cout << "SUCCESS: sign moved from (" << mid_x << "," << mid_y << ") to ("
                     << new_mid_x << "," << new_mid_y << ")." << endl;
                return true;
            } else {
                tmp_operator = ArithmeticOperator(mid_x, mid_y, current_operator_size, current_operator_sign);
                place_operator(&tmp_operator);
                return false;
            }
        } else if (direction == 'L') {       //Move left
            int new_mid_x = mid_x - move_by;
            int new_mid_y = mid_y;
            ArithmeticOperator tmp_operator = ArithmeticOperator(new_mid_x, new_mid_y, current_operator_size,
                                                                 current_operator_sign);
            if (place_operator(&tmp_operator)) {
                place_operator(&tmp_operator);
                cout << "SUCCESS: sign moved from (" << mid_x << "," << mid_y << ") to ("
                     << new_mid_x << "," << new_mid_y << ")." << endl;
                return true;
            } else {
                tmp_operator = ArithmeticOperator(mid_x, mid_y, current_operator_size, current_operator_sign);
                place_operator(&tmp_operator);
                return false;
            }
        }
    } else if (current_operator_sign == 'x'){
        //Initialize variables and indexes
        int mid_found = 0;
        int current_operator_size = 0;

        while (mid_found != 1){        //Search until the midpoint is found
            if (grid[mid_y - 2][mid_x] == 'x' && grid[mid_y][mid_x] == 'x' &&       //If mid is surrounded by four other equal signs
                    grid[mid_y][mid_x - 2] == 'x' && grid[mid_y - 2][mid_x - 2] == 'x'){        //We found the midpoint
                mid_found = 1;
            } else if (grid[mid_y - 2][mid_x] == 'x'){      //If the mid is to the northeast
                mid_y--;
                mid_x++;
            } else if (grid[mid_y][mid_x] == 'x') {      //If the mid is to the southeast
                mid_y++;
                mid_x++;
            } else if (grid[mid_y][mid_x - 2] == 'x') {      //If the mid is to the southwest
                mid_y++;
                mid_x--;
            } else if (grid[mid_y - 2][mid_x - 2] == 'x') {      //If the mid is to the northwest
                mid_y--;
                mid_x--;
            }
        }

        //Find the size iteration through all directions
        int dir_northeast = 0;
        int dir_southeast = 0;
        int dir_southwest = 0;
        int dir_northwest = 0;

        while (grid[mid_y - dir_northeast - 1][mid_x + dir_northeast - 1] == 'x'){  //Find the amount of matching sign to the northeast
            dir_northeast++;
        }
        while (grid[mid_y + dir_southeast - 1][mid_x + dir_southeast - 1] == 'x'){  //Find the amount of matching sign to the southeast
            dir_southeast++;
        }
        while (grid[mid_y + dir_southwest - 1][mid_x - dir_southwest - 1] == 'x'){  //Find the amount of matching sign to the southwest
            dir_southwest++;
        }
        while (grid[mid_y - dir_northwest - 1][mid_x - dir_northwest - 1] == 'x') {  //Find the amount of matching sign to the northwest
            dir_northwest++;
        }

        //Minimum amount of the directions is the likeliest value to be the size
        int tmp_min1, tmp_min2;
        if (dir_northeast > dir_southeast){
            tmp_min1 = dir_southeast;
        } else {
            tmp_min1 = dir_northeast;
        }
        if (dir_southwest > dir_northwest){
            tmp_min2 = dir_northwest;
        } else {
            tmp_min2 = dir_southwest;
        }
        if (tmp_min1 > tmp_min2){ //Assign the smallest integer
            current_operator_size = tmp_min2;
        } else {
            current_operator_size = tmp_min1;
        }

        //Clear the cells containing the current operator
        for (int i = -current_operator_size; i < current_operator_size + 1; ++i) {
            grid[mid_y + i - 1][mid_x + i - 1] = '\0'; //From northwest to southeast
            grid[mid_y + i - 1][mid_x - i - 1] = '\0'; //From northeast to southwest
        }

        //Check if movable towards the input direction
        //If so move, if not replace the already cleared operator
        if (direction == 'U'){      //Move up
            int new_mid_x = mid_x;
            int new_mid_y = mid_y - move_by;
            ArithmeticOperator tmp_operator = ArithmeticOperator(new_mid_x,new_mid_y,current_operator_size,current_operator_sign);
            if (place_operator(&tmp_operator)){
                place_operator(&tmp_operator);
                cout << "SUCCESS: sign moved from (" << mid_x << "," << mid_y << ") to ("
                << new_mid_x << "," << new_mid_y << ")." << endl;
                return true;
            } else {
                tmp_operator = ArithmeticOperator(mid_x,mid_y,current_operator_size,current_operator_sign);
                place_operator(&tmp_operator);
                return false;
            }
        } else if (direction == 'R'){       //Move right
            int new_mid_x = mid_x + move_by;
            int new_mid_y = mid_y;
            ArithmeticOperator tmp_operator = ArithmeticOperator(new_mid_x,new_mid_y,current_operator_size,current_operator_sign);
            if (place_operator(&tmp_operator)){
                place_operator(&tmp_operator);
                cout << "SUCCESS: sign moved from (" << mid_x << "," << mid_y << ") to ("
                << new_mid_x << "," << new_mid_y << ")." << endl;
                return true;
            } else {
                tmp_operator = ArithmeticOperator(mid_x,mid_y,current_operator_size,current_operator_sign);
                place_operator(&tmp_operator);
                return false;
            }
        } else if (direction == 'D'){       //Move down
            int new_mid_x = mid_x;
            int new_mid_y = mid_y + move_by;
            ArithmeticOperator tmp_operator = ArithmeticOperator(new_mid_x,new_mid_y,current_operator_size,current_operator_sign);
            if (place_operator(&tmp_operator)){
                place_operator(&tmp_operator);
                cout << "SUCCESS: sign moved from (" << mid_x << "," << mid_y << ") to ("
                << new_mid_x << "," << new_mid_y << ")." << endl;
                return true;
            } else {
                tmp_operator = ArithmeticOperator(mid_x,mid_y,current_operator_size,current_operator_sign);
                place_operator(&tmp_operator);
                return false;
            }
        } else if (direction == 'L'){       //Move left
            int new_mid_x = mid_x - move_by;
            int new_mid_y = mid_y;
            ArithmeticOperator tmp_operator = ArithmeticOperator(new_mid_x,new_mid_y,current_operator_size,current_operator_sign);
            if (place_operator(&tmp_operator)){
                place_operator(&tmp_operator);
                cout << "SUCCESS: sign moved from (" << mid_x << "," << mid_y << ") to ("
                << new_mid_x << "," << new_mid_y << ")." << endl;
                return true;
            } else {
                tmp_operator = ArithmeticOperator(mid_x,mid_y,current_operator_size,current_operator_sign);
                place_operator(&tmp_operator);
                return false;
            }
        }
    } else if (current_operator_sign == '/'){
        //Initialize variables and indexes
        int mid_found = 0;
        int current_operator_size = 0;

        while (mid_found != 1){        //Search until the midpoint is found
            if (grid[mid_y - 2][mid_x] == '/' && grid[mid_y][mid_x - 2] == '/'){        //If mid is surrounded by two other equal signs we found the midpoint
                mid_found = 1;
            } else if (grid[mid_y - 2][mid_x ] == '/'){      //If the mid is to the northeast
                mid_y--;
                mid_x++;
            } else if (grid[mid_y][mid_x - 2] == '/') {      //If the mid is to the southwest
                mid_y++;
                mid_x--;
            }
        }

        //Find the size iteration through all directions
        int dir_northeast = 0;
        int dir_southwest = 0;

        while (grid[mid_y - dir_northeast - 1][mid_x + dir_northeast - 1] == '/'){  //Find the amount of matching sign to the northeast
            dir_northeast++;
        }
        while (grid[mid_y + dir_southwest - 1][mid_x - dir_southwest - 1] == '/'){  //Find the amount of matching sign to the southwest
            dir_southwest++;
        }

        //Minimum amount of the directions is the likeliest value to be the size
        if (dir_northeast > dir_southwest){ //Assign the smallest integer
            current_operator_size = dir_southwest;
        } else {
            current_operator_size = dir_northeast;
        }

        //Clear the cells containing the current operator
        for (int i = -current_operator_size; i < current_operator_size + 1; ++i) {
            grid[mid_y + i - 1][mid_x - 1] = '\0'; //From north to south
            grid[mid_y - 1][mid_x + i - 1] = '\0'; //From west to east
        }

        //Check if movable towards the input direction
        //If so move, if not replace the already cleared operator
        if (direction == 'U'){      //Move up
            int new_mid_x = mid_x;
            int new_mid_y = mid_y - move_by;
            ArithmeticOperator tmp_operator = ArithmeticOperator(new_mid_x,new_mid_y,current_operator_size,current_operator_sign);
            if (place_operator(&tmp_operator)){
                place_operator(&tmp_operator);
                cout << "SUCCESS: sign moved from (" << mid_x << "," << mid_y << ") to ("
                << new_mid_x << "," << new_mid_y << ")." << endl;
                return true;
            } else {
                tmp_operator = ArithmeticOperator(mid_x,mid_y,current_operator_size,current_operator_sign);
                place_operator(&tmp_operator);
                return false;
            }
        } else if (direction == 'R'){       //Move right
            int new_mid_x = mid_x + move_by;
            int new_mid_y = mid_y;
            ArithmeticOperator tmp_operator = ArithmeticOperator(new_mid_x,new_mid_y,current_operator_size,current_operator_sign);
            if (place_operator(&tmp_operator)){
                place_operator(&tmp_operator);
                cout << "SUCCESS: sign moved from (" << mid_x << "," << mid_y << ") to ("
                << new_mid_x << "," << new_mid_y << ")." << endl;
                return true;
            } else {
                tmp_operator = ArithmeticOperator(mid_x,mid_y,current_operator_size,current_operator_sign);
                place_operator(&tmp_operator);
                return false;
            }
        } else if (direction == 'D'){       //Move down
            int new_mid_x = mid_x;
            int new_mid_y = mid_y + move_by;
            ArithmeticOperator tmp_operator = ArithmeticOperator(new_mid_x,new_mid_y,current_operator_size,current_operator_sign);
            if (place_operator(&tmp_operator)){
                place_operator(&tmp_operator);
                cout << "SUCCESS: sign moved from (" << mid_x << "," << mid_y << ") to ("
                << new_mid_x << "," << new_mid_y << ")." << endl;
                return true;
            } else {
                tmp_operator = ArithmeticOperator(mid_x,mid_y,current_operator_size,current_operator_sign);
                place_operator(&tmp_operator);
                return false;
            }
        } else if (direction == 'L'){       //Move left
            int new_mid_x = mid_x - move_by;
            int new_mid_y = mid_y;
            ArithmeticOperator tmp_operator = ArithmeticOperator(new_mid_x,new_mid_y,current_operator_size,current_operator_sign);
            if (place_operator(&tmp_operator)){
                place_operator(&tmp_operator);
                cout << "SUCCESS: sign moved from (" << mid_x << "," << mid_y << ") to ("
                << new_mid_x << "," << new_mid_y << ")." << endl;
                return true;
            } else {
                tmp_operator = ArithmeticOperator(mid_x,mid_y,current_operator_size,current_operator_sign);
                place_operator(&tmp_operator);
                return false;
            }
        }
    }
}

//Print method of the operators
void OperatorGrid::print_operators() {
    for (int i = 0; i < num_operators; ++i) {
        operators[i]->print_operator();
    }
}
