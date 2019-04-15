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
            return true; //Successfully placed
        } else {
            return false; //Placement wasn't successful
        }    }
}

//Move the placed operator
bool OperatorGrid::move_operator(int x, int y, char direction, int move_by) {

}