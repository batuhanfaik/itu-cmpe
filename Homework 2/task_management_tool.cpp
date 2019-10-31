/* @Author
Student Name: Batuhan Faik Derinbay
Student ID : 150180705
Date: 31.10.19 */

/*
PLEASE, DO NOT CHANGE void display(bool verbose, bool testing), int getUsableDay() and int getUsableTime() FUNCTIONS.
YOU HAVE TO WRITE THE REQUIRED  FUNCTIONS THAT IS MENTIONED ABOVE. YOU CAN ADD NEW FUNCTIONS IF YOU NEED.
*/

#include <iostream>
#include <cstdlib>
#include <cstring>
#include <iomanip>

#include "task_management_tool.h"

using namespace std;


void WorkPlan::display(bool verbose, bool testing) {
    string inone = "***";
    if (head != NULL) {
        Task *pivot = new Task;
        Task *compeer = new Task;

        pivot = head;
        do {
            if (testing)
                inone += " ";
            else
                cout << pivot->day << ". DAY" << endl;
            compeer = pivot;
            while (compeer != NULL) {
                string PREV = compeer->previous != NULL ? compeer->previous->name : "NULL";
                string NEXT = compeer->next != NULL ? compeer->next->name : "NULL";
                string CONT = compeer->counterpart != NULL ? compeer->counterpart->name : "NULL";
                if (testing)
                    inone += compeer->name;
                else if (verbose)
                    cout << "\t" << setw(2) << compeer->time << ":00\t" << PREV << "\t<- " << compeer->name << "("
                         << compeer->priority << ")->\t" << NEXT << "\t |_" << CONT << endl;
                else
                    cout << "\t" << setw(2) << compeer->time << ":00\t" << compeer->name << "(" << compeer->priority
                         << ")" << endl;
                compeer = compeer->counterpart;
            }
            pivot = pivot->next;
        } while (pivot != head);
        if (testing) {
            cout << inone << endl;
            cout << "(checking cycled list:";
            if (checkCycledList())
                cout << " PASS)" << endl;
            else
                cout << " FAIL)" << endl;
        }
    } else
        cout << "There is no task yet!" << endl;
}

int WorkPlan::getUsableDay() {
    return usable_day;
}

int WorkPlan::getUsableTime() {
    return usable_time;
}


void WorkPlan::create() {
    Task *head = nullptr;
}

void WorkPlan::close() {
    //THIS FUNCTION WILL BE CODED BY YOU
}

void WorkPlan::add(Task *task) {
     /* Oh assistant, my dear assistant
     Why are you passing the same Task object rather than deleting the old one and creating a new one
     Now I have to write a psuedo-constructor like structure for allocating new memory to such tasks
     And implement their destroyers :'( */
    Task *new_task = new Task;
    new_task->name = new char[strlen(task->name)];
    strcpy(new_task->name, task->name);
    new_task->day = task->day;
    new_task->time = task->time;
    new_task->priority = task->priority;
    task = new_task;

    // TODO: Check for priorities before insertion

    // Check if there exists a head
    if (!head) {
        head = task;
        head->next = head;
        head->previous = head;
    } else {    // If given task is not the first one find the suitable location
        Task *current_task = head;
        bool lap = false;
        while (current_task->day < task->day && !lap) {     // Check if the day exists
            current_task = current_task->next;
            if (current_task == head) {
                lap = true;
            }
        }
        // If it's the first task of a day insert the day in CMLL (circular multi linked list)
        if (current_task->day != task->day) {
            current_task = current_task->previous;
            Task *tmp_next = current_task->next;
            current_task->next = task;
            tmp_next->previous = task;
            task->next = tmp_next;
            task->previous = current_task;
            if (task->day < head->day){     // It's the first day, we found a new head
                head = task;
                head->next = task->next;
                head->previous = task->previous;
            }
        } else {    // There exists a day with previous tasks
            // Look for the time opening
            Task *prev_ctpart = nullptr;
            Task *tmp = current_task;
            while (tmp){
                if (current_task->time < task->time){
                    prev_ctpart = current_task;
                    if (current_task->counterpart){
                        current_task = current_task->counterpart;
                    }
                }
                tmp = tmp->counterpart;
            }

            if (current_task->time < 16) {   // Insert the task here, there is time available today
                if (prev_ctpart){   // If there is a ctpart previously it's not the first entry of the day
                    Task *tmp_ctpart = prev_ctpart->counterpart;
                    prev_ctpart->counterpart = task;
                    task->counterpart = tmp_ctpart;
//                    Task *tmp_ctpart = current_task->counterpart;
//                    current_task->counterpart = task;
//                    task->counterpart = tmp_ctpart;
                } else {    // It's the first entry of the day
                    Task *tmp_prev = current_task->previous;
                    Task *tmp_next = current_task->next;
                    tmp_prev->next = task;
                    tmp_next->previous = task;
                    task->next = current_task->next;
                    task->previous = current_task->previous;
                    task->counterpart = current_task;
                    current_task->next = nullptr;
                    current_task->previous = nullptr;
                }
                // It's the first time of the first day, we found a new head
                if (task->day == head->day && task->time < head->time) {
                    head = task;
                    head->next = task->next;
                    head->previous = task->previous;
                }
            } else {    // Move it to the next day
                checkAvailableNextTimesFor(task);
            }
        }
    }
}

Task *WorkPlan::getTask(int day, int time) {
    //THIS FUNCTION WILL BE CODED BY YOU
}


void WorkPlan::checkAvailableNextTimesFor(Task *delayed) {
    //THIS FUNCTION WILL BE CODED BY YOU
}

void WorkPlan::delayAllTasksOfDay(int day) {
    //THIS FUNCTION WILL BE CODED BY YOU
}

void WorkPlan::remove(Task *target) {
    //THIS FUNCTION WILL BE CODED BY YOU
}

bool WorkPlan::checkCycledList() {
    Task *pivot = new Task();
    pivot = head;
    int patient = 100;
    bool r = false;
    while (pivot != NULL && patient > 0) {
        patient--;
        pivot = pivot->previous;
        if (pivot == head) {
            r = true;
            break;
        }
    }
    cout << "(" << 100 - patient << ")";
    patient = 100;
    bool l = false;
    while (pivot != NULL && patient > 0) {
        patient--;
        pivot = pivot->next;
        if (pivot == head) {
            l = true;
            break;
        }
    }
    return r & l;
}
