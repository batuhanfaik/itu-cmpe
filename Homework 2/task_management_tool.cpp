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

void WorkPlan::setUsableDay(int d) {
    usable_day = d;
}

void WorkPlan::setUsableTime(int t) {
    usable_time = t;
}

void WorkPlan::create() {
    head = nullptr;
}

void WorkPlan::close() {
    //THIS FUNCTION WILL BE CODED BY YOU
    Task *this_day = head;
    Task *this_task;

    while (this_day) {
        Task *next_day = this_day->next;
        this_task = this_day;
        while (this_task) {
            Task *to_delete = this_task;
            this_task = this_task->counterpart;
            remove(to_delete);
        }
        // If there is no next day, we deleted the last day
        if (this_day != next_day) {
            this_day = next_day;
        } else {
            this_day = nullptr;
        }
    }
}

void WorkPlan::add(Task *task) {
    // Allocate new memory for tasks
    Task *new_task = new Task;
    new_task->name = new char[strlen(task->name) + 1];
    strcpy(new_task->name, task->name);
    new_task->day = task->day;
    new_task->time = task->time;
    new_task->priority = task->priority;
    task = new_task;

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
            if (task->day < head->day) {     // It's the first day, we found a new head
                head = task;
                head->next = task->next;
                head->previous = task->previous;
            }
        } else {    // There exists a day with previous tasks
            // Look for the time opening (as failsafe as possible)
            Task *prev_ctpart = nullptr;
            Task *tmp = current_task;
            while (tmp) {
                if (current_task->time < task->time) {
                    prev_ctpart = current_task;
                    if (current_task->counterpart) {
                        current_task = current_task->counterpart;
                    }
                }
                tmp = tmp->counterpart;
            }

            if (current_task->time <= 16) {   // Insert the task here, there is time available today
                // If there is an existing entry with higher priority delay it
                if (current_task->time == task->time && current_task->priority >= task->priority) {
                    checkAvailableNextTimesFor(task);
                } else {
                    if (prev_ctpart) {   // If there is a ctpart previously it's not the first entry of the day
                        Task *tmp_ctpart = prev_ctpart->counterpart;
                        prev_ctpart->counterpart = task;
                        task->counterpart = tmp_ctpart;
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
                }
                if (current_task->time == task->time &&
                    task->priority > current_task->priority) {  // If there is a task with less priority delay it
                    checkAvailableNextTimesFor(current_task);
                }
            } else {    // Move it to the next day
                checkAvailableNextTimesFor(task);
            }
        }
    }
}

Task *WorkPlan::getTask(int day, int time) {
    //THIS FUNCTION WILL BE CODED BY YOU
    Task *search_task = head;

    while (search_task->day != day) {    // Find the day
        search_task = search_task->next;
        if (search_task == head) {   // If it loops back to the head no day found
            return nullptr;
        }
    }
    while (search_task->time != time) {  // Find the time
        search_task = search_task->counterpart;
        if (!search_task) {  // If it reaches the end of counterparts no time found
            return nullptr;
        }
    }
    // Return the searched task
    return search_task;
}


void WorkPlan::checkAvailableNextTimesFor(Task *delayed) {
    //THIS FUNCTION WILL BE CODED BY YOU
    int available_day = delayed->day;
    int available_time = delayed->time;
    int last_day = head->previous->day;
    bool task_is_delayed = false;
    Task *day_holder = head;
    while (day_holder->day != delayed->day) {
        day_holder = day_holder->next;
    }
    // If the head is being delayed make sure to move the head
    if (delayed == head) {
        if (head->counterpart) {
            head = head->counterpart;
        } else if (head->next) {
            head = head->next;
        }
    }

    // If the task being delay is the first task of a day
    if ((delayed->time == 8 || delayed->time == 25) && ((delayed->next) || delayed->previous)) {
        if (delayed->counterpart) {  // Connect days on side the now first task
            Task *now_first_task = delayed->counterpart;
            now_first_task->next = delayed->next;
            now_first_task->previous = delayed->previous;
            delayed->next->previous = now_first_task;
            delayed->previous->next = now_first_task;
        } else {    // There is no counterpart, delete day
            if (delayed->next->previous) {
                delayed->next->previous = delayed->previous;
            }
            if (delayed->previous->next) {
                delayed->previous->next = delayed->next;
            }
        }
    }

    available_time++;
    while (available_day <= last_day && !task_is_delayed) {
        while (available_time >= 8 && available_time <= 16 && !task_is_delayed) {
            // If there exists a task in the given time and day
            if (getTask(available_day, available_time)) {
                available_time++;
            } else {    // Delay the task
                if (available_time == 8) {   // Insert as first task of the day
                    int first_time_of_day = 9;
                    Task *insert_before = nullptr;
                    while (first_time_of_day <= 16 and !insert_before) {     // Coding as failsafe as possible
                        insert_before = getTask(available_day, first_time_of_day);
                        first_time_of_day++;
                    }
                    if (insert_before) { // If there is a task to insert before (theoretically there should be) insert
                        delayed->previous = insert_before->previous;
                        delayed->next = insert_before->next;
                        delayed->counterpart = insert_before;
                        delayed->previous->next = delayed;
                        delayed->next->previous = delayed;
                        delayed->day = available_day;
                        delayed->time = available_time;
                        setUsableDay(available_day);
                        setUsableTime(available_time);
                    } else {
                        cout << "There are no tasks on this day!" << endl;
                    }

                } else if (available_time <= 16) {   // Insert as counterpart of an existing task
                    Task *insert_after = getTask(available_day, (available_time - 1));
                    Task *tmp_ctpart = insert_after->counterpart;
                    if (tmp_ctpart == delayed) {     // Trying to add a higher priority task
                        tmp_ctpart = delayed->counterpart;
                    }
                    insert_after->counterpart = delayed;
                    delayed->counterpart = tmp_ctpart;
                    delayed->day = available_day;
                    delayed->time = available_time;
                    setUsableDay(available_day);
                    setUsableTime(available_time);
                }

                task_is_delayed = true;
            }
        }
        available_time = 8;
        available_day = day_holder->next->day;
        day_holder = day_holder->next;
    }

    if (!task_is_delayed) {
        cout << "No available time in the schedule!" << endl;
    }
}

void WorkPlan::delayAllTasksOfDay(int day) {
    Task *to_delay = head;
    // Find the first task of day
    while (to_delay->day != day) {
        to_delay = to_delay->next;
    }
    // If the user is trying to delay the last day
    if (to_delay->next == head) {
        cout << "Sorry, can't delay the last day!" << endl;
    } else {
        // Delay the task to next available space
        while (to_delay) {
            to_delay->time = 25;
            Task *tmp_ctpart = to_delay->counterpart;
            checkAvailableNextTimesFor(to_delay);
            to_delay = tmp_ctpart;
        }
    }
}

void WorkPlan::remove(Task *target) {
    //THIS FUNCTION WILL BE CODED BY YOU
    Task *search_task = head;
    Task *prev_task = nullptr;

    while (search_task->day != target->day) {    // Find the day
        search_task = search_task->next;
        if (search_task == head) {   // If it loops back to the head no day found
            search_task = nullptr;
        }
    }
    if (search_task) {
        while (search_task->time != target->time) {  // Find the time
            prev_task = search_task;
            search_task = search_task->counterpart;
            if (!search_task) {  // If it reaches the end of counterparts no time found
                search_task = nullptr;
            }
        }
    }

    // If the head is being removed assign new head
    if (target == head) {
        if (head->counterpart) {
            head = head->counterpart;
        } else if (head->next) {
            head = head->next;
        }
    }

    if (prev_task) {     // If not the first task of day
        prev_task->counterpart = target->counterpart;
    } else if (target->counterpart) {   // If first task of day and there are more tasks in the day
        target->next->previous = target->counterpart;
        target->previous->next = target->counterpart;
        target->counterpart->next = target->next;
        target->counterpart->previous = target->previous;
    } else {    // If first task of day and there are nor any tasks in the day
        target->next->previous = target->previous;
        target->previous->next = target->next;
    }

    delete target->name;
    delete target;
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
