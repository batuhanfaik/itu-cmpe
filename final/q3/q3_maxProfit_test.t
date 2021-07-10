- init:
    run: rm -f q.o q
    blocker: true

- build:
    run: g++ -std=c++11 -Wall -Werror q3_maxProfit.cpp -o q3_maxProfit.out                  # timeout: 8
    blocker: true

- case1:
    run: ./q3_maxProfit.out
    points: 10
    script:
        - expect: "Enter the name of the input file: "                       # timeout: 8
        - send: "q3_maxProfit_test1.txt"
        - expect: "Dynaming Programming Table\r\n  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0\r\n  0  0  0  0  0  7  7  7  7  7  7  7  7  7  7\r\n  0  0  0  0  0  7  7  7  7  7  7  7 12 12 12\r\n  0  0  0  3  3  7  7  7 10 10 10 10 12 12 12\r\n  0  0  0 10 10 10 13 13 17 17 17 20 20 20 20\r\n  0  0  2 10 10 12 13 13 17 17 19 20 20 22 22\r\nMax profit is 22.\r\nCities visited: 1 3 4 5\r\n"                       # timeout: 8
        - expect: _EOF_                                         # timeout: 8

    return: 0

- case2:
    run: ./q3_maxProfit.out
    points: 10
    script:
        - expect: "Enter the name of the input file: "                       # timeout: 8
        - send: "q3_maxProfit_test2.txt"
        - expect: "Dynaming Programming Table\r\n  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0\r\n  0  0  0  0  0  7  7  7  7  7  7  7  7  7  7\r\n  0  0  0  0  0  7  7  7  7  7 12 12 12 12 12\r\n  0  3  3  3  3  7 10 10 10 10 12 15 15 15 15\r\n  0  3  3 10 13 13 13 13 17 20 20 20 20 22 25\r\nMax profit is 25.\r\nCities visited: 1 2 3 4\r\n"                       # timeout: 8
        - expect: _EOF_                                         # timeout: 8

    return: 0

- case3:
    run: ./q3_maxProfit.out
    points: 10
    script:
        - expect: "Enter the name of the input file: "                       # timeout: 8
        - send: "q3_maxProfit_test3.txt"
        - expect: "Dynaming Programming Table\r\n  0  0  0  0  0  0  0  0  0  0  0\r\n  0  0  0  0  0  7  7  7  7  7  7\r\n  0  0  0 10 10 10 10 10 17 17 17\r\n  0  0  2 10 10 12 12 12 17 17 19\r\nMax profit is 19.\r\nCities visited: 1 2 3\r\n"                         # timeout: 8
        - expect: _EOF_                                         # timeout: 8

    return: 0