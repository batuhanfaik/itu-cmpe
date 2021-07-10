- init:
    run: rm -f q1.o q1
    blocker: true

- build:
    run: g++ -std=c++11 -Wall -Werror  q1.cpp -o q1.out              # timeout: 8
    blocker: true

- case1:
    run: ./q1.out test1.txt
    points: 5
    script:
        - expect: "###########RESULTS###########\r\n\r\n1. WT ARRAY\r\n------------------------\r\n  1000, 4, 3, 3, 3, 3, 3, 3\r\n\r\n2. MAXIMUM CAPACITY PATH\r\n------------------------\r\n  0 -> 4 -> 5 -> 7\r\n\r\n3. MAXIMUM CAPACITY\r\n------------------------\r\n  3\r\n#############################\r\n"           # timeout: 8
        - expect: _EOF_                                              # timeout: 8

    return: 0

- case2:
    run: ./q1.out test2.txt
    points: 5
    script:
        - expect: "###########RESULTS###########\r\n\r\n1. WT ARRAY\r\n------------------------\r\n  1000, 8, 8, 8, 7, 6, 7, 7, 7\r\n\r\n2. MAXIMUM CAPACITY PATH\r\n------------------------\r\n  0 -> 2 -> 1 -> 3 -> 6 -> 8\r\n\r\n3. MAXIMUM CAPACITY\r\n------------------------\r\n  7\r\n#############################\r\n"           # timeout: 8
        - expect: _EOF_                                              # timeout: 8

    return: 0
