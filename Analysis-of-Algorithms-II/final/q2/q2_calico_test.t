- init:
    run: rm -f q2.o q2
    blocker: true

- build:
    run: g++ -std=c++11 -Wall -Werror  q2.cpp -o q2              # timeout: 8
    blocker: true

- case1:
    run: ./q2 points7.txt
    points: 1
    script:
        - expect: "Distance between the closest points: 5\r\n"           # timeout: 8
        - expect: _EOF_                                              # timeout: 8

    return: 0

- case2:
    run: ./q2 points10.txt
    points: 1
    script:
        - expect: "Distance between the closest points: 21.3776\r\n"           # timeout: 8
        - expect: _EOF_                                              # timeout: 8

    return: 0

- case3:
    run: ./q2 points20.txt
    points: 1
    script:
        - expect: "Distance between the closest points: 71.0634\r\n"           # timeout: 8
        - expect: _EOF_                                              # timeout: 8

    return: 0