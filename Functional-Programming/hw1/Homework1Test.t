- init:
    run: rm -f homework1
    blocker: true

- build:
    run: ghc homework1.hs -o homework1
    blocker: true

- case_1:
    run: ./homework1 d2c 16 14
    script:
        - expect: "^'E'\r\n"

- case_2:
    run: ./homework1 d2c 8 5
    script:
        - expect: "^'5'\r\n"

- case_3:
    run: ./homework1 d2c 10 14
    script:
        - expect: ": invalid digit\r\n"
    exit: 1

- case_4:
    run: ./homework1 c2d 16 E
    script:
        - expect: "^14\r\n"

- case_5:
    run: ./homework1 c2d 8 7
    script:
        - expect: "^7\r\n"

- case_6:
    run: ./homework1 c2d 8 A
    script:
        - expect: ": invalid digit\r\n"
    exit: 1

- case_7:
    run: ./homework1 n2l 8 21
    script:
        - expect: "^\\[2,5\\]\r\n"

- case_8:
    run: ./homework1 n2l 16 140
    script:
        - expect: "^\\[8,12\\]\r\n"

- case_9:
    run: ./homework1 l2n 8 2 5
    script:
        - expect: "^21\r\n"

- case_10:
    run: ./homework1 l2n 16 8 12
    script:
        - expect: "^140\r\n"

- case_11:
    run: ./homework1 add 16 77 330
    script:
        - expect: "^\\[4,13\\]\r\n"
        - expect: '^"4D"\r\n'
        - expect: "^\\[1,4,10\\]\r\n"
        - expect: '^"14A"\r\n'
        - expect: "^\\[1,9,7\\]\r\n"
        - expect: '^407\r\n'
