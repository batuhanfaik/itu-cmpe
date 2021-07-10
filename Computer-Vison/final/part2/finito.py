import numpy as np
import dlib
import cv2
import pyautogui
import time

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
def face_detect():
    global cc
    img = pyautogui.screenshot(region=(1680, 0, 235, 235))  # face coord, make 0 to 835 for exe
    img = np.array(img)[:, :, :3].astype(np.uint8)
    rec = detector(img)
    # cv2.imwrite(f'check{cc}.png',img)

    try:
        pts = predictor(img,rec[0])
        x1, y1 = pts.part(37).x, pts.part(37).y
        x2, y2 = pts.part(19).x, pts.part(19).y
        dist = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 1 / 2
        tmp = 1 if dist > 500 else 2
        return tmp
    except:# if no face
        return 0
def empty_check(direction):
    im = pyautogui.screenshot()
    im = np.array(im)[:, :, :3].astype(np.uint8)
    im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    x=958        # coord of monster
    y = 300
    if direction == 'A': #forward pixel checking
        x-= 130
    elif direction == 'D':
        x+=130
    elif direction == 'S':
        y+=130
    else:
        y-=130
    val = im[y,x]
    tmp = False if (val > 230 )^( val <80) else True ##empty check
    return tmp

def revert_direction(direction): #reversing the given direction
    if direction == 'A':
        return 'D'
    elif direction == 'D':
        return 'A'
    elif direction == 'W':
        return 'S'
    else :
        return 'W'

def move(direction,step=1,s=0.05): # Building block of movement
    for _ in range(step):
        pyautogui.keyDown(direction)
        time.sleep(s)
        pyautogui.keyUp(direction)

def change_dir(direction): #changing the given direction
    if direction == 'A':
        return 'S'
    elif direction == 'D':
        return 'A'
    elif direction == 'W':
        return 'D'
    else :
        return 'W'

def square_check(direction): # check wheter pixel white or black
    if direction == 'D':
        point = (290,920)
    elif direction == 'A':
        point = (290,997)
    elif direction == 'S':
        point = (212,954)
    else:
        point = (324,961)
    im = pyautogui.screenshot()
    im = np.array(im)[:, :, :3].astype(np.uint8)
    im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    if im[point] > 200:
        return 1 #white
    return 0 #black

def next_point(direction,ind): # returns the index of next point in the grid
    if direction == 'A':
        return (ind[0]-1,ind[1])
    elif direction == 'D':
        return (ind[0]+1,ind[1])
    elif direction == 'W':
        return (ind[0],ind[1]+1)
    else:
        return (ind[0],ind[1]-1)

def deadlock(grid,ind): # if deadlock occurs finds  the last visited cord
    if grid[ind[0]-1,ind[1]] ==2 :
        return 'A'
    elif grid[ind[0]+1,ind[1]] ==2:
        return 'D'
    elif  grid[ind[0],ind[1]+1] ==2:
        return 'W'
    elif grid[ind[0],ind[1]-1]:
        return 'S'

count = 0
dir = 'W'
time.sleep(2)
prev_bc = 1

grid = np.zeros((100,100))# 1 for blocked 2 seen
ind = (50,50)#x y initial point
grid[50,50]= 2 #make visited
bug_counter = 0
flag = 0

while True:
    # print(grid[50-5 : 50+5 ,50-5 : 50+5  ].T)
    if face_detect() ==0:
        exit()
    if grid[next_point(dir,ind)] == 0 and empty_check(dir) == False and face_detect() == 1 : # check face and emptiness and walk
        bc = square_check(dir)
        if prev_bc != bc: # if square changed
            print('Rite of a Passage')
            prev_bc = bc
            ind = next_point(dir,ind)
            grid[ind] = 2
            count = 0
            flag = 1
        move(dir,1,0.2)
        count+=1
        time.sleep(0.4)
        bug_counter = 0

    elif grid[ind[0]+1,ind[1]] !=0  and grid[ind[0]-1,ind[1]] !=0  and grid[ind[0],ind[1]+1] !=0  and grid[ind[0],ind[1]-1] !=0 : # if dead end occurs
        grid[ind] = 1
        r = deadlock(grid,ind)
        bc = square_check(dir)
        if prev_bc != bc:
            prev_bc = bc
            ind = next_point(r,ind)
        move(r, 1, 0.2)
        time.sleep(0.1)
        print('Life will find a way')

    elif grid[next_point(dir,ind)] !=0: # if next point is already checked, change direction
        dir = change_dir(dir)
        print('que sera sera ')

    elif (empty_check(dir) == True or face_detect() == 2) and bug_counter == 0: # either failed for face or emptiness
        print('YOU SHALL NOT PASS')
        if flag == 1 : #calculate halfstep for square change
            count = (count//2)
            if count == 0:
                count =1
            flag =0
        grid[next_point(dir, ind)] = 1 # make next grid blocked
        r = revert_direction(dir)
        for i in range(count):# move backwards
            move(r, 1, 0.2)
            time.sleep(0.1)
        dir = change_dir(dir)
        count = 0
        bug_counter+=1

    elif bug_counter != 0 : # if bug occurs do sth
            print('Ay Caramba')
            pyautogui.press(dir)
            pyautogui.press(revert_direction(dir))


