import cv2
import glob
drawing = False #鼠标按下为真
mode = True #如果为真，画矩形，按m切换为曲线
ix,iy=-1,-1
px,py=-1,-1
i = 0
position_point=[]
position_rect =[]
is_belt =True
def draw_circle(event,x,y,flags,param):
    global ix,iy,drawing,px,py,i,img,is_belt
    if event == cv2.EVENT_RBUTTONDOWN :             #右键点击
        is_belt=  not is_belt
        cv2.putText(img, "no safety belt", (50, 80), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 1)

    if event == cv2.EVENT_LBUTTONDOWN:              #左键点击
        drawing = True
        ix,iy=x,y
        cv2.circle(img, (ix,iy), 5, (255,0,0), -1)
        print("在画圆")
        i =i+1
        position_point.append([ix,iy])
        print(i,ix,iy)
    elif event == cv2.EVENT_MOUSEMOVE  :              #滑动
        if drawing == True:
            #cv2.rectangle(img,(ix,iy),(px,py),(255,0,0),10)#将刚刚拖拽的矩形涂黑
            #cv2.rectangle(img,(ix,iy),(x,y),(0,255,0),0)
            img = image.copy()
            cv2.line(img, (ix,iy), (x,iy), (255, 0,0),2)
            cv2.line(img, (ix,iy), (ix,y), (0, 255,0),2)
            cv2.line(img, (x, iy), (x, y), (0, 255, 0), 2)
            cv2.line(img, (ix, y), (x, y), (0, 255, 0), 2)
            print("在画线")
            px,py=x,y
            position_rect.append([px,py])
    elif event == cv2.EVENT_LBUTTONUP:               #左键释放
        if drawing ==True:
        #cv2.circle(img, (ix,iy), 5, (0,0,255), -1)
            cv2.rectangle(img,(ix,iy),(x,y),(0,0,255),2)
            print("在画框")
        drawing = False
        px,py=-1,-1
def write_txt_point(x, path):
    file_obj = open(path, 'w')
    line ="version: 1"+ '\n' +"n_points:  "+str(len(x))+ '\n'
    file_obj.write(line)
    file_obj.write('{\n')
    for index in range(len(x)):
        line = str(x[index][0]) + '  ' + str(x[index][1]) + '\n'
        file_obj.write(line)
    file_obj.write('}\n')
    file_obj.write(str(is_belt))
def write_txt_rect(x,path):
    file_obj = open(path, 'w')
    line = str(x[0][0]) + '  ' + str(x[0][1]) +'  '+str(x[-1][0]-x[0][0]) + '  ' + str(x[-1][1]-x[0][1])
    file_obj.write(line)
def get_picture(path):
    lines = []
    file = open(path)
    for line in file.readlines():
        line=line.strip('\n')
        lines.append(line)
    return lines
lines = get_picture("address.txt")

lines = glob.glob(r'picture/*.png')
pts =glob.glob(r'picture/*.pts')
print(lines)
print(pts)
for index in range(len(lines)):
    if lines[index].replace('png','pts') in pts:
        continue
    image =cv2.imread(lines[index])                  #原图
    img = image.copy()                               #副本
    cv2.namedWindow('image')
    cv2.setMouseCallback('image',draw_circle)
    while(1):
      #  cv2.putText(img, 'Please mark the face box and seat belt '  , (200,300), cv2.FONT_HERSHEY_PLAIN,2.0,(0,255,0),2)
        cv2.imshow('image',img)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q') :#下一张图
            write_txt_point(position_point[1:], str(lines[index][:-4])+".pts")
            print("fdsadfs",position_rect)
            write_txt_rect( position_rect,   str(lines[index][:-4]) + ".rect")
            position_point=[]
            position_rect=[]
            i=0
            break
        if k ==ord('o'):#终结程序
            exit()
        if k ==ord('c'):#完全刷新
            img =image.copy()
            position_point=[]
            position_rect=[]
            i=0
            is_belt =True
        elif k == 27:
            break
    cv2.destroyAllWindows()


