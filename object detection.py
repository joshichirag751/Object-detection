import cv2
from gui_buttons import Buttons



#initialize button
button =Buttons()
button.add_button("person",20,20)
button.add_button("cell phone",20,100)
#opencv dnn
net =cv2.dnn.readNet("dnn_model/yolov4-tiny.weights","dnn_model/yolov4-tiny.cfg")
model=cv2.dnn_DetectionModel(net)
model.setInputParams(size=(320,320),scale=1/255)

#load class list
classes =[]
with open ("dnn_model/classes.txt","r") as file_object:
    for class_name in file_object.readlines():
        #print(class_name)
        class_name=class_name.strip()
        classes.append(class_name)
print("classes",classes)



#initialize cam
cap =cv2.VideoCapture(0)
#cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
#FULL HD 1920*1080

#button_person= False
def click_button(events,x,y,flags,params):
    global button_person
    if events==cv2.EVENT_LBUTTONDOWN:
        #print(x,y)
        button.button_click(x,y)
        """polygon = np.array([[(20, 20), (220, 20), (220, 70), (20, 70)]])

        is_inside =cv2.pointPolygonTest(polygon,(x,y),False)
        if is_inside>0:
            print("we're clicking",x,y)

            if button_person is False:
                button_person=True
            else:
                 button_person=False
            print("now button person is",button_person)"""
#creating a window
cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame",click_button)

while True:
    #frames
    ret,frame =cap.read()
    # get button activated
    active_buttons=button.active_buttons_list()
    print("active button",active_buttons)
    #object detect

    (class_ids,scores,bboxes)= model.detect(frame)
    for class_id,score,bbox in zip(class_ids,scores,bboxes):
        x,y,w,h =bbox
        class_name=classes[class_id]
        #print(class_name)
        #print(x,y,w,h)
        #if class_name=="person"and button_person is True:
        if class_name in active_buttons:
            cv2.putText(frame,class_name,(x,y-10),cv2.FONT_HERSHEY_PLAIN,2,(200,0,50),2)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(200,0,50),2)


    #print("class ids",class_ids)
    #print("scores ",scores )
    #print("bboxes",bboxes )

    #create button
    #cv2.rectangle(frame,(20,20),(220,70),(0,0,200),-1)
    #polygon = np.array([[(20, 20), (220, 20), (220, 70), (20, 70)]])
    #cv2.fillPoly(frame,polygon,(0,0,200))
    #cv2.putText(frame,"Person",(30,60),cv2.FONT_HERSHEY_PLAIN,3,(255,255,255),3)
#display buttons
    button.display_buttons(frame)


    cv2.imshow("Frame",frame)
    cv2.waitKey(1)




