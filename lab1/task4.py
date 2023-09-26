import cv2

cap = cv2.VideoCapture(r'..\images\sample-5s.mp4', cv2.WINDOW_NORMAL)
cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Video', 800, 600)

# получение размеров кадра
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#выходной объект
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('../images/output_1.mp4', fourcc, 30.0, (width, height)) #30-кадры в секунду

ret, frame = cap.read()
while(ret):
    out.write(frame)
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break
    ret, frame = cap.read()
    if not(ret):
        break
cap.release()
out.release()
cv2.destroyAllWindows()