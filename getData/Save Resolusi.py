import cv2

vidCap=cv2.VideoCapture(0)
count = 1

def make_480p():
    vidCap.set(3, 640)
    vidCap.set(4, 480)
    
def make_720p():
    vidCap.set(3, 1280)
    vidCap.set(4, 720)

def change_res(width, height):
    vidCap.set(3, width)
    vidCap.set(4, height)

# make_480p()
# change_res(360, 240)

while True :
    ret, vidOri = vidCap.read()

    vidOri = change_res(240, 240)
    cv2.imshow("Image Ori", vidOri)

    #cv2.imshow("Image 480p ", xx)

    key=cv2.waitKey(1) & 0xFF
    if key==27 or key == ord('q'):               #tekan esc untuk keluar
        vidCap.release()
        cv2.destroyAllWindows()
        break
    if key == ord('s'):
        file_name1 = 'Ori '+str(count)+'.jpg'
        file_name2 = 'Grey '+str(count)+'.jpg'
        count = count+1

        # simpan image di folder yang aktif sekarang
        cv2.imwrite(file_name1, vidOri)
        cv2.imwrite(file_name2, vidGrey)
        print ("Image", file_name1, file_name2, "Telah Tersimpan")
