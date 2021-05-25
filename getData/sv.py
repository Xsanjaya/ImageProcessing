import cv2

cap=cv2.VideoCapture(0)
gambarr=1  # membuat variable untuk penamaan file yang akan ditulis nanti
while True :
    _,img=cap.read()
    cv2.imshow("tekan s untuk simpan foto",img)

    k=cv2.waitKey(1) & 0xFF
    if k==27:
        cap.release()
        cv2.destroyAllWindows()
        break
    if k==ord('s'):# jika tombol 's' kecil ditekan
              fileN=str(gambarr)+'.png'# membuat string nama image yang disimpan
              cv2.imwrite(fileN,img)# simpan image di folder yang aktif sekarang
              print (gambarr)   # menampilkan nama image yang telah tersimpan di command prompt(CMD) / console terminal
              gambarr = gambarr+1 # increase variable penamaan image agar penyimpanan selanjutnya tidak menimpa file yang lama
