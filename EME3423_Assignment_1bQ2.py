import cv2

img = cv2.imread('Resources/logo4.jpg')

if img is not None:
    img1 = img
    img2 = cv2.GaussianBlur(img, (15, 15), 0)
    img3 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    img4 = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    cv2.imshow('Original', img1)
    cv2.imshow('Gaussian Blur', img2)
    cv2.imshow('HSV Color Space', img3)
    cv2.imshow('Canny Edges', img4)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
