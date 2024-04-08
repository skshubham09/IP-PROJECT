from pre import process_image
import cv2
#image_path = 'kidney_image.jpg'
image_path = 'stone.png'

processed_image = process_image(image_path)
cv2.imshow('Processed Image', processed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
