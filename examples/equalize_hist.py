import cv2
import torch
import fastcv

img = cv2.imread("../artifacts/db5-vantage.jpg", cv2.IMREAD_GRAYSCALE)
img_tensor = torch.from_numpy(img).cuda()
equalized_tensor = fastcv.equalize(img_tensor) #add correct params from torch::Torch equalizeHist in .cu file!!!!!

equalized_image = equalized_tensor.cpu().numpy() #CHECK IF CORRECT
cv2.imwrite("output_equalized.jpg", equalized_image) #this should be fine

print("saved image with equalized histogram.")