from ultralytics import YOLO  # Import the YOLO class

# Create a YOLO object with the specified model and other parameters
yolo = YOLO(model='./best.pt')

# Perform detection on images in the specified folder
results = yolo.predict(source='./images', save=True)

# Save results if needed
print(results)  # This will save images with detections

# To view or further process results, you can do the following:
# results.show()  # To display images
# results.xyxy  # To get raw detection results
