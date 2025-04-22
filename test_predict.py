import requests
import json
import os
import traceback

# Specify the image path
image_path = 'yolov5/data/images/zidane.jpg'

# Check if the image exists
if not os.path.exists(image_path):
    print(f"Image {image_path} not found!")
    exit(1)

# Create the request
url = 'http://127.0.0.1:8000/predict'
files = {'file': open(image_path, 'rb')}

# Send the image to the server
print(f"Sending image {image_path} to {url}...")
try:
    response = requests.post(url, files=files)
    
    # Display the result
    print(f"Response code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print("Result:")
        print(json.dumps(result, indent=4, ensure_ascii=False))
    else:
        print("Error occurred:", response.text)
        try:
            error_json = response.json()
            print("Error details:", json.dumps(error_json, indent=4))
        except:
            print("Non-JSON error response")
except Exception as e:
    print(f"Exception: {str(e)}")
    traceback.print_exc()
finally:
    # Make sure to close the file
    files['file'].close() 