import static.request as request
import json

test_path = "C://Users//Windows//Desktop//PESEKI//test/"

data = {'image': test_path}
URL = "http://127.0.0.1:3000"

result = request.post(URL,json.dumps(data))


print(f"Prediction = {result.text}")