from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np

app = Flask(__name__)

# โหลด TFLite model
interpreter = tf.lite.Interpreter(model_path="best_float32.tflite")
interpreter.allocate_tensors()

# ดึง input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

@app.route('/')
def index():
    return "TFLite Model API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # สมมุติว่าผู้ใช้ส่ง JSON แบบ {"input": [1.0, 2.0, 3.0, ...]}
        data = request.get_json()
        input_data = np.array(data["input"], dtype=np.float32)
        input_data = input_data.reshape(input_details[0]['shape'])

        # รันโมเดล
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])

        # ส่งผลลัพธ์กลับ
        return jsonify({"prediction": output_data.tolist()})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
