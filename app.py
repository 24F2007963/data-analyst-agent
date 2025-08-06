from flask import Flask, request, jsonify
import base64
import os

app = Flask(__name__)

# Dummy Response Generator (Replace with actual processing logic)
def generate_dummy_response():
    # Example dummy data matching evaluation format
    answer1 = 1  # Example: Number of $2bn movies before 2000
    answer2 = "Titanic"  # Example: Movie name
    answer3 = 0.485782  # Example: Correlation value

    # Generate a small dummy plot image encoded in base64
    dummy_image_data = b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGMAAQAABQABDQottAAAAABJRU5ErkJggg=="
    image_base64 = f"data:image/png;base64,{dummy_image_data.decode()}"

    return [answer1, answer2, answer3, image_base64]

@app.route('/api/', methods=['POST'])
def analyze():
    if 'questions.txt' not in request.files:
        return jsonify({'error': 'questions.txt file is required'}), 400

    questions_file = request.files['questions.txt']
    questions_text = questions_file.read().decode('utf-8')

    # Process other attachments if needed
    attachments = {}
    for file_key in request.files:
        if file_key != 'questions.txt':
            attachments[file_key] = request.files[file_key]

    # Placeholder: Log input files (later use for parsing/processing)
    print("Received Questions:", questions_text)
    print("Received Attachments:", list(attachments.keys()))

    # TODO: Replace with actual task processing logic
    response = generate_dummy_response()

    return jsonify(response)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
