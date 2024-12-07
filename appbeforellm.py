from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg 
    return get_chat_response(input)

def get_chat_response(text):
    # Here you can generate your response based on user_input
    response = "This is a response to your message."
    # Append image HTML to the response
    response += '<br><div style="max-width: 100%; overflow: hidden;"><img src="/static/image.png" alt="Image" style="max-width: 100%; height: auto;"></div>'
    return jsonify(response)


if __name__ == '__main__':
    app.run()