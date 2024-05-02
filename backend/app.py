from flask import Flask

app = Flask(__name__)
api_base = "/api"


if __name__ == '__main__':
    app.run(use_reloader=False)