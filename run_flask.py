#used to open the browser shortly after the flask server starts
import threading
# opens the application on its own in the users default browser
import webbrowser
# imports the flask application factory
from flask_app import create_app
# imports the embedding model loader used by the resource retriever
from services.assistant_service import get_embedding_model
# imports the helper used to warm up the local language model
from services.llm_service import warm_llm


# create the flask application instance
app = create_app()


# opens the local application address in the default web browser
def open_browser():
    # open the flask application running on the local machine
    webbrowser.open(
        "http://127.0.0.1:5000"
    )


# loads the main ai models before the application is used
def warm_models():
    # show progress while the resource retrieval model is loading
    print("Loading resource retriever")
    # load the embedding model into memory
    get_embedding_model()
    # confirm that the resource retriever is ready
    print("Resource retriever ready")
    # show progress while the local language model is loading
    print("Loading local LLM")
    # warm the local language model so the first request is faster
    warm_llm()


# run this section only when the file is started directly
if __name__ == "__main__":
    # load the main models before starting the web application
    warm_models()
    # open the browser 1sec after the application begins starting
    threading.Timer(
        1.0,
        open_browser
    ).start()
    # start the flask development server on the local computer
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        use_reloader=False
    )