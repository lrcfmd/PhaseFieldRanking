import os
import webbrowser
from notebook import notebookapp



def start_jupyter():
    # Path to the notebook
    notebook_path = os.path.abspath("EXE/PhaseFieldRanking.ipynb")
    
    # Start the Jupyter Notebook server

    #os.system("jupyter notebook")
    server = notebookapp.NotebookApp()
    server.notebook_file = notebook_path
    server.port = 8888
    server.open_browser=True
    server_thread = server.start()
    
    # Automatically open it in the browser
    # Open in default browser (or specify a browser)
    webbrowser.open(server.url)
    #webbrowser.open("http://localhost:8888")

if __name__ == "__main__":
    start_jupyter()
