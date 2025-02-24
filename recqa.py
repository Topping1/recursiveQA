import sys
import json
import requests

from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import QThread, pyqtSignal

# Import tiktoken for token counting
import tiktoken

# Import the RecursiveChunker from chonkie
from chonkie import RecursiveChunker, RecursiveRules

# --- Model Configuration ---
MODEL_NAME = "Qwen2.5-3B-Instruct-Q8_0"
MAX_MODEL_TOKENS = 4026      # Total context tokens allowed by the model, taking into account
                             # the extra tokens used by the instructions in the prompt sent to the model
SAFETY_FACTOR = 1.2          # Safety factor for user query tokens

API_URL = "http://127.0.0.1:8080/v1/chat/completions"  # Update if your local server uses a different endpoint

# --- Utility Functions ---

def count_tokens(text: str, encoding_name="gpt2") -> int:
    """
    Counts the number of tokens in a text using the specified encoding.
    """
    encoding = tiktoken.get_encoding(encoding_name)
    tokens = encoding.encode(text)
    return len(tokens)

def query_llm(user_query: str, text_chunk: str) -> str:
    """
    Sends a request to an OpenAI-compatible Chat Completions API with a prompt that
    concatenates the user_query and the text_chunk.
    The prompt is sent as a user message following a developer message.
    """
    # Create the prompt by concatenating the user query and the current chunk.
    prompt = (
        f"Your task is to answer the user query using ONLY the information from the provided context. "
        f"If the user query cannot be answered, respond with 'This chunk has no information about the user query'. "
        f"The user query: {user_query}\nThe context: {text_chunk}"
    )
    
    body = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful AI assistant. Your top priority is to analyze carefully the user request and the context to fulfill the user request."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.1  # Example temperature setting
    }

    try:
        response = requests.post(API_URL, json=body)
        response.raise_for_status()
        completion = response.json()
        return completion["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"Error during API call: {str(e)}"

# --- Worker Thread for Processing ---
class ProcessWorker(QThread):
    progress_update = pyqtSignal(int)   # emits current progress (percentage)
    verbose_update = pyqtSignal(str)    # emits log messages (shown in GUI)
    finished_signal = pyqtSignal(str)   # emits final result

    def __init__(self, file_path: str, user_query: str, verbose: bool = False):
        super().__init__()
        self.file_path = file_path
        self.user_query = user_query
        self.verbose = verbose

    def log_always(self, message: str):
        """Always emit log messages to the GUI, regardless of `verbose`."""
        self.verbose_update.emit(message)

    def log_partial_prompt(self, message: str):
        """Emit partial prompt logs ONLY if `verbose` is True."""
        if self.verbose:
            self.verbose_update.emit(message)

    def run(self):
        # Read the entire file
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                text = f.read()
        except Exception as e:
            self.finished_signal.emit(f"Error reading file: {str(e)}")
            return

        # Begin recursive processing
        final_answer = self.recursive_process(text, self.user_query)
        self.finished_signal.emit(final_answer)

    def recursive_process(self, text: str, user_query: str) -> str:
        """
        Process the text in rounds until the concatenated answer is within token limits.
        """
        token_count_text = count_tokens(text)
        self.log_always(f"Current text token count: {token_count_text}")

        # If the text is within our max context, do one final query and return
        if token_count_text < MAX_MODEL_TOKENS:
            self.log_always("Text is within context limit. Sending final query...")
            final = query_llm(user_query, text)
            self.log_always("Final answer obtained.")
            return final
        else:
            # Compute allowed chunk size: subtract the (user query token count * SAFETY_FACTOR)
            query_tokens = count_tokens(user_query)
            allowed_chunk_size = int(MAX_MODEL_TOKENS - (query_tokens * SAFETY_FACTOR))
            if allowed_chunk_size <= 0:
                return "Error: User query is too long."

            self.log_always(f"Allowed chunk size: {allowed_chunk_size} tokens.")

            # Initialize the chunker with the computed chunk size
            chunker = RecursiveChunker(
                tokenizer_or_token_counter="gpt2",
                chunk_size=allowed_chunk_size,
                rules=RecursiveRules(),
                min_characters_per_chunk=12,
                return_type="chunks"  # returns objects with a .text attribute
            )
            chunks = chunker(text)
            self.log_always(f"Document split into {len(chunks)} chunks.")

            answers = []
            total_chunks = len(chunks)
            for i, chunk in enumerate(chunks, start=1):
                # Update progress bar
                self.progress_update.emit(int((i / total_chunks) * 100))
                chunk_text = chunk.text

                self.log_always(f"Processing chunk {i}/{total_chunks} (token count: {chunk.token_count}).")

                # Show partial prompt only if verbose is True
                prompt_for_chunk = (
                    f"Your task is to answer the user query using ONLY the information from the provided context. "
                    f"If the user query cannot be answered, respond with 'This chunk has no information about the user query'. "
                    f"The user query: {self.user_query}\nThe context: {chunk_text}"
                )
                self.log_partial_prompt(f"Prompt for chunk {i}: {prompt_for_chunk}")

                answer = query_llm(user_query, chunk_text)
                self.log_always(f"Answer for chunk {i}: {answer}")

                answers.append(answer)

            concatenated = "\n".join(answers)
            self.log_always("Concatenated answer length (in tokens): " + str(count_tokens(concatenated)))
            self.log_always("Recursing on concatenated answers...")

            # Recursively process the concatenated answer
            return self.recursive_process(concatenated, user_query)

# --- Main GUI Window ---
class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Recursive LLM File Processor")
        self.setup_ui()

    def setup_ui(self):
        layout = QtWidgets.QVBoxLayout()

        # File selection
        file_layout = QtWidgets.QHBoxLayout()
        self.file_line_edit = QtWidgets.QLineEdit()
        browse_button = QtWidgets.QPushButton("Browse File")
        browse_button.clicked.connect(self.browse_file)
        file_layout.addWidget(QtWidgets.QLabel("File:"))
        file_layout.addWidget(self.file_line_edit)
        file_layout.addWidget(browse_button)
        layout.addLayout(file_layout)

        # User query input (multiline)
        self.query_text_edit = QtWidgets.QTextEdit()
        self.query_text_edit.setPlaceholderText("Enter your query here...")
        layout.addWidget(QtWidgets.QLabel("User Query:"))
        layout.addWidget(self.query_text_edit)

        # Verbose output checkbox
        self.verbose_checkbox = QtWidgets.QCheckBox("Show verbose output")
        layout.addWidget(self.verbose_checkbox)

        # Process button
        self.process_button = QtWidgets.QPushButton("Process File")
        self.process_button.clicked.connect(self.start_processing)
        layout.addWidget(self.process_button)

        # Progress bar
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 100)
        layout.addWidget(self.progress_bar)

        # Output text area (read-only with scrollbars)
        self.output_text_edit = QtWidgets.QTextEdit()
        self.output_text_edit.setReadOnly(True)
        layout.addWidget(QtWidgets.QLabel("Output:"))
        layout.addWidget(self.output_text_edit)

        self.setLayout(layout)

    def browse_file(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select File", "", "Text Files (*.txt);;All Files (*)"
        )
        if file_path:
            self.file_line_edit.setText(file_path)

    def start_processing(self):
        file_path = self.file_line_edit.text().strip()
        user_query = self.query_text_edit.toPlainText().strip()
        verbose = self.verbose_checkbox.isChecked()

        if not file_path:
            self.output_text_edit.append("Please select a file.")
            return
        if not user_query:
            self.output_text_edit.append("Please enter a query.")
            return

        self.output_text_edit.clear()
        self.progress_bar.setValue(0)
        self.process_button.setEnabled(False)

        # Create and start the worker thread
        self.worker = ProcessWorker(file_path, user_query, verbose)
        self.worker.progress_update.connect(self.update_progress)
        self.worker.verbose_update.connect(self.append_output)
        self.worker.finished_signal.connect(self.process_finished)
        self.worker.start()

    def update_progress(self, value: int):
        self.progress_bar.setValue(value)

    def append_output(self, message: str):
        self.output_text_edit.append(message)

    def process_finished(self, final_result: str):
        self.append_output("\n=== Final Answer ===")
        self.append_output(final_result)
        self.process_button.setEnabled(True)
        self.progress_bar.setValue(100)

# --- Main execution ---
def main():
    app = QtWidgets.QApplication(sys.argv)
    main_win = MainWindow()
    main_win.resize(800, 600)
    main_win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
