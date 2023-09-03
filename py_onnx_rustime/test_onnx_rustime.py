# DEMO USAGE OF THE RUST BINDING FOR ONNX RUNTIME

# This script showcases how to use the Rust binding for the ONNX runtime.
# It's designed to provide a user-friendly interface for loading and running ONNX models and viewing their results.

# The Rust binding abstracts the complexities and details of the underlying ONNX structures by exposing them as simple IDs.
# For example, when an ONNX model or data is loaded, the binding returns an ID.
# This ID acts like an opaque pointer - it hides the actual data structure behind it.
# Therefore, all the ONNX structures remain private to the Rust implementation, and Python simply uses these IDs for further interactions.

# This abstraction offers several benefits:
# - Simplifies the Python interface by reducing complex structures to simple integers.
# - Ensures that Python doesn't directly manipulate the underlying ONNX structures, providing an added layer of safety and encapsulation.
# - Makes the Rust-Python interplay more efficient, as passing simple IDs is faster and less error-prone than transferring complex structures.

# PRE-REQUISITES:
# Before running this script, ensure that you've activated the appropriate Python environment. Depending on your platform:

# - On MacOS / Linux:
#     ```
#     source py_onnx_rustime/rust-binding/bin/activate
#     ```

# - On Windows:
#     ```
#     py_onnx_rustime\rust-binding\Scripts\activate
#     ```

# To use this demo:
# 1. Ensure that the Rust binding (`onnx_rustime_project`) is correctly installed and available in the Python environment.
# 2. Run this script.
# 3. Follow the interactive prompts to load an ONNX model, run it, and see the results.

# Note: This demo assumes the presence of specific ONNX models and data files at predefined paths.
# Ensure these paths and files are available for successful execution.


import onnx_rustime_lib as onnx_rustime

# Define the paths for each model, input, and expected output
MODEL_DATA = [
    {
        "name": "bvlcalexnet-12",
        "model_path": "models/bvlcalexnet-12/bvlcalexnet-12.onnx",
        "input_path": "models/bvlcalexnet-12/test_data_set_0/input_0.pb",
        "output_path": "models/bvlcalexnet-12/test_data_set_0/output_0.pb",
    },
    {
        "name": "caffenet-12",
        "model_path": "models/caffenet-12/caffenet-12.onnx",
        "input_path": "models/caffenet-12/test_data_set_0/input_0.pb",
        "output_path": "models/caffenet-12/test_data_set_0/output_0.pb",
    },
    {
        "name": "mnist-8",
        "model_path": "models/mnist-8/mnist-8.onnx",
        "input_path": "models/mnist-8/test_data_set_0/input_0.pb",
        "output_path": "models/mnist-8/test_data_set_0/output_0.pb",
    },
    {
        "name": "resnet152-v2-7",
        "model_path": "models/resnet152-v2-7/resnet152-v2-7.onnx",
        "input_path": "models/resnet152-v2-7/test_data_set_0/input_0.pb",
        "output_path": "models/resnet152-v2-7/test_data_set_0/output_0.pb",
    },
    {
        "name": "squeezenet1.0-12",
        "model_path": "models/squeezenet1.0-12/squeezenet1.0-12.onnx",
        "input_path": "models/squeezenet1.0-12/test_data_set_0/input_0.pb",
        "output_path": "models/squeezenet1.0-12/test_data_set_0/output_0.pb",
    },
    {
        "name": "zfnet512-12",
        "model_path": "models/zfnet512-12/zfnet512-12.onnx",
        "input_path": "models/zfnet512-12/test_data_set_0/input_0.pb",
        "output_path": "models/zfnet512-12/test_data_set_0/output_0.pb",
    },
]


def display_menu(selected_index=None):
    """Display the model selection menu with the selected item highlighted."""
    print("Select a model:\n")
    for idx, model_data in enumerate(MODEL_DATA, 1):
        prefix = "=> " if idx == selected_index else "   "
        print(f"{prefix}[{idx}] {model_data['name']}")


def select_model():
    """Display a menu to let the user select a model and return the paths."""
    while True:
        display_menu()
        try:
            choice = int(input("\nEnter your choice (number): "))
            if 1 <= choice <= len(MODEL_DATA):
                return (
                    MODEL_DATA[choice - 1]["model_path"],
                    MODEL_DATA[choice - 1]["input_path"],
                    MODEL_DATA[choice - 1]["output_path"],
                )
            else:
                print("Invalid choice! Please select a valid number.")
        except ValueError:
            print("Please enter a valid number.\n")


def select_verbose():
    """Let the user choose if they want the verbose mode or not."""
    while True:
        choice = input("\nRun in verbose mode? (yes/no): ").strip().lower()
        if choice == "yes":
            return True
        elif choice == "no":
            return False
        else:
            print("Invalid choice! Please select 'yes' or 'no'.")


def main():
    # Get the model, input, and expected output paths from the menu
    model_path, input_path, exp_output_path = select_model()

    verbose = select_verbose()

    if model_path and input_path and exp_output_path:
        model = onnx_rustime.py_load_model(model_path)
        print(f"\nLoaded model with ID: {model}")

        input = onnx_rustime.py_load_data(input_path)
        print(f"\nLoaded input with ID: {input}")

        expected_output = onnx_rustime.py_load_data(exp_output_path)
        print(f"\nLoaded input with ID: {expected_output}")

        # onnx_rustime.py_print_data(input)
        # onnx_rustime.py_print_data(expected_output)

        predicted_output = onnx_rustime.py_run(model, input, verbose)

        onnx_rustime.py_display_outputs(predicted_output, expected_output)

    else:
        print("Exiting program due to invalid choice.")


if __name__ == "__main__":
    main()
