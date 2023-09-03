const onnx_rustime = require('.')
const readline = require('readline');


// Define the model data
const MODEL_DATA = [
    {
        name: 'bvlcalexnet-12',
        model_path: 'models/bvlcalexnet-12/bvlcalexnet-12.onnx',
        input_path: 'models/bvlcalexnet-12/test_data_set_0/input_0.pb',
        output_path: 'models/bvlcalexnet-12/test_data_set_0/output_0.pb',
    },
    {
        name: 'caffenet-12',
        model_path: 'models/caffenet-12/caffenet-12.onnx',
        input_path: 'models/caffenet-12/test_data_set_0/input_0.pb',
        output_path: 'models/caffenet-12/test_data_set_0/output_0.pb',
    },
    {
        name: 'mnist-8',
        model_path: 'models/mnist-8/mnist-8.onnx',
        input_path: 'models/mnist-8/test_data_set_0/input_0.pb',
        output_path: 'models/mnist-8/test_data_set_0/output_0.pb',
    },
    {
        name: 'resnet152-v2-7',
        model_path: 'models/resnet152-v2-7/resnet152-v2-7.onnx',
        input_path: 'models/resnet152-v2-7/test_data_set_0/input_0.pb',
        output_path: 'models/resnet152-v2-7/test_data_set_0/output_0.pb',
    },
    {
        name: 'squeezenet1.0-12',
        model_path: 'models/squeezenet1.0-12/squeezenet1.0-12.onnx',
        input_path: 'models/squeezenet1.0-12/test_data_set_0/input_0.pb',
        output_path: 'models/squeezenet1.0-12/test_data_set_0/output_0.pb',
    },
    {
        name: 'zfnet512-12',
        model_path: 'models/zfnet512-12/zfnet512-12.onnx',
        input_path: 'models/zfnet512-12/test_data_set_0/input_0.pb',
        output_path: 'models/zfnet512-12/test_data_set_0/output_0.pb',
    },
];

// Create an interface for reading input from the command line
const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout
});

// Display the model selection menu with the selected item highlighted
function displayMenu(selectedIndex) {
    console.log('Select a model:\n');
    MODEL_DATA.forEach((modelData, idx) => {
        const prefix = idx + 1 === selectedIndex ? '=> ' : '   ';
        console.log(`${prefix}[${idx + 1}] ${modelData.name}`);
    });
}

// Let the user choose if they want to run in verbose mode
function selectVerbose() {
    return new Promise((resolve) => {
        rl.question('\nRun in verbose mode? (yes/no): ', (answer) => {
            if (answer.trim().toLowerCase() === 'yes') {
                resolve(true);
            } else {
                resolve(false);
            }
        });
    });
}

// Main function
async function main() {
    // Get the model, input, and expected output paths from the menu
    displayMenu();
    const choice = await new Promise((resolve) => {
        rl.question('\nEnter your choice (number): ', (answer) => {
            const numericChoice = parseInt(answer);
            if (!isNaN(numericChoice) && numericChoice >= 1 && numericChoice <= MODEL_DATA.length) {
                resolve(numericChoice);
            } else {
                console.log('Invalid choice! Please select a valid number.');
                resolve(null);
            }
        });
    });

    if (choice !== null) {
        const selectedModel = MODEL_DATA[choice - 1];
        const model = onnx_rustime.js_load_model(selectedModel.model_path);
        console.log(`\nLoaded model with ID: ${model}`);

        const input = onnx_rustime.js_load_data(selectedModel.input_path);
        console.log(`\nLoaded input with ID: ${input}`);

        const expectedOutput = onnx_rustime.js_load_data(selectedModel.output_path);
        console.log(`\nLoaded input with ID: ${expectedOutput}`);

        const verbose = await selectVerbose();
        const predictedOutput = onnx_rustime.js_run(model, input, verbose);

        onnx_rustime.js_display_outputs(predictedOutput, expectedOutput);

        rl.close();
    } else {
        rl.close();
        console.log('Exiting program due to invalid choice.');
    }
}

main();
