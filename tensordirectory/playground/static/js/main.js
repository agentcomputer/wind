document.addEventListener('DOMContentLoaded', () => {
    const toolSelect = document.getElementById('api-tool-select');
    const parameterFormContainer = document.getElementById('parameter-form-container');
    const mcpRequestDisplay = document.getElementById('mcp-request-display');
    const mcpResponseDisplay = document.getElementById('mcp-response-display');
    const executeToolButton = document.getElementById('execute-tool-button');
    const demoDataSelector = document.getElementById('demo-data-selector');

    const demoDataSets = {
        "upload_tensor": [
            {
                name: "Small 2D Tensor (Integers)",
                args: {
                    name: "demo_tensor_2d_int",
                    description: "A small 2D integer tensor for demo purposes.",
                    tensor_data: [[1, 2, 3], [4, 5, 6]]
                }
            },
            {
                name: "1D Float Tensor",
                args: {
                    name: "demo_tensor_1d_float",
                    description: "A 1D float tensor.",
                    tensor_data: [0.1, 0.2, 0.3, 0.4, 0.5]
                }
            }
        ],
        "upload_model": [
            {
                name: "Simple Code-Only Model",
                args: {
                    name: "demo_model_code_only",
                    description: "A simple Python model that adds 1 to the input.",
                    model_code: "import numpy as np\ndef predict(input_tensors_dict):\n  input_tensor = list(input_tensors_dict.values())[0]\n  return input_tensor + 1"
                }
            },
            {
                name: "Weights-Only Model",
                args: {
                    name: "demo_model_weights_only",
                    description: "Some example weights for a model.",
                    model_weights: [[0.1, 0.5], [-0.2, 0.8]]
                }
            }
        ],
        "list_tensors": [
            {
                name: "List with 'demo' name filter",
                args: {
                    filter_by_name_contains: "demo",
                    limit: 5,
                    offset: 0
                }
            },
            {
                name: "List all (limit 10)",
                args: {
                    limit: 10,
                    offset: 0
                }
            }
        ],
        "update_tensor_metadata": [
            {
                name: "Update 'demo_tensor_1d_float'",
                args: {
                    name_or_uuid: "demo_tensor_1d_float",
                    metadata_updates: {
                        description: "Updated description for the 1D float tensor via demo.",
                        user_name: "demo_tensor_1d_float_updated"
                    }
                }
            }
        ],
        "query_tensor_directory": [
            {
                name: "Query for 'demo_tensor'",
                args: {
                    prompt: "Find tensors with 'demo_tensor' in their name"
                }
            }
        ]
    };

    function displayDemoDataOptions(toolName) {
        demoDataSelector.innerHTML = '';
        const demosForTool = demoDataSets[toolName];

        if (demosForTool && demosForTool.length > 0) {
            const title = document.createElement('p');
            title.style.fontWeight = '500';
            title.style.marginBottom = '10px'; // Added more margin
            title.textContent = "Load Demo Data:";
            demoDataSelector.appendChild(title);

            demosForTool.forEach(scenario => {
                const button = document.createElement('button');
                button.className = 'demo-data-button';
                button.textContent = scenario.name;
                button.type = 'button';
                button.addEventListener('click', () => {
                    loadDemoDataToForm(toolName, scenario.name);
                });
                demoDataSelector.appendChild(button);
            });
        } else {
            demoDataSelector.innerHTML = '<p>No demo data sets available for this tool.</p>';
        }
    }

    function loadDemoDataToForm(toolName, scenarioName) {
        const scenario = demoDataSets[toolName]?.find(s => s.name === scenarioName);
        if (!scenario) {
            console.error(`Demo scenario "${scenarioName}" not found for tool "${toolName}".`);
            return;
        }

        const currentForm = parameterFormContainer.querySelector('form');
        if (!currentForm) {
            console.error("Cannot load demo data: Form not found.");
            return;
        }

        for (const key in scenario.args) {
            const value = scenario.args[key];
            const inputField = currentForm.elements[key];

            if (inputField) {
                // Handle NodeList for radio buttons (though not currently used)
                if (inputField instanceof NodeList) {
                    inputField.forEach(radio => {
                        if (radio.value === String(value)) {
                            radio.checked = true;
                        }
                    });
                } else if (inputField.type === 'checkbox') {
                    inputField.checked = Boolean(value);
                } else if (inputField.dataset.inputType === 'json') {
                    inputField.value = JSON.stringify(value, null, 2); // Pretty print JSON
                } else {
                    inputField.value = value;
                }
            } else {
                console.warn(`Form field with name "${key}" not found for tool "${toolName}".`);
            }
        }
    }

    fetch('/api/tools')
        .then(response => response.json())
        .then(data => {
            if (data.tools && Array.isArray(data.tools)) {
                data.tools.forEach(tool => {
                    const option = document.createElement('option');
                    option.value = tool.name;
                    option.textContent = tool.name;
                    toolSelect.appendChild(option);
                });
            } else {
                console.error('Error: Expected an array of tools, but received:', data);
                parameterFormContainer.innerHTML = '<p>Error loading tools. Check console.</p>';
            }
        })
        .catch(error => {
            console.error('Error fetching API tools:', error);
            parameterFormContainer.innerHTML = '<p>Error fetching API tools. See console for details.</p>';
        });

    toolSelect.addEventListener('change', () => {
        const selectedToolName = toolSelect.value;
        parameterFormContainer.innerHTML = '';
        demoDataSelector.innerHTML = '<p>Options for loading demo data will appear here.</p>';

        if (selectedToolName) {
            fetch(`/api/tools/${selectedToolName}/schema`)
                .then(response => response.json())
                .then(schema => {
                    if (schema.message && !schema.properties) {
                        parameterFormContainer.innerHTML = `<p>${schema.message}</p>`;
                        displayDemoDataOptions(selectedToolName);
                        return;
                    }
                    generateFormFromSchema(schema, selectedToolName);
                    displayDemoDataOptions(selectedToolName);
                })
                .catch(error => {
                    console.error(`Error fetching schema for tool ${selectedToolName}:`, error);
                    parameterFormContainer.innerHTML = `<p>Error fetching schema for ${selectedToolName}. See console.</p>`;
                    demoDataSelector.innerHTML = '';
                });
        } else {
            parameterFormContainer.innerHTML = '<p>Select an API tool to see its parameters.</p>';
        }
    });

    function generateFormFromSchema(schema, toolName) {
        const form = document.createElement('form');
        form.id = `${toolName}-form`;

        const titleText = schema.title || `${toolName.replace(/_/g, " ")} Parameters`;
        const titleElement = document.createElement('h3');
        titleElement.className = 'section-title';
        // Basic title case conversion
        titleElement.textContent = titleText.toLowerCase().replace(/\b(\w)/g, s => s.toUpperCase());
        parameterFormContainer.appendChild(titleElement);

        const properties = schema.properties || {};
        const requiredFields = schema.required || [];

        for (const paramName in properties) {
            const paramInfo = properties[paramName];
            const formGroup = document.createElement('div');
            formGroup.className = 'form-group';

            const label = document.createElement('label');
            label.setAttribute('for', `param-${paramName}`);
            let labelText = paramInfo.title || paramName.replace(/_/g, " ");
            labelText = labelText.charAt(0).toUpperCase() + labelText.slice(1);
            label.textContent = labelText;

            if (requiredFields.includes(paramName)) {
                label.classList.add('required'); // Add class for CSS styling of asterisk
            }
            formGroup.appendChild(label);

            let inputElement;
            const inputType = paramInfo.type;
            const inputId = `param-${paramName}`;

            if (paramInfo.enum) {
                inputElement = document.createElement('select');
                paramInfo.enum.forEach(enumValue => {
                    const option = document.createElement('option');
                    option.value = enumValue;
                    option.textContent = enumValue;
                    inputElement.appendChild(option);
                });
                if (paramInfo.default !== undefined) {
                    inputElement.value = paramInfo.default;
                }
            } else if (inputType === 'string') {
                const treatAsStringJson = (paramInfo.title && paramInfo.title.toLowerCase().includes('json')) ||
                                          paramName.toLowerCase().includes('json') ||
                                          paramName === 'params';

                if (paramName.includes('code') || paramName.includes('description') || paramInfo.format === 'multi-line' || treatAsStringJson) {
                    inputElement = document.createElement('textarea');
                    inputElement.rows = treatAsStringJson ? 4 : 3;
                    if (treatAsStringJson) {
                        inputElement.dataset.inputType = 'json';
                        inputElement.placeholder = 'Enter JSON string, e.g., {"key": "value"} or ["item"]';
                    }
                } else {
                    inputElement = document.createElement('input');
                    inputElement.type = 'text';
                }
            } else if (inputType === 'integer' || inputType === 'number') {
                inputElement = document.createElement('input');
                inputElement.type = 'number';
                if (inputType === 'number') inputElement.step = 'any';
            } else if (inputType === 'boolean') {
                inputElement = document.createElement('input');
                inputElement.type = 'checkbox';
                inputElement.style.width = 'auto';
                inputElement.style.marginRight = '5px';
                // Append checkbox before label text or adjust label structure for better alignment
                const labelContainer = document.createElement('div');
                labelContainer.style.display = 'flex';
                labelContainer.style.alignItems = 'center';
                labelContainer.appendChild(inputElement);
                labelContainer.appendChild(label); // Original label text goes after checkbox
                formGroup.innerHTML = ''; // Clear existing label
                formGroup.appendChild(labelContainer);

                if (paramInfo.default !== undefined) {
                    inputElement.checked = paramInfo.default;
                }
            } else if (inputType === 'array' || inputType === 'object') {
                inputElement = document.createElement('textarea');
                inputElement.rows = 4;
                inputElement.dataset.inputType = 'json';
                const helpTextContent = inputType === 'array' ? 'e.g., [[1,2],[3,4]] or ["item1"]' : 'e.g., {"key": "value"}';
                inputElement.placeholder = `Enter JSON ${inputType}, ${helpTextContent}`;

                const helpSmallText = document.createElement('small');
                helpSmallText.style.display = 'block';
                helpSmallText.style.marginTop = '4px';
                helpSmallText.textContent = `Expected format: JSON ${inputType}.`;
                formGroup.appendChild(helpSmallText);
            } else {
                inputElement = document.createElement('input');
                inputElement.type = 'text';
                inputElement.placeholder = `Type: ${inputType}`;
            }

            if (inputElement) {
                inputElement.id = inputId;
                inputElement.name = paramName;
                if (paramInfo.default !== undefined && inputElement.type !== 'checkbox' && inputElement.tagName.toLowerCase() !== 'select') {
                     inputElement.value = typeof paramInfo.default === 'object' ? JSON.stringify(paramInfo.default, null, 2) : paramInfo.default;
                }
                 if (inputElement.type !== 'checkbox') { // Checkbox already handled with its label
                    formGroup.appendChild(inputElement);
                }
            }
            form.appendChild(formGroup);
        }
        parameterFormContainer.appendChild(form);
    }

    executeToolButton.addEventListener('click', async () => {
        const toolName = toolSelect.value;
        if (!toolName) {
            alert('Please select an API tool first.');
            return;
        }

        executeToolButton.disabled = true;
        executeToolButton.textContent = 'Executing...';
        mcpResponseDisplay.classList.remove('error-message');
        mcpRequestDisplay.textContent = ""; // Clear previous request
        mcpResponseDisplay.textContent = 'Executing...';


        const args = {};
        const formContainer = document.getElementById('parameter-form-container');
        const currentForm = formContainer.querySelector('form');
        if (!currentForm) {
            alert("Could not find the parameter form.");
            executeToolButton.disabled = false;
            executeToolButton.textContent = 'Execute Tool';
            return;
        }

        const inputs = currentForm.querySelectorAll('input, textarea, select');
        let parseError = false;
        let missingRequiredField = false;

        inputs.forEach(input => {
            if (parseError || missingRequiredField) return;

            const name = input.name;
            let value = input.value;
            // Check if label has 'required' class
            let isRequired = false;
            const labelElement = currentForm.querySelector(`label[for='param-${name}']`);
            if (labelElement && labelElement.classList.contains('required')) {
                isRequired = true;
            }


            if (input.type === 'number') {
                if (value.trim() === '') {
                    if (isRequired) {
                        alert(`Required field '${name}' cannot be empty.`);
                        missingRequiredField = true; return;
                    }
                    return;
                }
                value = input.valueAsNumber;
                if (isNaN(value)) {
                    alert(`Invalid number for field: ${name}.`);
                    parseError = true; return;
                }
            } else if (input.type === 'checkbox') {
                value = input.checked;
            } else if (input.dataset.inputType === 'json') {
                if (value.trim() === '') {
                    if (isRequired) {
                        alert(`JSON input for required field '${name}' cannot be empty.`);
                        missingRequiredField = true; return;
                    }
                    return;
                }
                try {
                    value = JSON.parse(value);
                } catch (e) {
                    alert(`Invalid JSON in field '${name}': ${e.message}`);
                    parseError = true; return;
                }
            } else {
                if (value.trim() === '' && isRequired) {
                    alert(`Required field '${name}' cannot be empty.`);
                    missingRequiredField = true; return;
                }
                // For optional empty strings, we'll pass them as empty strings.
                // Pydantic models can handle Optional[str] = "" if needed.
                // If the field is not required and empty, it will be sent as an empty string.
            }

            args[name] = value;
        });

        if (parseError || missingRequiredField) {
            executeToolButton.disabled = false;
            executeToolButton.textContent = 'Execute Tool';
            mcpResponseDisplay.textContent = 'Form validation error. Please check fields.';
            mcpResponseDisplay.classList.add('error-message');
            return;
        }

        const requestPayload = { tool_name: toolName, args: args };
        mcpRequestDisplay.textContent = JSON.stringify(requestPayload, null, 2);

        try {
            const response = await fetch('/api/execute_tool', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(requestPayload),
            });

            const resultText = await response.text();
            let resultJson;
            try { resultJson = JSON.parse(resultText); }
            catch (e) {
                mcpResponseDisplay.textContent = `Error parsing JSON response: ${e.message}\nRaw Response:\n${resultText}`;
                mcpResponseDisplay.classList.add('error-message');
                return;
            }

            if (!response.ok) {
                let errorDisplay = `HTTP Error ${response.status}: ${response.statusText}\n`;
                if (resultJson.detail) {
                    if (typeof resultJson.detail === 'string') errorDisplay += resultJson.detail;
                    else if (resultJson.detail.errors && Array.isArray(resultJson.detail.errors)) errorDisplay += `Validation Error:\n${JSON.stringify(resultJson.detail.errors, null, 2)}`;
                    else if (resultJson.detail.error_type && resultJson.detail.message) errorDisplay += `${resultJson.detail.error_type}: ${resultJson.detail.message}`;
                    else errorDisplay += JSON.stringify(resultJson.detail, null, 2);
                } else errorDisplay += JSON.stringify(resultJson, null, 2);
                mcpResponseDisplay.textContent = errorDisplay;
                mcpResponseDisplay.classList.add('error-message');
            } else {
                 mcpResponseDisplay.textContent = JSON.stringify(resultJson.result !== undefined ? resultJson.result : resultJson, null, 2);
            }
        } catch (error) {
            mcpResponseDisplay.textContent = `Network or other client-side error: ${error.message}`;
            mcpResponseDisplay.classList.add('error-message');
            console.error('Error executing tool:', error);
        } finally {
            executeToolButton.disabled = false;
            executeToolButton.textContent = 'Execute Tool';
        }
    });
});
