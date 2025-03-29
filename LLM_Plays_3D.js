import * as THREE from 'https://cdnjs.cloudflare.com/ajax/libs/three.js/0.164.1/three.module.min.js';
import * as webllm from "https://esm.run/@mlc-ai/web-llm";

// --- Configuration ---
const PLAYER_SPEED = 5.0;
const PLAYER_ROTATION_SPEED = 2.0;
const MOUSE_SENSITIVITY = 0.002;
const INITIAL_MODEL = "Qwen2.5-Coder-7B-Instruct-q4f32_1-MLC"; // or "Qwen2.5-Coder-3B-Instruct-q4f32_1-MLC"
const TOOL_RETRY_LIMIT = 2;
const SYSTEM_PROMPT = "You are a helpful AI assistant controlling a 3D world in a game. You can interact with the user through chat and use available tools to modify the world. Be descriptive and engaging.";

// --- Global Variables ---
let scene, camera, renderer, playerAvatar, clock, controlsState;
let engine;
let currentModel = INITIAL_MODEL;
let temperature = 0.7;
let chatHistory = []; // Stores conversation for LLM context
let availableTools = {}; // Stores LLM tools { name: { description, function } }
let isLLMThinking = false;
let llmRetryCount = 0;
let activeGamepad = null;
let joystickMove = null;
let joystickRotate = null;
let mouseDrag = { x: 0, y: 0, active: false };

// Sound
let audioContext;
let sounds = {};

// DOM Elements
const loadingScreen = document.getElementById('loading-screen');
const loadingText = document.getElementById('loading-text');
const loadingProgress = document.getElementById('loading-progress');
const loadingDetails = document.getElementById('loading-details');
const gameCanvas = document.getElementById('game-canvas');
const chatHistoryDiv = document.getElementById('chat-history');
const chatInput = document.getElementById('chat-input');
const sendButton = document.getElementById('send-button');
const modelSelect = document.getElementById('model-select');
const reloadButton = document.getElementById('reload-button');
const temperatureSlider = document.getElementById('temperature-slider');
const temperatureValue = document.getElementById('temperature-value');
const llmStatusDiv = document.getElementById('llm-status');
const joystickMoveZone = document.getElementById('joystick-move-zone');
const joystickRotateZone = document.getElementById('joystick-rotate-zone');


// --- Initialization ---
function init() {
    if (!navigator.gpu) {
         showFatalError("WebGPU is not supported on this browser. Cannot run WebLLM.");
         return;
    }
    initSound();
    setupThreeJS();
    setupControls();
    setupUI();
    loadLLM(); // Start loading the LLM
    animate();
    console.log("Game Initialized");
}

function showFatalError(message) {
    loadingText.textContent = "Error";
    loadingDetails.textContent = message;
    loadingProgress.style.width = '100%';
    loadingProgress.style.backgroundColor = 'red';
     // Hide UI elements that shouldn't be interactive
    document.getElementById('ui-container').style.display = 'none';
    document.getElementById('game-container').style.display = 'none';
}

// --- WebLLM ---
async function loadLLM() {
    try {
        isLLMThinking = true;
        updateLLMStatus('Loading Model...');
        loadingScreen.style.display = 'flex';
        loadingText.textContent = `Loading ${modelSelect.options[modelSelect.selectedIndex].text}...`;
        loadingProgress.style.width = '0%';
        loadingDetails.textContent = '';
        chatInput.disabled = true;
        sendButton.disabled = true;
        reloadButton.disabled = true;

        if (engine) {
            console.log("Unloading previous engine...");
            await engine.unload();
            engine = null;
            console.log("Previous engine unloaded.");
        }
         // Clear chat history when changing models? Optional.
        // chatHistory = [];
        // updateChatDisplay();

        console.log(`Loading model: ${currentModel}`);
        engine = await webllm.CreateMLCEngine(
            currentModel,
            { initProgressCallback: reportProgress }
        );
        console.log("MLC Engine created:", engine);

        loadingScreen.style.display = 'none';
        chatInput.disabled = false;
        sendButton.disabled = false;
        reloadButton.disabled = false;
        isLLMThinking = false;
        updateLLMStatus('Idle');
        chatHistory = []; // Reset history for the new model
        addMessageToChat("llm", `Hello! I'm ready to play. I'm currently running the ${modelSelect.options[modelSelect.selectedIndex].text} model.`);

        // Define initial tools (including the tool creator)
        setupInitialTools();

    } catch (error) {
        console.error("Error loading LLM:", error);
        showFatalError(`Failed to load model ${currentModel}. Error: ${error.message}. Check console for details.`);
        updateLLMStatus('Error', true);
        isLLMThinking = false; // Ensure thinking state is reset
    }
}

function reportProgress(report) {
    // Find the progress percentage
    const progressPercentage = report.progress ? (report.progress * 100) : 0;
    loadingProgress.style.width = `${progressPercentage.toFixed(2)}%`;
    loadingDetails.textContent = report.text || '';
    // console.log("Loading progress:", report);
}

async function sendMessageToLLM(messageText) {
    if (!engine || isLLMThinking) return;

    isLLMThinking = true;
    llmRetryCount = 0; // Reset retry count for new message
    updateLLMStatus('Thinking...');
    chatInput.disabled = true;
    sendButton.disabled = true;
    playSound('llm_start'); // Sound for LLM start

    // Add user message to history and display
    addMessageToChat("user", messageText);
    chatHistory.push({ role: "user", content: messageText });

    await generateLLMResponse();
}

async function generateLLMResponse() {
    if (!engine) {
        console.error("Engine not available for generation.");
        addMessageToChat("error", "LLM Engine is not loaded or has failed.");
        resetLLMState();
        return;
    }

    try {
        const messages = buildLLMPrompt();
        console.log("Sending messages to LLM:", JSON.stringify(messages, null, 2));

        const reply = await engine.chat.completions.create({
            messages: messages,
            temperature: temperature,
            stream: false // Required for custom tool parsing logic
        });

        console.log("LLM Raw Response:", reply);

        if (!reply.choices || reply.choices.length === 0 || !reply.choices[0].message) {
             throw new Error("Invalid response structure from LLM.");
        }

        const llmMessageContent = reply.choices[0].message.content || "";
        handleLLMResponse(llmMessageContent);
        llmRetryCount = 0; // Reset retry count on success

    } catch (error) {
        console.error(`LLM generation error (Attempt ${llmRetryCount + 1}/${TOOL_RETRY_LIMIT + 1}):`, error);
        llmRetryCount++;
        if (llmRetryCount <= TOOL_RETRY_LIMIT) {
            addMessageToChat("error", `LLM generation failed, retrying... (${llmRetryCount}/${TOOL_RETRY_LIMIT})`);
            updateLLMStatus('Retrying...');
            // Optional: Add a small delay before retrying
            await new Promise(resolve => setTimeout(resolve, 500));
            await generateLLMResponse(); // Retry
        } else {
            addMessageToChat("error", `LLM generation failed after ${TOOL_RETRY_LIMIT} retries. Error: ${error.message}`);
            resetLLMState();
            playSound('error');
        }
    }
}

function handleLLMResponse(responseText) {
    console.log("Handling LLM Response Text:", responseText);
    let textualResponse = responseText;
    let toolCall = null;

    // --- Custom Tool Call Parsing ---
    // Look for ```json ... ``` block
    const jsonBlockRegex = /```json\s*([\s\S]*?)\s*```/;
    const match = responseText.match(jsonBlockRegex);
    console.log("match:", JSON.stringify(match));

    if (match && match[1]) {
        const jsonString = match[1].trim();
        // Remove the JSON block from the textual response displayed to the user
        textualResponse = responseText.replace(jsonBlockRegex, "").trim();
        try {
            toolCall = JSON.parse(jsonString);
            console.log("Parsed Tool Call:", toolCall);
            if (!toolCall.tool_name || !toolCall.arguments) {
                throw new Error("Tool call JSON must have 'tool_name' and 'arguments'.");
            }
        } catch (error) {
            console.error("Failed to parse tool call JSON:", error);
            addMessageToChat("error", `LLM tried to call a tool with invalid JSON format: ${error.message}. Raw JSON: ${jsonString}`);
            toolCall = null; // Invalidate the tool call
            // Keep the original textual response including the malformed block for context
            textualResponse = responseText;
        }
    }

    // Display LLM's textual response (if any)
    if (textualResponse) {
        addMessageToChat("llm", textualResponse);
        chatHistory.push({ role: "assistant", content: textualResponse }); // Add text part to history
    }

    // Execute tool call if valid
    if (toolCall) {
        executeToolCall(toolCall.tool_name, toolCall.arguments);
    } else {
        // If no tool call, or if parsing failed but there was text, we are done for this turn
         if (!textualResponse) { // Handle case where LLM ONLY outputs a (malformed) tool block
             addMessageToChat("llm", "[LLM attempted an action but formatting was incorrect]");
             chatHistory.push({ role: "assistant", content: "[LLM attempted an action but formatting was incorrect]" });
         }
        resetLLMState(); // Ready for next user input
    }
     playSound('llm_end');
}

function resetLLMState() {
    isLLMThinking = false;
    chatInput.disabled = false;
    sendButton.disabled = false;
    updateLLMStatus('Idle');
    chatInput.focus();
}

function buildLLMPrompt() {
    const messages = [];

    // 1. System Prompt (Only if chatHistory is empty)
    if (chatHistory.length === 0 && SYSTEM_PROMPT) {
        messages.push({ role: "system", content: SYSTEM_PROMPT });
    }

    // 2. Dynamic Assistant Prompt for Tools (Add THIS time, but NOT to history)
    let toolDescriptions = "You have the following tools available:\n";
    if (Object.keys(availableTools).length === 0) {
        toolDescriptions += "- None currently defined.\n";
    } else {
        for (const toolName in availableTools) {
            const tool = availableTools[toolName];
            // Describe parameters if possible (simplified here)
             const paramsDesc = tool.parameters ? ` Parameters: ${JSON.stringify(tool.parameters)}` : "";
             toolDescriptions += `- ${toolName}: ${tool.description}${paramsDesc}\n`;
        }
    }
    toolDescriptions += "\nTo use a tool, output a JSON block like this, along with any normal text:\n\`\`\`json\n{\n  \"tool_name\": \"<name_of_tool>\",\n  \"arguments\": { <arguments_object> }\n}\n\`\`\`";

    messages.push({ role: "assistant", content: toolDescriptions });

    // 3. Conversation History
    messages.push(...chatHistory);

    return messages;
}


// --- Tool Handling ---

function setupInitialTools() {
    availableTools = {}; // Clear existing tools

    // Tool Creation Tool
    availableTools['tool_creation_tool'] = {
        description: "Creates a new tool that you can use later. Provide a 'name' for the new tool and a 'description' of what it should do and what parameters it needs (as properties of a single object argument). The description should be clear enough to generate JavaScript code for the tool's function.",
        parameters: { // Parameters *for the tool_creation_tool itself*
             type: "object",
             properties: {
                 name: { type: "string", description: "The name for the new tool." },
                 description: { type: "string", description: "Detailed description of the new tool's function and its parameters." }
             },
             required: ["name", "description"]
         },
        function: createNewTool
    };

    // Example World Interaction Tool
    availableTools['create_object'] = {
        description: "Creates a simple geometric object in the 3D world.",
        parameters: {
             type: "object",
             properties: {
                 shape: { type: "string", description: "Shape of the object (e.g., 'cube', 'sphere').", enum: ["cube", "sphere", "cylinder", "cone"] },
                 position: { type: "object", properties: { x: {type: "number"}, y: {type: "number"}, z: {type: "number"} }, required: ["x", "y", "z"], description: "World coordinates {x, y, z}." },
                 size: { type: "number", description: "Approximate size of the object (e.g., 1). Default 1.", default: 1},
                 color: { type: "string", description: "Color of the object (e.g., 'red', '#00ff00'). Default 'gray'.", default: "gray"}
             },
             required: ["shape", "position"]
        },
        function: tool_createObject
    };

     availableTools['change_ground_color'] = {
         description: "Changes the color of the ground plane.",
         parameters: {
             type: "object",
             properties: {
                 color: { type: "string", description: "The new color for the ground (e.g., 'green', '#ff00ff')." }
             },
             required: ["color"]
         },
         function: tool_changeGroundColor
     };

     availableTools['list_objects'] = {
         description: "Lists the names and types of objects currently in the scene (excluding player and ground).",
         parameters: {}, // No parameters needed
         function: tool_listObjects
     };

    console.log("Initial tools set up:", Object.keys(availableTools));
}

async function createNewTool(args) {
     // Note: 'args' comes directly from the LLM's parsed JSON arguments
    const { name, description } = args;

    if (!name || !description) {
        return "Error: Tool creation requires both 'name' and 'description'.";
    }
    if (availableTools[name]) {
         return `Error: A tool with the name '${name}' already exists.`;
    }
    if (!/^[a-zA-Z_][a-zA-Z0-9_]*$/.test(name)) {
         return `Error: Tool name '${name}' is invalid. Use letters, numbers, and underscores, starting with a letter or underscore.`;
    }

    addMessageToChat("tool-call", `Attempting to create new tool: '${name}'...`);
    updateLLMStatus('Generating Tool...'); // Indicate sub-task

    try {
        // Prompt the Coder LLM to generate JUST the function body
        const codeGenPrompt = `Generate only the JavaScript code for the body of a function.
The function should be named '${name}'.
It must accept a single object argument named 'params'.
The function should perform the following action based on the description: ${description}.
The function should interact with the Three.js scene variable 'scene' and potentially 'playerAvatar' if needed. Use standard Three.js (r140+) methods.
The function *must* return a string indicating success or failure (e.g., "Object moved successfully." or "Error: Object not found.").
Only output the raw JavaScript code inside a \`\`\`javascript ... \`\`\` block. Do not include the function signature itself (e.g. function ${name}(params) { ... }), just the code inside the curly braces {}.

Example Description: "Moves an object named 'targetObjectName' found in 'params' to the position specified in 'params.targetPosition' {x, y, z}."
Example Output:
\`\`\`javascript
const objectName = params.targetObjectName;
const position = params.targetPosition;
if (!objectName || !position) return "Error: Missing required parameters 'targetObjectName' or 'targetPosition'.";
const object = scene.getObjectByName(objectName);
if (!object) return \`Error: Object named '\${objectName}' not found.\`;
try {
    object.position.set(position.x, position.y, position.z);
    return \`Object '\${objectName}' moved successfully to \${JSON.stringify(position)}.\`;
} catch (e) {
    console.error("Error moving object:", e);
    return \`Error moving object '\${objectName}': \${e.message}\`;
}
\`\`\``;


        const codeGenMessages = [{ role: "user", content: codeGenPrompt }];
        console.log("Sending code generation prompt:", codeGenMessages);

        // Use a slightly higher temperature for creative coding? Or lower for precision? Let's keep it standard for now.
        const codeReply = await engine.chat.completions.create({
            messages: codeGenMessages,
            temperature: 0.5, // Lower temp for code generation consistency
            stream: false
        });

        console.log("Code Generation Raw Reply:", codeReply);
        if (!codeReply.choices || codeReply.choices.length === 0 || !codeReply.choices[0].message || !codeReply.choices[0].message.content) {
             throw new Error("LLM did not return valid code content.");
        }

        const codeContent = codeReply.choices[0].message.content;
        const codeBlockRegex = /```javascript\s*([\s\S]*?)\s*```/;
        const codeMatch = codeContent.match(codeBlockRegex);

        if (!codeMatch || !codeMatch[1]) {
            console.error("Could not extract JS code block from LLM response:", codeContent);
            throw new Error("LLM response did not contain the expected JavaScript code block. Response: "+ codeContent.substring(0, 200) + "...");
        }

        const functionBody = codeMatch[1].trim();
        console.log(`Generated function body for '${name}':\n`, functionBody);

        // Create the actual function using the Function constructor (use with caution!)
        // We make `scene`, `THREE`, `playerAvatar`, `findObjectByNameCaseInsensitive`, `playSound` available in the function's scope.
        const newToolFunction = new Function('params', 'scene', 'THREE', 'playerAvatar', 'findObjectByNameCaseInsensitive', 'playSound', functionBody);

        // Add the new tool to our available tools list
        // We don't explicitly parse parameter definitions here, relying on the description
        availableTools[name] = {
            description: description, // Store the original description
             parameters: { type: "object", description: "Parameters defined by the tool description." }, // Generic parameter placeholder
            function: (args) => {
                try {
                    // Execute the newly created function, passing necessary context
                    const result = newToolFunction(args, scene, THREE, playerAvatar, findObjectByNameCaseInsensitive, playSound);
                    // Ensure the function returns a string as required by the prompt
                    if (typeof result !== 'string') {
                        console.warn(`Tool '${name}' did not return a string. Returning generic success message.`);
                        return `Tool '${name}' executed.`;
                    }
                    return result;
                } catch (e) {
                    console.error(`Error executing dynamically created tool '${name}':`, e);
                    return `Error executing tool '${name}': ${e.message}`;
                }
            }
        };

        console.log(`Tool '${name}' created successfully.`);
        updateLLMStatus('Idle'); // Back to idle after sub-task
        return `Tool '${name}' created successfully! You can now use it.`;

    } catch (error) {
        console.error("Error during tool creation process:", error);
        updateLLMStatus('Error', true); // Show error status briefly
        await new Promise(resolve => setTimeout(resolve, 1500)); // Keep error visible
        updateLLMStatus('Idle');
        return `Failed to create tool '${name}'. Error: ${error.message}`;
    }
}


function executeToolCall(toolName, args) {
    addMessageToChat("tool-call", `Executing tool: ${toolName} with args: ${JSON.stringify(args)}`);
    playSound('tool_use');

    if (availableTools[toolName]) {
        const tool = availableTools[toolName];
        try {
            // Execute the tool's function
            const result = tool.function(args); // Assuming sync execution for now

            // Display the result of the tool execution
            addMessageToChat("tool-call", `Tool Result: ${result}`);

            // **Crucially, do NOT add tool execution messages/results to chatHistory**
            // The LLM will know the tools are available next turn via the dynamic prompt.
            // It doesn't need the execution log in its context unless specifically designed for it.

        } catch (error) {
            console.error(`Error executing tool ${toolName}:`, error);
            addMessageToChat("error", `Error during execution of tool '${toolName}': ${error.message}`);
        }
    } else {
        console.warn(`LLM tried to call unknown tool: ${toolName}`);
        addMessageToChat("error", `Unknown tool called: '${toolName}'. Available tools: ${Object.keys(availableTools).join(', ')}`);
    }

    // After processing tool call (success or failure), reset state for next user input
    resetLLMState();
}


// --- Basic Tool Implementations ---

// Helper to find objects ignoring case, as LLM might not get case right
function findObjectByNameCaseInsensitive(name) {
    let foundObject = null;
    scene.traverse((object) => {
        // Check if object has a name and if it matches case-insensitively
        // Also exclude the Player and the Ground unless specifically named 'player' or 'ground'
        if (object.name && object.name.toLowerCase() === name.toLowerCase()) {
             // Basic check to avoid modifying player/ground accidentally unless named explicitly
             if (object !== playerAvatar && object.userData.isGround !== true) {
                 foundObject = object;
             } else if ( (name.toLowerCase() === 'player' && object === playerAvatar) ||
                         (name.toLowerCase() === 'ground' && object.userData.isGround === true) ) {
                 foundObject = object; // Allow targeting player/ground if named explicitly
             }
        }
    });
    return foundObject;
}


function tool_createObject(params) {
    try {
        const shape = params.shape?.toLowerCase() || 'cube';
        const pos = params.position;
        const size = Math.max(0.1, params.size || 1); // Ensure minimum size
        const color = params.color || 'gray';

        if (!pos || typeof pos.x !== 'number' || typeof pos.y !== 'number' || typeof pos.z !== 'number') {
            return "Error: Invalid or missing 'position' object with x, y, z coordinates.";
        }

        let geometry;
        switch (shape) {
            case 'sphere':
                geometry = new THREE.SphereGeometry(size / 2, 32, 16);
                break;
            case 'cylinder':
                 geometry = new THREE.CylinderGeometry(size / 2, size / 2, size, 32);
                 break;
             case 'cone':
                 geometry = new THREE.ConeGeometry(size / 2, size, 32);
                 break;
            case 'cube':
            default:
                geometry = new THREE.BoxGeometry(size, size, size);
        }

        const material = new THREE.MeshStandardMaterial({ color: color });
        const object = new THREE.Mesh(geometry, material);

        object.position.set(pos.x, pos.y, pos.z);

         // Assign a unique-ish name for potential future reference
        const objectBaseName = `${shape}_${color.replace('#','')}`;
        let objectName = objectBaseName;
        let counter = 1;
        while (scene.getObjectByName(objectName)) {
            counter++;
            objectName = `${objectBaseName}_${counter}`;
        }
        object.name = objectName;
        object.castShadow = true;
        object.receiveShadow = true;


        scene.add(object);
        playSound('create');
        return `Object '${object.name}' (${shape}) created successfully at ${JSON.stringify(pos)}.`;

    } catch (error) {
        console.error("Error in tool_createObject:", error);
        return `Error creating object: ${error.message}`;
    }
}

function tool_changeGroundColor(params) {
     try {
         const color = params.color;
         if (!color) return "Error: 'color' parameter is required.";

         const ground = scene.getObjectByProperty('userData', { isGround: true });
         if (ground && ground.material) {
             ground.material.color.set(color);
             playSound('modify');
             return `Ground color changed to ${color}.`;
         } else {
             return "Error: Could not find the ground object.";
         }
     } catch (error) {
         console.error("Error changing ground color:", error);
          // Attempt to provide a more specific error if color conversion failed
         if (error instanceof TypeError && error.message.includes('convert parameter to Color')) {
             return `Error: Invalid color value "${params.color}". Please use standard color names or hex codes (e.g., 'blue', '#00ff00').`;
         }
         return `Error changing ground color: ${error.message}`;
     }
}

function tool_listObjects(params) {
     let objectList = [];
     scene.traverse((object) => {
         // List named Mesh objects that are not the player or the ground
         if (object.isMesh && object.name && object !== playerAvatar && !object.userData.isGround) {
             let type = object.geometry?.type?.replace('Geometry', '') || 'Unknown';
             objectList.push(`- ${object.name} (Type: ${type})`);
         }
     });

     if (objectList.length === 0) {
         return "There are no user-created objects currently in the scene.";
     } else {
         return "Objects in the scene:\n" + objectList.join('\n');
     }
}


// --- Three.js Setup ---
function setupThreeJS() {
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x33334d); // Dark blueish background
    scene.fog = new THREE.Fog(0x33334d, 10, 50); // Add fog

    camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    camera.position.set(0, 2, 5); // Initial camera position relative to origin

    renderer = new THREE.WebGLRenderer({ canvas: gameCanvas, antialias: true });
    renderer.setSize(gameCanvas.clientWidth, gameCanvas.clientHeight); // Use canvas size
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap; // Softer shadows


    // Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6); // Soft ambient light
    scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 1.0); // Sun light
    directionalLight.position.set(5, 10, 7);
    directionalLight.castShadow = true;
    // Configure shadow properties for better quality/performance trade-off
     directionalLight.shadow.mapSize.width = 1024;
     directionalLight.shadow.mapSize.height = 1024;
     directionalLight.shadow.camera.near = 0.5;
     directionalLight.shadow.camera.far = 50;
     directionalLight.shadow.camera.left = -20;
     directionalLight.shadow.camera.right = 20;
     directionalLight.shadow.camera.top = 20;
     directionalLight.shadow.camera.bottom = -20;

    scene.add(directionalLight);
    // const lightHelper = new THREE.DirectionalLightHelper(directionalLight, 1); // Optional: Visualize light
    // scene.add(lightHelper);
    // const shadowHelper = new THREE.CameraHelper(directionalLight.shadow.camera); // Optional: Visualize shadow frustum
    // scene.add(shadowHelper);


    // Ground Plane
    const groundGeometry = new THREE.PlaneGeometry(100, 100);
    const groundMaterial = new THREE.MeshStandardMaterial({ color: 0x556b2f, side: THREE.DoubleSide }); // Dark Olive Green
    const ground = new THREE.Mesh(groundGeometry, groundMaterial);
    ground.rotation.x = -Math.PI / 2; // Rotate flat
    ground.receiveShadow = true;
    ground.name = "ground";
    ground.userData.isGround = true; // Mark as ground
    scene.add(ground);

    // Player Avatar (Simple Capsule)
    const radius = 0.4;
    const height = 1.0; // Height of the cylindrical part
    const playerGeometry = new THREE.CapsuleGeometry(radius, height, 4, 10); // Use CapsuleGeometry
    const playerMaterial = new THREE.MeshStandardMaterial({ color: 0xff4500 }); // OrangeRed
    playerAvatar = new THREE.Mesh(playerGeometry, playerMaterial);
    playerAvatar.position.set(0, radius + height/2, 0); // Position feet slightly above ground
    playerAvatar.castShadow = true;
    playerAvatar.name = "player"; // Name for potential targeting
    scene.add(playerAvatar);

    camera.lookAt(playerAvatar.position); // Look at the player initially

    clock = new THREE.Clock();

    window.addEventListener('resize', onWindowResize, false);
    onWindowResize(); // Call once initially to set size
}

function onWindowResize() {
    const container = document.getElementById('game-container');
    if (!container) return;
    const width = container.clientWidth;
    const height = container.clientHeight;

    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    renderer.setSize(width, height);
    console.log(`Resized to ${width}x${height}`);
}

// --- Controls Setup ---
function setupControls() {
    controlsState = {
        forward: 0, // -1, 0, 1
        backward: 0,
        left: 0,
        right: 0,
        rotateLeft: 0,
        rotateRight: 0,
        moveX: 0, // For joystick/analog input (-1 to 1)
        moveZ: 0, // For joystick/analog input (-1 to 1)
        rotateY: 0, // For joystick/analog input (-1 to 1)
    };

    // Keyboard
    window.addEventListener('keydown', (event) => {
        if (document.activeElement === chatInput) return; // Don't capture keys if typing in chat
        switch (event.key.toLowerCase()) {
            case 'w': case 'arrowup': controlsState.forward = 1; break;
            case 's': case 'arrowdown': controlsState.backward = 1; break;
            case 'a': case 'arrowleft': controlsState.left = 1; break;
            case 'd': case 'arrowright': controlsState.right = 1; break;
            case 'q': controlsState.rotateLeft = 1; break;
            case 'e': controlsState.rotateRight = 1; break;
        }
    });
    window.addEventListener('keyup', (event) => {
        if (document.activeElement === chatInput) return;
         switch (event.key.toLowerCase()) {
            case 'w': case 'arrowup': controlsState.forward = 0; break;
            case 's': case 'arrowdown': controlsState.backward = 0; break;
            case 'a': case 'arrowleft': controlsState.left = 0; break;
            case 'd': case 'arrowright': controlsState.right = 0; break;
            case 'q': controlsState.rotateLeft = 0; break;
            case 'e': controlsState.rotateRight = 0; break;
        }
    });

     // Mouse Look (Drag to rotate) - Use Left Button Drag
    gameCanvas.addEventListener('mousedown', (event) => {
        if (event.button === 0) { // Left mouse button
             mouseDrag.active = true;
             mouseDrag.x = event.clientX;
             mouseDrag.y = event.clientY; // Y not used for rotation here, but maybe later
             gameCanvas.style.cursor = 'grabbing';
        }
    });
     window.addEventListener('mouseup', (event) => { // Listen on window to catch mouseup outside canvas
        if (event.button === 0) {
            mouseDrag.active = false;
             gameCanvas.style.cursor = 'grab'; // Or 'default'
        }
    });
    window.addEventListener('mousemove', (event) => { // Listen on window
        if (mouseDrag.active) {
            const deltaX = event.clientX - mouseDrag.x;
            // const deltaY = event.clientY - mouseDrag.y; // Not used for Y rotation

            controlsState.rotateY = -deltaX * MOUSE_SENSITIVITY; // Adjust rotation based on horizontal drag

            // Update reference point
            mouseDrag.x = event.clientX;
            mouseDrag.y = event.clientY;
        } else {
            controlsState.rotateY = 0; // Stop rotation if mouse isn't dragged
        }
    });
    gameCanvas.addEventListener('mouseleave', () => { // Stop rotation if mouse leaves canvas while dragging
         if (mouseDrag.active) {
             // Optional: Stop rotation, or let it continue until mouseup? Let's stop it.
             // mouseDrag.active = false;
             // gameCanvas.style.cursor = 'grab';
             // controlsState.rotateY = 0;
         }
    });
    gameCanvas.style.cursor = 'grab'; // Initial cursor state


    // Gamepad
    window.addEventListener("gamepadconnected", (event) => {
        console.log("Gamepad connected:", event.gamepad);
        if (!activeGamepad) { // Take the first connected gamepad
            activeGamepad = event.gamepad;
        }
         addMessageToChat("system", `Gamepad connected: ${event.gamepad.id}`);
    });
    window.addEventListener("gamepaddisconnected", (event) => {
        console.log("Gamepad disconnected:", event.gamepad);
        if (activeGamepad && activeGamepad.index === event.gamepad.index) {
            activeGamepad = null;
            // Reset gamepad part of controlsState if needed
            controlsState.moveX = 0;
            controlsState.moveZ = 0;
            controlsState.rotateY = 0;
             addMessageToChat("system", `Gamepad disconnected: ${event.gamepad.id}`);
        }
    });

    // Touch / On-screen Joysticks (using NippleJS)
    if ('ontouchstart' in window || navigator.maxTouchPoints > 0) {
        joystickMoveZone.style.display = 'block';
        joystickRotateZone.style.display = 'block';

        const moveOptions = {
            zone: joystickMoveZone,
            mode: 'static',
            position: { left: '50%', top: '50%' },
            color: 'white',
            size: 100 // Match CSS zone size roughly
        };
        joystickMove = nipplejs.create(moveOptions);

        joystickMove.on('move', (evt, data) => {
            if (data.vector) {
                // Invert Y because screen Y is down, world Z is forward
                controlsState.moveX = data.vector.x;
                controlsState.moveZ = -data.vector.y;
            }
        });
        joystickMove.on('end', () => {
            controlsState.moveX = 0;
            controlsState.moveZ = 0;
        });

        const rotateOptions = {
            zone: joystickRotateZone,
            mode: 'static',
            position: { left: '50%', top: '50%' },
            color: 'white',
            size: 100
        };
        joystickRotate = nipplejs.create(rotateOptions);

        joystickRotate.on('move', (evt, data) => {
            if (data.vector) {
                // Use X axis of right stick for rotation
                controlsState.rotateY = -data.vector.x; // Negative for intuitive rotation
            }
        });
         joystickRotate.on('end', () => {
            controlsState.rotateY = 0;
        });

    } else {
         // Hide joystick zones if not a touch device
         joystickMoveZone.style.display = 'none';
         joystickRotateZone.style.display = 'none';
    }
}

function updateGamepad() {
    if (!activeGamepad) return;

    // Need to re-acquire the gamepad object on each frame
    const gamepads = navigator.getGamepads();
    const pad = Array.from(gamepads).find(p => p && p.index === activeGamepad.index);

    if (!pad) {
         activeGamepad = null; // Gamepad might have been disconnected without event
         return;
    }

    // Standard Gamepad Mapping (adjust indices if necessary)
    // Axes: 0=LeftStickX, 1=LeftStickY, 2=RightStickX, 3=RightStickY
    const leftStickX = pad.axes[0] || 0;
    const leftStickY = pad.axes[1] || 0;
    const rightStickX = pad.axes[2] || 0;
    // const rightStickY = pad.axes[3] || 0; // Often used for camera pitch, not implemented here

    // Deadzone
    const deadzone = 0.15;
    const applyDeadzone = (value) => Math.abs(value) < deadzone ? 0 : value;

    controlsState.moveX = applyDeadzone(leftStickX);
    controlsState.moveZ = applyDeadzone(-leftStickY); // Invert Y-axis
    controlsState.rotateY = applyDeadzone(-rightStickX); // Invert X-axis for rotation

    // Example Button Mapping (can add more)
    // Button 0 (A/Cross) - maybe jump later?
    // Button 1 (B/Circle)
    // etc.
}


// --- UI & Chat ---
function setupUI() {
    sendButton.addEventListener('click', () => {
        const message = chatInput.value.trim();
        if (message) {
            sendMessageToLLM(message);
            chatInput.value = '';
        }
    });

    chatInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !sendButton.disabled) {
            sendButton.click();
        }
    });

     modelSelect.addEventListener('change', () => {
         const selected = modelSelect.value;
         if (selected !== currentModel) {
             currentModel = selected;
             // Don't auto-reload, wait for button press
             // loadLLM();
         }
     });
     reloadButton.addEventListener('click', () => {
         currentModel = modelSelect.value; // Ensure currentModel is up-to-date
         console.log("Reload button clicked, loading model:", currentModel);
         loadLLM(); // Reload the selected model
     });

    temperatureSlider.addEventListener('input', () => {
        temperature = parseFloat(temperatureSlider.value);
        temperatureValue.textContent = temperature.toFixed(1);
    });
    // Set initial UI values
     modelSelect.value = currentModel;
     temperatureSlider.value = temperature;
     temperatureValue.textContent = temperature.toFixed(1);
}

function addMessageToChat(sender, message) {
    const messageElement = document.createElement('div');
    messageElement.classList.add('chat-message');
    let senderPrefix = "";

    switch (sender) {
        case 'user':
            messageElement.classList.add('user-message');
            senderPrefix = "You: ";
            break;
        case 'llm':
            messageElement.classList.add('llm-message');
            senderPrefix = "LLM: ";
            break;
        case 'tool-call':
            messageElement.classList.add('tool-call-message');
            senderPrefix = "Game> "; // Indicate system/game action
            break;
        case 'error':
             messageElement.classList.add('error-message');
             senderPrefix = "Error: ";
             break;
         case 'system': // For system messages like gamepad connect
             messageElement.classList.add('tool-call-message'); // Reuse style
             senderPrefix = "System: ";
             break;
    }

    // Basic Markdown Bold/Italics - very simple replacement
     message = message.replace(/\*\*(.*?)\*\*/g, '<b>$1</b>'); // Bold
     message = message.replace(/\*(.*?)\*/g, '<i>$1</i>');   // Italics

    messageElement.innerHTML = senderPrefix + message.replace(/\n/g, '<br>'); // Replace newlines
    chatHistoryDiv.appendChild(messageElement);

    // Auto-scroll to bottom
    chatHistoryDiv.scrollTop = chatHistoryDiv.scrollHeight;
}

function updateChatDisplay() {
    chatHistoryDiv.innerHTML = ''; // Clear existing messages
    // Rebuild display from stored history (if needed, e.g. after model reload)
    // Note: Current implementation adds messages incrementally, so this might only be needed
    // if we clear history on model reload.
    // chatHistory.forEach(msg => addMessageToChat(msg.role, msg.content));
}

function updateLLMStatus(statusText, isError = false) {
     llmStatusDiv.textContent = `LLM Status: ${statusText}`;
     llmStatusDiv.classList.remove('status-idle', 'status-thinking', 'status-error');
     if (isError) {
         llmStatusDiv.classList.add('status-error');
     } else if (isLLMThinking) {
         llmStatusDiv.classList.add('status-thinking');
     } else {
         llmStatusDiv.classList.add('status-idle');
     }
}

// --- Sound ---
function initSound() {
    try {
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
        // Define some basic sounds
        sounds.step = createSound(0.05, 'square', 80, 120, 0.01, 0.04);
        sounds.interact = createSound(0.1, 'sine', 440, 660, 0.01, 0.08);
        sounds.create = createSound(0.2, 'triangle', 660, 880, 0.02, 0.15);
        sounds.modify = createSound(0.15, 'sawtooth', 330, 220, 0.01, 0.1);
        sounds.llm_start = createSound(0.08, 'sine', 500, 700, 0.01, 0.05);
        sounds.llm_end = createSound(0.08, 'sine', 700, 500, 0.01, 0.05);
        sounds.tool_use = createSound(0.12, 'square', 1000, 500, 0.01, 0.08);
        sounds.error = createSound(0.3, 'sawtooth', 200, 100, 0.01, 0.25);
        console.log("AudioContext initialized.");
    } catch (e) {
        console.warn("Web Audio API is not supported or failed to initialize.", e);
        audioContext = null; // Disable sound
    }
}

// Factory for creating simple sound playback functions
function createSound(duration, type, freqStart, freqEnd, attack, decay) {
    return () => {
        if (!audioContext) return;
        const now = audioContext.currentTime;
        const oscillator = audioContext.createOscillator();
        const gainNode = audioContext.createGain();

        oscillator.type = type;
        oscillator.frequency.setValueAtTime(freqStart, now);
        oscillator.frequency.linearRampToValueAtTime(freqEnd, now + duration);

        gainNode.gain.setValueAtTime(0, now);
        gainNode.gain.linearRampToValueAtTime(0.5, now + attack); // Quick attack
        gainNode.gain.linearRampToValueAtTime(0, now + duration); // Decay over full duration

        oscillator.connect(gainNode);
        gainNode.connect(audioContext.destination);

        oscillator.start(now);
        oscillator.stop(now + duration);
    };
}

function playSound(soundName) {
    if (audioContext && sounds[soundName]) {
        // Resume context if suspended (e.g., due to user interaction policy)
         if (audioContext.state === 'suspended') {
             audioContext.resume();
         }
        sounds[soundName]();
    }
}

// --- Game Loop ---
function animate() {
    requestAnimationFrame(animate);
    const delta = clock.getDelta();

    updateGamepad(); // Poll gamepad state
    updatePlayer(delta); // Update player movement and camera

    renderer.render(scene, camera);
}

function updatePlayer(delta) {
    const moveSpeed = PLAYER_SPEED * delta;
    const rotateSpeed = PLAYER_ROTATION_SPEED * delta;

    // --- Combine Inputs ---
    // Keyboard / Button Input (Prioritized if held)
    let moveZ = controlsState.forward - controlsState.backward;
    let moveX = controlsState.right - controlsState.left;
    let rotateY = controlsState.rotateRight - controlsState.rotateLeft;

    // Add Analog/Joystick Input (if buttons aren't pressed)
    if (moveZ === 0) moveZ = controlsState.moveZ;
    if (moveX === 0) moveX = controlsState.moveX;
    if (rotateY === 0) rotateY += controlsState.rotateY; // Add mouse/joystick rotation

    // Rotation
    if (Math.abs(rotateY) > 0.01) { // Add small threshold
        playerAvatar.rotation.y -= rotateY * rotateSpeed * (rotateY === controlsState.rotateY ? 30 : 1); // Make mouse/joystick rotation faster
        playSound('step'); // Sound for turning
    }

    // Movement
     if (Math.abs(moveX) > 0.01 || Math.abs(moveZ) > 0.01) {
        // Calculate movement direction based on player's current rotation
        const moveDirection = new THREE.Vector3(moveX, 0, moveZ);
        moveDirection.normalize();
        moveDirection.applyQuaternion(playerAvatar.quaternion); // Apply player's rotation

        playerAvatar.position.addScaledVector(moveDirection, moveSpeed);
         playSound('step'); // Sound for moving
    }

    // --- Camera Follow ---
    // Simple third-person camera: offset behind and slightly above the player
    const cameraOffset = new THREE.Vector3(0, 2.5, 5.0); // Adjust as needed (x, y, z offset)
    const cameraTarget = new THREE.Vector3();

    // Apply the player's rotation to the offset vector
    cameraOffset.applyQuaternion(playerAvatar.quaternion);
    // Calculate the desired camera position
    cameraTarget.copy(playerAvatar.position).add(cameraOffset);

    // Smoothly interpolate camera position and lookAt
     camera.position.lerp(cameraTarget, 0.1); // Adjust lerp factor (0.1 = smooth, 1.0 = instant)
     const lookAtTarget = playerAvatar.position.clone().add(new THREE.Vector3(0, 0.8, 0)); // Look slightly above the player's base
     camera.lookAt(lookAtTarget);

     // Reset mouse delta rotation after applying it
     controlsState.rotateY = 0;
}

// --- Start ---
document.addEventListener('DOMContentLoaded', init);
