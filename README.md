# LLM_Plays_3D

https://neuroidss.github.io/LLM_Plays_3D

![llm in 3d creates red cune]|(https://github.com/neuroidss/LLM_Plays_3D/blob/main/Screencast_from_2025-03-29_22-28-54.gif?raw=true)

```
make game where llm controlling 3d world and responding to interaction with user avatar, so llm playing with user. game for llm should be in continuous messages context with ability of interaction with user in chat. use tools calling. don't invent fake moves when llm error, try again until correct llm response. use threejs >=0.140.0. make keyboard&mouse, real gamepad and multitouch onscreen gamepad joysticks controls for 3d move and 3d rotate, should be desktop and mobile ready. make sure not using browser alert() popups. add some sounds, but without external sound resources.

include 2025 Vibe Coding Game Jam participation note:
<a target="_blank" href="https://jam.pieter.com" style="font-family: 'system-ui', sans-serif; position: fixed; bottom: -1px; right: -1px; padding: 7px; font-size: 14px; font-weight: bold; background: #fff; color: #000; text-decoration: none; z-index: 10; border-top-left-radius: 12px; z-index: 10000; border: 1px solid #fff;"> Vibe Jam 2025</a>

add llm chat with custom tool calling, as native tool calling not yet supported in webllm for qwen.

use web-llm. 
<script type="module">
import * as webllm from "https://esm.run/@mlc-ai/web-llm";
                engine = await webllm.CreateMLCEngine(
                    "Qwen2.5-Coder-7B-Instruct-q4f32_1-MLC",
                    { initProgressCallback: reportProgress }
                );
                don't use appConfig.
                with reportProgress use progress.progress if needed.
use engine.chat.completions.create but without its native tools feature, indtead use custom tools, stream: false, also set temperature in it.

add in llm tools list for chatbot llm tool_creation_tool with parameters 'name' and 'description' and which calls engine chat create to create js function which makes what in 'description' and named as 'name', and evaluates this new js function, then creates llm tool with same name and description and parameters which was used in new created js function, and adding new llm tool in available llm tools list for chatbot, js function should take a single parameters object as named array. use another role instead of role tool. as system prompt can be placed only on first message, for tools declaration use assistant prompt but place it each time before new user message in history, but don't keep this tools usage message in history as it will be always updated. last message should be from user. don't use system role anywhere but first message. for created tools js functions and json parameters use only code inside blocks ```javascript ```, ```json ```, remove these block markers. if using, json block for tools use detection then remove ```json ``` block markers before parsing.

make ability to switch between Qwen2.5-Coder-7B-Instruct-q4f32_1-MLC and Qwen2.5-Coder-3B-Instruct-q4f32_1-MLC, and set temperature of generation.

initialize after all elements created.
```
