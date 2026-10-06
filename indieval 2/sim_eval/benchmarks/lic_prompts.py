"""Pinned Lost in Conversation assistant prompts.

Source: https://github.com/microsoft/lost_in_conversation/tree/c865793fe34a929d316119b0451d01bd9183bcfd
The code task uses prompts/lcb/lcb_system_prompt.txt for every code source,
including HumanEval (TaskCode.generate_system_prompt). No task payload is
interpolated into either assistant system prompt.

    MIT License

    Copyright (c) Microsoft Corporation.

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE

"""

LIC_SOURCE_REVISION = 'c865793fe34a929d316119b0451d01bd9183bcfd'
MATH_SYSTEM_PROMPT = 'As an expert problem solver solve step by step the following mathematical questions.'
CODE_SYSTEM_PROMPT = 'You are an expert Python programmer. You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests.\n\nFormat:\n- [Standalone] Make sure that your answer consists of only one Python function at the top level. Do not wrap with a class or split into multiple functions.'

PROMPT_SHA256 = {'math': 'db8fe01172e7c2f919763dcb26b253a20b0d6eb17fdb2aae198d681dbb1a65e7', 'code': '8a7f647e7ca3ba931986147ddfb50e3e32ac8c66ca9857d2d9a5eada157f807f'}
