# Configuring GitHub Copilot Chat in VS Code

## Prerequisites

Before setting up GitHub Copilot Chat in VS Code, ensure you have the following requirements:

- **Visual Studio Code**: Latest version installed[1]
- **GitHub Account**: Active GitHub account with Copilot access[2][3]
- **GitHub Copilot Subscription**: Either Copilot Free, Pro, or Pro+ plan[4]
- **VS Code version 17.10 or later** for full feature compatibility[3]

## Installation and Setup

### Step 1: Install the Extensions

Open VS Code and install the required extensions[5]:

1.  Press `Ctrl+Shift+X` (Windows/Linux) or `Cmd+Shift+X` (Mac) to open the Extensions view
2.  Search for "**GitHub Copilot**" and install it
3.  The "**GitHub Copilot Chat**" extension will be automatically installed alongside it[6]

Both extensions are official GitHub extensions that provide complementary functionality - the main Copilot extension handles code completions while Copilot Chat supports conversational AI interactions[5].

### Step 2: Authentication

Once extensions are installed, you'll need to authenticate with GitHub[2][1]:

1.  VS Code will prompt you to sign in when you first try to use Copilot
2.  Select **Set up Copilot** when hovering over the Copilot icon in the Status Bar[1]
3.  Choose your preferred sign-in method and follow the prompts
4.  Click **Allow** when asked to authorize the GitHub Copilot extension[2]
5.  Complete the GitHub authentication process in your browser
6.  Click **Open Visual Studio Code** when prompted to return to VS Code[2]

### Step 3: Access Copilot Chat

After authentication, you can access Copilot Chat in several ways[7][8]:

- **Chat View**: Select **Open Chat** from the Chat menu in the title bar or use `Ctrl+Alt+I` (Windows/Linux) or `⌃⌘I` (Mac)[7]
- **Inline Chat**: Use `Ctrl+I` (Windows/Linux) or `⌘I` (Mac) directly in the editor[9]
- **Command Palette**: Type `/mcp` and `/tools` to see available Copilot tools[10]

## Configuration Options

### Chat Modes

Copilot Chat operates in different modes optimized for specific use cases[7][8]:

- **Ask Mode**: For questions about coding concepts and explanations
- **Edit Mode**: For making code modifications and improvements
- **Agent Mode**: For autonomous multi-step coding workflows[11]

### Customization Features

**Custom Instructions**: Create project-specific guidelines by adding a `.github/copilot-instructions.md` file to your project root[12]:

```markdown
# Project general coding guidelines

## Code Style

- Use semantic HTML5 elements
- Prefer modern JavaScript (ES6+) features

## Naming Conventions

- Use PascalCase for components
- Use camelCase for variables and functions
```

**Settings Configuration**: Access Copilot settings through VS Code Settings (`Ctrl+,` or `Cmd+,`) and search for "copilot"[9]. Key settings include:

- `github.copilot.enable`: Enable/disable code completions for specific languages
- `github.copilot.chat.localeOverride`: Specify response language
- `chat.agent.enabled`: Enable agent mode functionality[11]

## Model Context Protocol (MCP) Integration

The Morph documentation[2] and VS Code now support full MCP specification[10][13], which enables enhanced functionality:

**MCP Server Setup**: Configure MCP servers to extend Copilot's capabilities with external tools and data sources[10]. This allows Copilot to access databases, APIs, and other services through a standardized interface.

**Security**: MCP includes authorization specifications developed collaboratively with Microsoft, Anthropic, and identity providers like Okta/Auth0[13], ensuring enterprise-grade security for remote MCP servers.

## Advanced Features

### Voice Interactions

Enable voice control capabilities with the VS Code Speech extension[8]:

- Use voice to dictate chat prompts
- Activate "Hey Code" voice commands
- Enable "hold to speak" mode for faster voice input

### Chat Sessions and Agent Mode

**Agent Mode Configuration**[11]:

- Maximum requests: 5 for Copilot Free users, 15 for paid users
- Auto-fix functionality enabled by default
- Terminal command auto-approval settings for security

**Multiple Chat Sessions**: Open chat in separate editor tabs or floating windows for simultaneous conversations[8].

## Troubleshooting

Common issues and solutions[2][5]:

- **Server won't start**: Check API key validity, ensure Node.js 16+, run `npm cache clean --force`
- **Authentication issues**: Remove existing GitHub authentication in VS Code settings and re-authenticate[14]
- **Missing tools**: Restart VS Code and validate JSON configuration
- **Slow performance**: Prefer `edit_file` over `write_file` for modifications[2]

GitHub Copilot Chat in VS Code provides a comprehensive AI-powered development experience with extensive customization options, security features, and integration capabilities through the Model Context Protocol framework.

### Sources

[1] Set up GitHub Copilot in VS Code https://code.visualstudio.com/docs/copilot/setup
[2] Step-by-Step: How to Setup Copilot Chat in VS Code https://techcommunity.microsoft.com/t5/educator-developer-blog/step-by-step-how-to-setup-copilot-chat-in-vs-code/ba-p/3849227
[3] Customize chat responses - Visual Studio (Windows) https://learn.microsoft.com/en-us/visualstudio/ide/copilot-chat-context?view=vs-2022
[4] Quickstart for GitHub Copilot https://docs.github.com/copilot/quickstart
[5] Configure GitHub Copilot in VSCode with a Privacy-First ... https://paulsorensen.io/github-copilot-vscode-privacy/
[6] GitHub Copilot Chat https://marketplace.visualstudio.com/items?itemName=GitHub.copilot-chat
[7] Getting started with chat in VS Code https://code.visualstudio.com/docs/copilot/chat/getting-started-chat
[8] Use chat in VS Code https://code.visualstudio.com/docs/copilot/chat/copilot-chat
[9] GitHub Copilot in VS Code settings reference https://code.visualstudio.com/docs/copilot/reference/copilot-settings
[10] Use MCP servers in VS Code https://code.visualstudio.com/docs/copilot/customization/mcp-servers
[11] Use agent mode in VS Code https://code.visualstudio.com/docs/copilot/chat/chat-agent-mode
[12] Get started with GitHub Copilot in VS Code https://code.visualstudio.com/docs/copilot/getting-started
[13] The Complete MCP Experience: Full Specification Support ... https://code.visualstudio.com/blogs/2025/06/12/full-mcp-spec-support
[14] copilot in vscode setup fail and can't sign in #158947 https://github.com/orgs/community/discussions/158947
[15] Customize chat to your workflow https://code.visualstudio.com/docs/copilot/customization/overview
[16] You can use Copilot Extensions to interact with external ... https://docs.github.com/copilot/using-github-copilot/using-extensions-to-integrate-external-tools-with-copilot-chat
[17] GitHub Copilot Setup in VS Code: Complete Installation & ... https://www.youtube.com/watch?v=wxaxlIlN7BA
[18] microsoft/vscode-copilot-chat https://github.com/microsoft/vscode-copilot-chat
[19] About GitHub Copilot Chat in Visual Studio https://learn.microsoft.com/en-us/visualstudio/ide/visual-studio-github-copilot-chat?view=vs-2022
[20] Ask, Edit, & Agent - In-depth Overview of GitHub Copilot ... https://www.youtube.com/watch?v=s7Qzq0ejhjg
[21] How to enable and disable GitHub Copilot in Visual Studio ... https://www.reddit.com/r/vscode/comments/1hrsiek/how_to_enable_and_disable_github_copilot_in/
[22] Installing the GitHub Copilot extension in your environment https://docs.github.com/copilot/managing-copilot/configure-personal-settings/installing-the-github-copilot-extension-in-your-environment
[23] Streamlined coding, debugging, and testing with ... https://github.blog/changelog/2024-10-03-streamlined-coding-debugging-and-testing-with-github-copilot-chat-in-vs-code/
[24] Asking GitHub Copilot questions in your IDE https://docs.github.com/copilot/using-github-copilot/asking-github-copilot-questions-in-your-ide
[25] Copilot Chat in Visual Studio Code | GitHub Universe https://www.youtube.com/watch?v=a2DDYMEPwbE
[26] Configuring network settings for GitHub Copilot https://docs.github.com/copilot/configuring-github-copilot/configuring-network-settings-for-github-copilot
[27] MCP developer guide | Visual Studio Code Extension API https://code.visualstudio.com/api/extension-guides/ai/mcp
[28] Model Context Protocol (MCP) support in VS Code is ... https://github.blog/changelog/2025-07-14-model-context-protocol-mcp-support-in-vs-code-is-generally-available/
[29] GitHub Copilot Chat https://docs.github.com/en/copilot/how-tos/chat-with-copilot
[30] Configure GitHub Copilot in VS Code - Domino Documentation https://docs.dominodatalab.com/en/latest/user_guide/00f51f/configure-github-copilot-in-vs-code/
[31] Architecture overview https://modelcontextprotocol.io/docs/learn/architecture
[32] Discover and install MCP Servers in VS Code https://code.visualstudio.com/mcp
[33] Set up Visual Studio Code with Copilot https://code.visualstudio.com/docs/copilot/setup-simplified
