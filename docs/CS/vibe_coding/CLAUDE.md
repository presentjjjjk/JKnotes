# Project Overview

This is a documentation project about **Claude Code**, a CLI tool for Claude AI, maintained as part of a larger CS documentation collection under `my-project/docs/CS/vibe_coding/`.

## Project Structure

```
vibe_coding/
├── claude_code.md    # Main documentation about Claude Code installation and usage
├── image.png         # Model mapping reference (Claude models to Zhipu AI models)
└── CLAUDE.md         # This file - project context description
```

## Content Summary

The main documentation (`claude_code.md`) covers:

### 1. Installation
- Requires Node.js
- Install via: `npm install -g @anthropic-ai/claude-code`

### 2. Configuration Methods
Three methods to configure API keys:

**Method 1: Environment Variables**
- `ANTHROPIC_BASE_URL` - Model provider's base URL
- `ANTHROPIC_API_KEY` - API key
- `HTTP_PROXY` / `HTTPS_PROXY` - For overseas models requiring VPN

**Method 2: settings.json File**
- Located at: `~/.claude/settings.json`
- Configure auth token, base URL, timeout, and optional model renaming

**Method 3: claude-code-router**
- For models not directly compatible with Claude Code
- GitHub: https://github.com/musistudio/claude-code-router
- Install via: `npm install -g @musistudio/claude-code-router`
- Config file: `~/claude-code-router/config.json`
- Example uses ModelScope API (2000 free Qwen-3-coder calls daily with Aliyun account)

### 3. Important Commands
| Command | Description |
|---------|-------------|
| `\init` | Read all files in current directory and generate CLAUDE.md project description |
| `\compact` | Compress context to reduce token usage |
| `\clear` | Clear conversation context |
| `think/harder/ultrathink` | Prefixes to control model thinking depth |
| `!` | Enter temporary command-line mode |

## Model Mapping (Zhipu AI)

Based on the reference image:

| Claude Model | Zhipu AI Model |
|--------------|----------------|
| Haiku | GLM-4.5-Air / GLM-4-Air |
| Sonnet | GLM-4.7 |
| Opus | GLM-4.7 |

Can be customized via environment variables:
- `ANTHROPIC_DEFAULT_HAIKU_MODEL`
- `ANTHROPIC_DEFAULT_SONNET_MODEL`
- `ANTHROPIC_DEFAULT_OPUS_MODEL`

## Notes

- Documentation is written in Chinese
- Part of a larger personal project structure
- Focuses on practical setup and usage instructions for Chinese users
